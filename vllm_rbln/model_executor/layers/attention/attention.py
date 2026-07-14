# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import vllm.model_executor.layers.attention.attention as vllm_attn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_rbln.v1.attention.kv_cache_bindings import materialize_kv_cache_view
from vllm_rbln.v1.attention.prefill_aware_swa import (
    detect_prefill_aware_swa_status_from_vllm_config,
)
from vllm_rbln.v1.kv_cache import (
    RBLNPrefillAwareSlidingWindowSpec,
    RBLNSlidingWindowSpec,
)

# ---------------------------------------------------------------------------
# Snapshots of upstream implementations (used by RBLN overrides)
# ---------------------------------------------------------------------------
_upstream_init = Attention.__init__
_upstream_get_kv_cache_spec = Attention.get_kv_cache_spec


def _rbln_attention_init(self, *args, **kwargs) -> None:
    _upstream_init(self, *args, **kwargs)

    # NOTE(jiwoo.park) layer index is required to use external binding KV cache.
    self.layer_index = extract_layer_index(self.layer_name)

    # NOTE(RBLN) - consider PP
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    if model_config is not None:
        start, _end = model_config.get_layers_start_end_indices(
            vllm_config.parallel_config
        )
        self.layer_index -= start


def _resolve_kv_cache(
    attn_metadata, layer_index: int, embedded_kv_cache: torch.Tensor | None = None
) -> torch.Tensor:
    """Resolve the KV cache for a given layer, either from deduplicated
    base tensors (for torch.export compatibility) or from the direct list."""
    kv_cache_view_infos = getattr(attn_metadata, "kv_cache_view_infos", None)
    forward_context = get_forward_context()
    additional_kwargs = getattr(forward_context, "additional_kwargs", {}) or {}
    kv_cache_bases = additional_kwargs.get("kv_cache_bases")
    if kv_cache_bases and kv_cache_view_infos:
        assert layer_index < len(kv_cache_view_infos)
        return materialize_kv_cache_view(
            kv_cache_bases, kv_cache_view_infos[layer_index]
        )

    if embedded_kv_cache is not None:
        return embedded_kv_cache

    assert attn_metadata.kv_caches is not None
    assert layer_index < len(attn_metadata.kv_caches)
    return attn_metadata.kv_caches[layer_index]


def _rbln_unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    attn_metadata, self, _kv_cache, _ = get_attention_context(layer_name)

    # NOTE(RBLN) - To represent kv cache as model input,
    # modify attention instead of using the attention layer's embedded
    # kv cache (self.kv_cache); use attention metadata's kv cache.
    # attention metadata's kv cache must equal the attention layer's
    # embedded kv cache.
    kv_cache = _resolve_kv_cache(attn_metadata, self.layer_index, _kv_cache)

    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)
    return output


def _rbln_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    attn_metadata, self, _kv_cache, _ = get_attention_context(layer_name)

    # NOTE(RBLN) - To represent kv cache as model input,
    # modify attention instead of using the attention layer's embedded
    # kv cache (self.kv_cache); use attention metadata's kv cache.
    # attention metadata's kv cache must equal the attention layer's
    # embedded kv cache.
    kv_cache = _resolve_kv_cache(attn_metadata, self.layer_index, _kv_cache)
    result = self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )
    # vLLM's Attention.forward now allocates `output` with torch.empty()
    # (uninitialized) and treats unified_attention_with_output as a
    # side-effecting op that must fill it in place -- it ignores the return
    # value and hands `output` straight back to the caller. RBLNFlashAttentionImpl
    # (like most attention impls) still *returns* its result instead of writing
    # `output`, so without this copy vLLM would propagate the uninitialized
    # buffer downstream: garbage/NaN that varies run-to-run. Mirror the result
    # into `output` when the impl didn't already write it.
    if output is not None and result is not None and result is not output:
        output.copy_(result.reshape(output.shape))


def _rbln_get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    # Block size may get updated after model loading, refresh it
    block_size = vllm_config.cache_config.block_size
    # Should not be called for enc-dec or encoder-only attention.
    assert self.attn_type == AttentionType.DECODER

    pa_swa_status = detect_prefill_aware_swa_status_from_vllm_config(vllm_config)
    if pa_swa_status.required:
        sliding_window = pa_swa_status.sliding_window
        if sliding_window is None:
            raise NotImplementedError(
                "RBLN PA-SWA is required but the checkpoint/config does not "
                "expose sliding_window_size/sliding_window."
            )
        if not pa_swa_status.backend_ready:
            raise NotImplementedError(
                "RBLN PA-SWA backend is not ready. Refusing to create a "
                "FullAttentionSpec or regular RBLNSlidingWindowSpec because "
                "Unlimited-OCR decode requires all prefill KV plus the latest "
                "checkpoint-derived sliding-window decode KV."
            )
        assert not vllm_config.model_config.use_mla, (
            "MLA is not supported for PA-SWA slidingwindow"
        )
        return RBLNPrefillAwareSlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            sliding_window=sliding_window,
        )

    if self.sliding_window is not None:
        assert not vllm_config.model_config.use_mla, (
            "MLA is not supported for slidingwindow"
        )
        return RBLNSlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            sliding_window=self.sliding_window,
        )
    else:
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
        )


vllm_attn.unified_attention = maybe_transfer_kv_layer(_rbln_unified_attention)
vllm_attn.unified_attention_with_output = maybe_transfer_kv_layer(
    _rbln_unified_attention_with_output
)

Attention.__init__ = _rbln_attention_init
Attention.get_kv_cache_spec = _rbln_get_kv_cache_spec
