"""Neutral runtime facade for the Unlimited-OCR implementation."""

from __future__ import annotations

import contextlib
import copy
import inspect
import io
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .metrics import RuntimeMetrics
from .processor import OCRInputs, prepare_single_image_inputs
from .rbln_compile import (
    available_rbln_device_ids,
    compile_embed_tokens_shapes,
    compile_projector_shapes,
    compile_sam_shapes,
    compile_vision_signatures,
    rbln_compile_options,
    rbln_num_devices,
)

_UNLIMITED_OCR_CONFIG_REGISTERED = False


def _gpt2_unicode_to_byte() -> dict[str, int]:
    """GPT-2 byte-level BPE unicode->byte map (stable, standard function).

    Unlimited-OCR's tokenizer is byte-level BPE: spaces/newlines/raw bytes are
    encoded as printable unicode surrogates (space -> 'Ġ' U+0120, newline ->
    'Ċ' U+010A, etc.). transformers 5.8's fast tokenizer for this checkpoint no
    longer applies the ByteLevel decoder step, so `tokenizer.decode()` leaves
    those surrogates literal (and `encode` drops spaces). Reconstruct the map
    ourselves so detokenization is correct regardless of the library's decoder
    wiring.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_UNICODE_TO_BYTE = _gpt2_unicode_to_byte()


def _decode_bytelevel(tokenizer: Any, output_ids: list[int]) -> str:
    """Detokenize byte-level BPE ids, preserving special tokens verbatim.

    Groups runs of ordinary (byte-level) tokens and maps their surrogate
    characters back to raw bytes before UTF-8 decoding, while rendering
    special/added tokens (``<|det|>``, ``<｜end of sentence｜>`` etc.) through
    the tokenizer so their exact surface form is kept.
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    pieces: list[str] = []
    buf: list[str] = []

    def _flush() -> None:
        if buf:
            data = bytes(_UNICODE_TO_BYTE.get(ch, 0x3F) for ch in "".join(buf))
            pieces.append(data.decode("utf-8", errors="replace"))
            buf.clear()

    for tid in output_ids:
        tid = int(tid)
        token = tokenizer.convert_ids_to_tokens(tid)
        is_special = tid in special_ids or (
            isinstance(token, str) and (token.startswith("<|") or token.startswith("<｜"))
        )
        if is_special:
            _flush()
            pieces.append(tokenizer.decode([tid], skip_special_tokens=False))
        else:
            buf.append(token if isinstance(token, str) else "")
    _flush()
    return "".join(pieces)



def _ensure_unlimited_ocr_config_registered() -> None:
    """Register a vLLM-native config class for ``model_type="unlimited-ocr"``.

    Baidu's own HF checkpoint only exposes this shape via remote code
    (``auto_map`` -> ``modeling_unlimitedocr.UnlimitedOCRConfig``), and that
    remote module unconditionally imports ``modeling_deepseekv2.py`` (written
    for transformers 4.x) even when only the config class is needed. Avoiding
    ``trust_remote_code`` entirely -- for config loading too, not just model
    weights -- means registering our own class into vLLM's config registry,
    mirroring the (not-yet-released) official vLLM `UnlimitedOCRConfig`.
    ``DeepseekVLV2Config`` already knows how to build ``text_config`` /
    ``vision_config`` / ``projector_config`` from the raw nested dicts.
    """

    global _UNLIMITED_OCR_CONFIG_REGISTERED
    if _UNLIMITED_OCR_CONFIG_REGISTERED:
        return
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config

    if "unlimited-ocr" not in _CONFIG_REGISTRY:

        class UnlimitedOCRConfig(DeepseekVLV2Config):
            model_type = "unlimited-ocr"

        _CONFIG_REGISTRY["unlimited-ocr"] = UnlimitedOCRConfig
    _UNLIMITED_OCR_CONFIG_REGISTERED = True


def _load_unlimited_ocr_raw_weights(
    model_name_or_path: str,
) -> dict[str, torch.Tensor]:
    """Load raw checkpoint weights directly, without any remote code.

    Handles both the single-safetensors-file layout (current baidu/Unlimited-OCR
    checkpoint) and a sharded layout with a ``model.safetensors.index.json``.
    """

    from pathlib import Path

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if Path(model_name_or_path).is_dir():
        local_dir = Path(model_name_or_path)
    else:
        local_dir = Path(hf_hub_download(model_name_or_path, "config.json")).parent

    index_path = local_dir / "model.safetensors.index.json"
    if index_path.exists():
        import json

        shard_names = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        weights: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            shard_path = local_dir / shard_name
            if not shard_path.exists():
                shard_path = Path(hf_hub_download(model_name_or_path, shard_name))
            weights.update(load_file(shard_path))
        return weights

    single_file = local_dir / "model.safetensors"
    if not single_file.exists():
        single_file = Path(hf_hub_download(model_name_or_path, "model.safetensors"))
    return load_file(single_file)


@dataclass(frozen=True)
class ExecutionPlan:
    """Backend selection for one Unlimited-OCR runtime invocation."""

    vision_backends: tuple[str, ...] = ()
    language_backend: str = "cpu"

    @classmethod
    def from_cli(
        cls,
        rbln_components: set[str],
        *,
        language_backend: str,
    ) -> ExecutionPlan:
        return cls(
            vision_backends=tuple(sorted(rbln_components)),
            language_backend=language_backend,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vision_backends": list(self.vision_backends),
            "language_backend": self.language_backend,
        }


@dataclass(frozen=True)
class RuntimeBackendSummary:
    """Observed backend labels after optional component compilation."""

    sam_model: str
    vision_model: str
    projector: str
    embed_tokens: str
    language_decode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sam_model": self.sam_model,
            "vision_model": self.vision_model,
            "projector": self.projector,
            "embed_tokens": self.embed_tokens,
            "language_decode": self.language_decode,
        }


@dataclass
class LoadedUnlimitedOCR:
    """Loaded tokenizer/model pair plus structured facade."""

    tokenizer: Any
    native_model: Any
    facade: UnlimitedOCRFacade


class ObservedRBLNCallable:
    """Count calls through compiled components installed into the HF path."""

    def __init__(self, component: str, module: Any, counts: dict[str, int]) -> None:
        self.component = component
        self.module = module
        self.counts = counts

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.counts[self.component] = self.counts.get(self.component, 0) + 1
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)


class UnlimitedOCRFacade:
    """Stable Unlimited-OCR execution boundaries.

    A vLLM-native ``UnlimitedOCRForCausalLM`` (vision towers + projector +
    DeepSeek-V2 language backbone, all vLLM-authored, no HF remote code) is
    the source of truth. The facade exposes explicit backend-selectable
    chunks: vision frontend, multimodal projection/layout, token embeddings,
    decoder stack/norm, and lm_head.
    """

    def __init__(
        self,
        native_model: Any,
        tokenizer: Any,
        raw_weights: dict[str, torch.Tensor] | None = None,
        model_name_or_path: str = "baidu/Unlimited-OCR",
        vllm_config: Any | None = None,
    ) -> None:
        self.native_model = native_model
        self.tokenizer = tokenizer
        self.raw_weights = raw_weights
        self.model_name_or_path = model_name_or_path
        self.vllm_config = vllm_config
        self.root_model = native_model
        self.sam_model = native_model.sam_model
        self.vision_model = native_model.vision_model
        self.projector = native_model.projector
        self.image_newline = native_model.image_newline
        self.view_seperator = native_model.view_seperator
        self.sam_backend = "cpu"
        self.vision_backend = "cpu"
        self.projector_backend = "cpu"
        self.embed_tokens_backend = "cpu"
        language_model = native_model.language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.hidden_size = int(language_model.model.embed_tokens.embedding_dim)
        self.layers = language_model.model.layers
        self.norm = language_model.model.norm
        self.lm_head = language_model.lm_head
        self._active_metrics: RuntimeMetrics | None = None
        self._rbln_call_counts: dict[str, int] = {}
        self._rbln_language_runner: NativeRBLNLanguagePrefillRunner | None = None
        self._rbln_language_runner_lock = threading.Lock()

    def backend_summary(self) -> RuntimeBackendSummary:
        """Return current backend labels without implying full decoder NPU support."""

        return RuntimeBackendSummary(
            sam_model=self.sam_backend,
            vision_model=self.vision_backend,
            projector=self.projector_backend,
            embed_tokens=self.embed_tokens_backend,
            language_decode="cpu",
        )

    def module_summary(self) -> dict[str, dict[str, Any]]:
        """Return stable module-boundary metadata for inspection output."""

        return {
            "sam_model": _module_info(self.sam_model),
            "vision_model": _module_info(self.vision_model),
            "projector": _module_info(self.projector),
            "embed_tokens": _module_info(self.embed_tokens),
            "decoder_layers": _module_info(self.layers),
            "norm": _module_info(self.norm),
            "lm_head": _module_info(self.lm_head),
        }

    def enable_rbln_components(
        self,
        components: set[str],
        inputs: "OCRInputs | list[OCRInputs]",
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        """Compile selected components for one or more prepared input shapes.

        ``inputs`` may be a single ``OCRInputs`` or a list. Every component is
        compiled from its ORIGINAL module for the union of all input
        signatures, so the installed signature dispatcher can serve every
        requested prompt length / image shape. Compiling from the original
        (not the already-compiled wrapper) is what makes repeated / multi-shape
        enable safe: callers may re-invoke this with a growing input list as new
        shapes appear and each component is rebuilt to cover the full set.
        """
        inputs_list = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        if not inputs_list:
            return
        if getattr(self, "_frontend_origin", None) is None:
            self._frontend_origin: dict[str, Any] = {}
        for comp in ("sam_model", "vision_model", "projector", "embed_tokens"):
            if comp in components and comp not in self._frontend_origin:
                self._frontend_origin[comp] = getattr(self, comp)
        dtype = next(self.native_model.parameters()).dtype
        input_summary = inputs_list[0].shape_summary()

        def _union(method: Any) -> list[Any]:
            collected: list[Any] = []
            for inp in inputs_list:
                collected.extend(method(inp))
            return collected

        if "sam_model" in components:
            try:
                compiled_sam = compile_sam_shapes(
                    self._frontend_origin["sam_model"],
                    _union(self._image_shapes_for_inputs), dtype
                )
            except Exception as exc:
                _raise_rbln_compile_error("sam_model", input_summary, exc)
            self.sam_backend = compiled_sam.backend
            observed_sam = self._observe_rbln_component(
                "sam_model", compiled_sam.module
            )
            self.sam_model = observed_sam
            _set_runtime_attr(self.root_model, "sam_model", observed_sam)
        if "vision_model" in components:
            try:
                compiled_vision = compile_vision_signatures(
                    self._frontend_origin["vision_model"],
                    _union(self._vision_signatures_for_inputs), dtype
                )
            except Exception as exc:
                _raise_rbln_compile_error("vision_model", input_summary, exc)
            self.vision_backend = compiled_vision.backend
            observed_vision = self._observe_rbln_component(
                "vision_model", compiled_vision.module
            )
            self.vision_model = observed_vision
            _set_runtime_attr(self.root_model, "vision_model", observed_vision)
        if "projector" in components:
            try:
                compiled_projector = compile_projector_shapes(
                    self._frontend_origin["projector"],
                    _union(self._projector_shapes_for_inputs), dtype
                )
            except Exception as exc:
                _raise_rbln_compile_error("projector", input_summary, exc)
            self.projector_backend = compiled_projector.backend
            observed_projector = self._observe_rbln_component(
                "projector", compiled_projector.module
            )
            self.projector = observed_projector
            _set_runtime_attr(self.root_model, "projector", observed_projector)
        if "embed_tokens" in components:
            try:
                compiled_embed = compile_embed_tokens_shapes(
                    self._frontend_origin["embed_tokens"],
                    _union(self._embed_tokens_shapes_for_inputs)
                )
            except Exception as exc:
                _raise_rbln_compile_error("embed_tokens", input_summary, exc)
            self.embed_tokens_backend = compiled_embed.backend
            observed_embed = self._observe_rbln_component(
                "embed_tokens", compiled_embed.module
            )
            self.embed_tokens = observed_embed
            # Reassigning self.embed_tokens does not touch
            # native_model.language_model.model.embed_tokens (a separately
            # constructed DeepseekV2ForCausalLM instance): the PA-SWA language
            # runner reloads decoder weights from that submodule's own
            # state_dict(), independent of whatever the facade's
            # self.embed_tokens alias currently points to for prompt-embedding
            # construction.

    def rbln_call_counts(self) -> dict[str, int]:
        return dict(self._rbln_call_counts)

    def _observe_rbln_component(self, component: str, module: Any) -> Any:
        self._rbln_call_counts.setdefault(component, 0)
        return ObservedRBLNCallable(component, module, self._rbln_call_counts)

    def _image_tensors_for_inputs(self, inputs: OCRInputs) -> list[torch.Tensor]:
        patches, image_ori = inputs.images[0]
        if torch.sum(patches).item() != 0:
            return [patches, image_ori]
        return [image_ori[idx : idx + 1] for idx in range(image_ori.shape[0])]

    def _image_shapes_for_inputs(
        self, inputs: OCRInputs
    ) -> list[tuple[int, int, int, int]]:
        return [
            _image_shape_tuple(images)
            for images in self._image_tensors_for_inputs(inputs)
        ]

    def _vision_signatures_for_inputs(
        self, inputs: OCRInputs
    ) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
        return [
            (image_shape, _sam_output_shape_from_image_shape(image_shape))
            for image_shape in self._image_shapes_for_inputs(inputs)
        ]

    def _projector_shapes_for_inputs(
        self, inputs: OCRInputs
    ) -> list[tuple[int, int, int]]:
        return [
            _projector_shape_from_image_tensor(images)
            for images in self._image_tensors_for_inputs(inputs)
        ]


    def _embed_tokens_shapes_for_inputs(
        self, inputs: OCRInputs
    ) -> list[tuple[int, int]]:
        return [(1, inputs.prompt_length), (1, 1)]


    def prepare_inputs(
        self,
        image_path: str,
        prompt: str = "<image>document parsing.",
        image_mode: str = "gundam",
        pdf_page: int = 1,
        pdf_dpi: int = 200,
        metrics: RuntimeMetrics | None = None,
    ) -> OCRInputs:
        dtype = next(self.native_model.parameters()).dtype
        if metrics is None:
            return prepare_single_image_inputs(
                self.tokenizer,
                image_path=image_path,
                prompt=prompt,
                image_mode=image_mode,
                dtype=dtype,
                pdf_page=pdf_page,
                pdf_dpi=pdf_dpi,
            )
        with metrics.measure("preprocess", "cpu", image_mode):
            return prepare_single_image_inputs(
                self.tokenizer,
                image_path=image_path,
                prompt=prompt,
                image_mode=image_mode,
                dtype=dtype,
                pdf_page=pdf_page,
                pdf_dpi=pdf_dpi,
            )

    @torch.no_grad()
    def generate_hf(
        self,
        inputs: OCRInputs,
        max_new_tokens: int = 128,
        no_repeat_ngram_size: int = 0,
        temperature: float = 0.0,
        metrics: RuntimeMetrics | None = None,
    ) -> dict[str, Any]:
        """Independent HF-`.generate()`-based reference path -- unavailable.

        This used Baidu's remote-code model's own ``transformers`` generate()
        loop as a cross-check independent of our PA-SWA runner. Removing
        ``trust_remote_code`` (to avoid the transformers-4.x-only remote
        modeling file breaking on transformers 5.x) means that independent
        HF model is no longer loaded, so this reference path has no backing
        implementation. Use the production path instead
        (``generate_rbln_pa_swa_cached`` / CLI ``--rbln-components`` including
        ``language_decode``, which is also the CLI default).
        """

        raise NotImplementedError(
            "generate_hf() requires the HF remote-code reference model, which "
            "is no longer loaded (Baidu's modeling_deepseekv2.py is "
            "transformers-4.x-only and breaks under transformers 5.x). Use "
            "generate_rbln_pa_swa_cached() / --rbln-components including "
            "language_decode (the CLI default) instead."
        )

    def _get_rbln_language_runner(self) -> tuple[NativeRBLNLanguagePrefillRunner, bool]:
        if self._rbln_language_runner is None:
            self._rbln_language_runner = NativeRBLNLanguagePrefillRunner(
                self.raw_weights,
                self.native_model.config,
                mode="pa_swa_cached_runtime",
                model_name_or_path=self.model_name_or_path,
            )
            return self._rbln_language_runner, True
        return self._rbln_language_runner, False

    @torch.no_grad()
    def warmup_compact_decode_buckets(
        self,
        buckets: list[int],
        *,
        warmup_tokens: int = 3,
        metrics: RuntimeMetrics | None = None,
    ) -> dict[str, Any]:
        """Pre-compile the compact PA-SWA decode graph for each prompt bucket.

        Drives the long-lived native runner so warmed compile artifacts stay
        alive for subsequent real requests. Intended to be called once after
        load so that every later request whose prompt length maps to a warmed
        bucket decodes with zero compile miss.
        """

        with self._rbln_language_runner_lock:
            runner, runner_created = self._get_rbln_language_runner()
            warmed = runner.warmup_compact_buckets(
                buckets, warmup_tokens=warmup_tokens, metrics=metrics
            )
            return {
                "runner_created": runner_created,
                "buckets": warmed,
                "runner_state": runner.runtime_cache_summary(),
            }

    @torch.no_grad()
    def generate_rbln_pa_swa_cached(
        self,
        inputs: OCRInputs,
        *,
        max_new_tokens: int,
        metrics: RuntimeMetrics | None = None,
    ) -> dict[str, Any]:
        """Production-style PA-SWA cached decode path.

        This path intentionally does not generate an HF reference and does not
        run per-token eager parity replay.  It compiles the one-token
        decoder+logits graph once, uses compiled logits for token selection,
        and carries compiled KV-cache outputs forward between decode steps.
        The scope is intentionally restricted to the current single-request,
        single-block PA-SWA envelope; concurrent/multi-request serving must use
        separate request-scoped runners or move PA-SWA builder bookkeeping out of
        the long-lived runner before sharing this path.
        """

        started = time.perf_counter()
        with _maybe_measure(
            metrics,
            "language_prefill_embeddings",
            "mixed",
            "ocr_multimodal",
        ):
            embeds = self._build_multimodal_inputs_embeds(
                inputs, metrics=metrics
            ).contiguous()
        # The native PA-SWA runner owns mutable KV/cache contexts. Keep compile
        # artifacts warm across calls, but serialize access so a long-lived
        # facade cannot interleave two requests through the same runner.
        with self._rbln_language_runner_lock:
            runner, runner_created = self._get_rbln_language_runner()
            runner_state_before = runner.runtime_cache_summary()
            candidate = runner.run_pa_swa_cached_greedy(
                embeds.squeeze(0).contiguous(),
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                fail_on_compile_blocker=True,
                metrics=metrics,
            )
            output_ids = candidate["output_ids"]
            text = _decode_bytelevel(self.tokenizer, output_ids).strip()
            compiled = candidate["compiled_decode"]
            runner_state_after = runner.runtime_cache_summary()
        compact_profile = candidate.get("compact_decode_profile") or {}
        return {
            "text": text,
            "elapsed_sec": time.perf_counter() - started,
            "tokens": len(output_ids),
            "prompt_length": int(embeds.shape[1]),
            "output_ids_shape": [1, len(output_ids)],
            "decode_mode": "native_vllm_rbln_pa_swa_cached",
            "compact_prompt_bucket": candidate.get("compact_prompt_bucket"),
            "prompt_buckets": candidate.get("prompt_buckets"),
            "checkpoint_load": runner.load_summary,
            "supported_envelope": candidate["supported_envelope"],
            "prefill_kv_nonzero": candidate["prefill_kv_nonzero"],
            "pa_swa_visibility": candidate["pa_swa_visibility"],
            "compact_decode": candidate.get("compact_decode"),
            "compact_decode_profile": compact_profile,
            "decode_cache_reuse": {
                "runner_created": runner_created,
                "runner_generation_count": runner_state_after["generation_count"],
                "native_runtime_rebuilds": runner_state_after["native_runtime_rebuilds"],
                "native_runtime_recreated_this_run": candidate.get(
                    "native_runtime_recreated_this_run", False
                ),
                "native_max_model_len": runner_state_after["native_max_model_len"],
                "compact_cache_entries_before": runner_state_before[
                    "compact_cache_entries"
                ],
                "compact_cache_entries_after": runner_state_after[
                    "compact_cache_entries"
                ],
                "compile_misses": int(compact_profile.get("compile_misses", 0)),
                "cache_hits": int(compact_profile.get("cache_hits", 0)),
                "request_used_warmed_compact_decode": (
                    runner_state_before["compact_cache_entries"] > 0
                    and int(compact_profile.get("invocations", 0)) > 0
                    and int(compact_profile.get("cache_hits", 0))
                    == int(compact_profile.get("invocations", 0))
                    and int(compact_profile.get("compile_misses", 0)) == 0
                    and not bool(
                        candidate.get("native_runtime_recreated_this_run", False)
                    )
                ),
            },
            "compiled_decode_status": compiled["compiled_decode_status"],
            "compiled_decode_backend": compiled["compiled_decode_backend"],
            "compiled_scope": compiled["compiled_scope"],
            "full_decoder_compiled": compiled["full_decoder_compiled"],
            "compiled_decode_no_fallback": compiled["compiled_decode_no_fallback"],
            "compiled_decode_invocations": compiled["compiled_decode_invocations"],
            "compiled_decode_fingerprint": compiled["compiled_decode_fingerprint"],
            "compiled_decode_blocker": compiled.get("compiled_decode_blocker"),
            "compiled_decode_cache_hit": compiled.get("compiled_decode_cache_hit"),
            "compiled_decode_cache_entries": compiled.get("compiled_decode_cache_entries"),
            "compiled_decode_backend_compiled_this_call": compiled.get(
                "compiled_decode_backend_compiled_this_call"
            ),
            "compiled_decode_output_used_for_token": compiled.get(
                "compiled_decode_output_used_for_token"
            ),
            "compiled_decode_eager_compare_uses_cloned_kv": compiled.get(
                "compiled_decode_eager_compare_uses_cloned_kv"
            ),
            "runtime_contract": {
                "hf_reference_generation": False,
                "per_step_eager_compare": False,
                "compiled_logits_drive_token_selection": True,
                "compiled_kv_outputs_drive_decode_state": True,
                "fail_on_compile_or_fallback": True,
            },
        }

    @torch.no_grad()
    def _build_multimodal_inputs_embeds(
        self, inputs: OCRInputs, metrics: RuntimeMetrics | None = None
    ) -> torch.Tensor:
        input_ids = inputs.input_ids.unsqueeze(0)
        images = inputs.images
        images_seq_mask = inputs.images_seq_mask.unsqueeze(0)
        images_spatial_crop = inputs.images_spatial_crop
        inputs_embeds = self.embed_tokens(input_ids).clone()

        if input_ids.shape[1] == 1 or torch.sum(images[0][1]).item() == 0:
            return inputs_embeds

        for idx, (image, crop_shape) in enumerate(zip(images, images_spatial_crop)):
            patches, image_ori = image
            image_features: list[torch.Tensor] = []
            if torch.sum(patches).item() != 0:
                with _maybe_measure(metrics, "language_prefill_vision", self._hf_generation_backend(), "local"):
                    local_features_1 = self.sam_model(patches)
                    local_features_2 = self.vision_model(patches, local_features_1)
                    local_features = torch.cat(
                        (
                            local_features_2[:, 1:],
                            local_features_1.flatten(2).permute(0, 2, 1),
                        ),
                        dim=-1,
                    )
                    local_features = self.projector(local_features)
                with _maybe_measure(metrics, "language_prefill_vision", self._hf_generation_backend(), "global"):
                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1)
                    global_features = torch.cat(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(2).permute(0, 2, 1),
                        ),
                        dim=-1,
                    )
                    global_features = self.projector(global_features)

                _, hw, n_dim = global_features.shape
                h = w = int(hw ** 0.5)
                _, hw2, n_dim2 = local_features.shape
                h2 = w2 = int(hw2 ** 0.5)
                width_crop_num, height_crop_num = int(crop_shape[0]), int(crop_shape[1])

                global_features = global_features.view(h, w, n_dim)
                global_features = torch.cat(
                    [
                        global_features,
                        self.root_model.image_newline[None, None, :].expand(h, 1, n_dim),
                    ],
                    dim=1,
                ).view(-1, n_dim)

                local_features = (
                    local_features.view(
                        height_crop_num, width_crop_num, h2, w2, n_dim2
                    )
                    .permute(0, 2, 1, 3, 4)
                    .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                )
                local_features = torch.cat(
                    [
                        local_features,
                        self.root_model.image_newline[None, None, :].expand(
                            height_crop_num * h2, 1, n_dim2
                        ),
                    ],
                    dim=1,
                ).view(-1, n_dim2)
                image_features.append(
                    torch.cat(
                        [local_features, global_features, self.root_model.view_seperator[None, :]],
                        dim=0,
                    )
                )
            else:
                for img_idx in range(image_ori.shape[0]):
                    single_img = image_ori[img_idx : img_idx + 1]
                    global_features_1 = self.sam_model(single_img)
                    global_features_2 = self.vision_model(single_img, global_features_1)
                    global_features = torch.cat(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(2).permute(0, 2, 1),
                        ),
                        dim=-1,
                    )
                    global_features = self.projector(global_features)
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)
                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat(
                        [
                            global_features,
                            self.root_model.image_newline[None, None, :].expand(h, 1, n_dim),
                        ],
                        dim=1,
                    ).view(-1, n_dim)
                    image_features.append(
                        torch.cat([global_features, self.root_model.view_seperator[None, :]], dim=0)
                    )

            if image_features:
                image_features_tensor = torch.cat(image_features, dim=0).to(inputs_embeds.dtype)
                mask = images_seq_mask[idx].unsqueeze(-1).to(torch.bool)
                inputs_embeds[idx].masked_scatter_(mask, image_features_tensor)
        return inputs_embeds

    def _hf_generation_backend(self) -> str:
        if any(
            backend == "rbln"
            for backend in (
                self.sam_backend,
                self.vision_backend,
                self.projector_backend,
                self.embed_tokens_backend,
            )
        ):
            return "mixed"
        return "cpu"


class NativeRBLNLanguagePrefillRunner:
    """Native vLLM DeepSeek decoder runner for PA-SWA cached OCR decode."""

    def __init__(
        self,
        hf_model: Any,
        hf_config: Any,
        mode: str = "pa_swa_cached_runtime",
        model_name_or_path: str = "baidu/Unlimited-OCR",
    ) -> None:
        # `hf_model` is the raw checkpoint weights dict (see
        # load_unlimited_ocr's raw_weights), not a constructed model: this
        # runner builds its OWN DeepseekV2ForCausalLM sized to each request's
        # actual max_model_len (prompt_len + max_new_tokens), which the
        # facade's own fixed-size native_model can't provide (that one is
        # sized generously once, at load time, for the vision-side model;
        # reusing it here fed the PA-SWA compact-cache machinery a KV cache
        # ~36x larger than the request needed, which crashed the RBLN
        # runtime natively during decode compilation).
        self.hf_model = hf_model
        self.hf_config = hf_config
        self.mode = mode
        self.model_name_or_path = model_name_or_path
        self.native_model: Any | None = None
        self.vllm_config: Any | None = None
        self.compiled_compact_pa_swa_decode_by_key: dict[tuple[Any, ...], Any] = {}
        self.load_summary: dict[str, Any] = {}
        self.generation_count = 0
        self.native_runtime_rebuilds = 0
        self.native_runtime_recreated_this_run = False
        self._native_max_model_len: int | None = None
        self._pa_swa_builders: dict[str, Any] = {}
        self._pa_swa_kv_caches: list[torch.Tensor] = []
        self._pa_swa_runtime_symbols: dict[str, Any] | None = None
        self.prompt_buckets = self._parse_prompt_buckets(
            os.environ.get("UOCR_PROMPT_BUCKETS", "")
        )

    @staticmethod
    def _parse_prompt_buckets(raw: str) -> list[int]:
        """Parse a comma-separated UOCR_PROMPT_BUCKETS list into sorted ints.

        Empty/unset disables bucketing and preserves the exact
        ``prompt_len + sliding_window`` compact shape (legacy behavior).
        """

        buckets: list[int] = []
        for part in raw.replace(" ", "").split(","):
            if not part:
                continue
            value = int(part)
            if value <= 0:
                raise ValueError(
                    "UOCR_PROMPT_BUCKETS entries must be positive, "
                    f"got {value}"
                )
            buckets.append(value)
        return sorted(set(buckets))

    def _bucketed_prompt_len(self, prompt_len: int) -> int:
        """Round ``prompt_len`` up to the smallest configured compile bucket.

        Bucketing keeps the compact PA-SWA decode graph shape constant across
        requests whose true prompt length differs but maps to the same bucket,
        so a warmed compiled decode graph is reused without a per-length
        compile miss. Returns ``prompt_len`` unchanged when no buckets are
        configured. When ``prompt_len`` exceeds the largest bucket, the exact
        length is used (which may trigger its own one-time compile); size the
        largest bucket to cover the longest expected prompt to avoid this.
        """

        if not self.prompt_buckets:
            return prompt_len
        for bucket in self.prompt_buckets:
            if prompt_len <= bucket:
                return bucket
        return prompt_len

    def warmup_compact_buckets(
        self,
        buckets: list[int],
        *,
        warmup_tokens: int = 3,
        metrics: RuntimeMetrics | None = None,
    ) -> list[dict[str, Any]]:
        """Pre-compile and device-load the compact PA-SWA decode graph per bucket.

        For each bucket we synthesize a zero-valued prompt of exactly ``bucket``
        tokens and run a minimal prefill plus a few compact decode steps. The
        compiled decode graph shape depends only on ``bucket + sliding_window``,
        so a real request whose true prompt length maps to the same bucket
        reuses this warmed graph with zero compile miss. Buckets are warmed
        largest-first so the native KV runtime is sized once and smaller buckets
        reuse it without a shape-growth rebuild (which would clear the compile
        cache). Prompt embeds are synthetic, but graph shape — and therefore the
        compile cache key — is identical to the real path, so the warmed graph
        is genuinely reused.

        ``warmup_tokens`` must be >= 3: the first compact decode step is a
        compile miss, and at least one further step exercises the cache-hit
        execution path. That hit-path call is what loads the compiled program
        onto the RBLN device; without it the first real request pays a large
        one-time device-load cost on its first decode token even though the
        graph compile is already cached.
        """

        if int(warmup_tokens) < 3:
            raise ValueError(
                "warmup_tokens must be >= 3 so warmup runs both the compile-miss "
                "and at least one cache-hit decode step (device program load); "
                f"got {warmup_tokens}"
            )

        ordered = sorted({int(b) for b in buckets}, reverse=True)
        results: list[dict[str, Any]] = []
        for bucket in ordered:
            if bucket <= 0:
                raise ValueError(f"warmup bucket must be positive, got {bucket}")
            self._ensure_native_model(max_model_len=bucket + int(warmup_tokens))
            assert self.native_model is not None
            sliding_window = int(self._pa_swa_sliding_window())
            token_ids = torch.zeros((1, bucket), dtype=torch.long)
            embeds = (
                self.native_model.model.embed_tokens(token_ids)
                .squeeze(0)
                .contiguous()
            )
            before = self.runtime_cache_summary()
            candidate = self.run_pa_swa_cached_greedy(
                embeds,
                max_new_tokens=int(warmup_tokens),
                eos_token_id=None,
                fail_on_compile_blocker=True,
                metrics=metrics,
            )
            after = self.runtime_cache_summary()
            profile = candidate.get("compact_decode_profile") or {}
            results.append(
                {
                    "bucket": bucket,
                    "compact_len": bucket + sliding_window,
                    "compile_misses": int(profile.get("compile_misses", 0)),
                    "graph_call_sec": float(profile.get("graph_call_sec", 0.0)),
                    "first_compile_miss_sec": float(
                        profile.get("first_compile_miss_sec", 0.0)
                    ),
                    "compact_cache_entries_before": before["compact_cache_entries"],
                    "compact_cache_entries_after": after["compact_cache_entries"],
                    "native_max_model_len": after["native_max_model_len"],
                    "native_runtime_recreated_this_run": bool(
                        candidate.get("native_runtime_recreated_this_run", False)
                    ),
                    "compiled_decode_status": candidate["compiled_decode"][
                        "compiled_decode_status"
                    ],
                }
            )
        return results

    def run_pa_swa_cached_greedy(
        self,
        inputs_embeds: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        fail_on_compile_blocker: bool = False,
        metrics: RuntimeMetrics | None = None,
    ) -> dict[str, Any]:
        self.native_runtime_recreated_this_run = False
        self._ensure_native_model(
            max_model_len=int(inputs_embeds.shape[0]) + int(max_new_tokens)
        )
        self.reset_pa_swa_request_state()
        self.generation_count += 1
        assert self.native_model is not None
        assert self.vllm_config is not None
        self._assert_pa_swa_cached_envelope()

        prompt_len = int(inputs_embeds.shape[0])
        eos_id = int(eos_token_id) if eos_token_id is not None else None
        sliding_window = int(self._pa_swa_sliding_window())
        decode_timings = {
            "prefill_model": 0.0,
            "prefill_logits": 0.0,
            "decode_loop": 0.0,
            "decode_embed": 0.0,
            "decode_compact_pack": 0.0,
            "decode_compact_total": 0.0,
            "decode_compact_visible": 0.0,
            "decode_compact_context": 0.0,
            "decode_compact_cache_lookup": 0.0,
            "decode_compact_compile_miss": 0.0,
            "decode_compact_graph_call": 0.0,
            "decode_compact_unpack": 0.0,
            "decode_compact_kv_apply": 0.0,
            "token_select": 0.0,
        }
        output_ids: list[int] = []
        compiled_decode: dict[str, Any] = self._initial_compiled_decode_status(
            requested=True
        )
        compact_kv_caches: list[torch.Tensor] | None = None
        compact_decode_report: dict[str, Any] | None = None
        compact_decode_profile = self._new_compact_decode_profile()

        timing_started = time.perf_counter()
        hidden = self._run_pa_swa_model_step(
            inputs_embeds.contiguous(),
            start_position=0,
            is_prefill=True,
        )
        decode_timings["prefill_model"] += time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        logits = self.native_model.compute_logits(hidden)
        decode_timings["prefill_logits"] += time.perf_counter() - timing_started
        if logits is None:
            raise RuntimeError("native vLLM compute_logits returned None")
        timing_started = time.perf_counter()
        next_id = int(torch.argmax(logits[-1].float(), dim=-1).item())
        decode_timings["token_select"] += time.perf_counter() - timing_started
        output_ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            self._record_pa_swa_decode_timings(metrics, decode_timings)
            return {
                "output_ids": output_ids,
                "supported_envelope": self._pa_swa_supported_envelope(),
                "prefill_kv_nonzero": self._prefill_kv_nonzero(),
                "pa_swa_visibility": self._pa_swa_visibility_summary(
                    prompt_len=prompt_len,
                    generated_tokens=len(output_ids),
                ),
                "compiled_decode": compiled_decode,
                "compact_decode_profile": self._compact_decode_profile_summary(
                    compact_decode_profile
                ),
                "runner_state": self.runtime_cache_summary(),
                "native_runtime_recreated_this_run": self.native_runtime_recreated_this_run,
            }

        loop_started = time.perf_counter()
        try:
            for step in range(1, max_new_tokens):
                timing_started = time.perf_counter()
                token = torch.tensor([[output_ids[-1]]], dtype=torch.long)
                next_embed = self.native_model.model.embed_tokens(token).to(
                    dtype=inputs_embeds.dtype
                )
                decode_timings["decode_embed"] += time.perf_counter() - timing_started

                start_position = prompt_len + step - 1
                if compact_kv_caches is None:
                    timing_started = time.perf_counter()
                    compact_kv_caches = self._pack_compact_pa_swa_kv_from_full(
                        start_position=start_position,
                        prompt_len=prompt_len,
                        fixed_len=self._bucketed_prompt_len(prompt_len)
                        + sliding_window,
                    )
                    decode_timings["decode_compact_pack"] += (
                        time.perf_counter() - timing_started
                    )
                timing_started = time.perf_counter()
                compact_outputs = self._run_compiled_compact_pa_swa_decode_step(
                    token_embeds=next_embed.squeeze(0).contiguous(),
                    start_position=start_position,
                    prompt_len=prompt_len,
                    compact_kv_caches=compact_kv_caches,
                    compiled_decode=compiled_decode,
                    fail_on_compile_blocker=fail_on_compile_blocker,
                )
                compact_call_elapsed = time.perf_counter() - timing_started
                compact_step_profile = dict(compact_outputs.get("profile", {}))
                token_select_elapsed_sec = float(
                    compact_outputs.get("token_select_elapsed_sec", 0.0)
                )
                self._accumulate_compact_decode_profile(
                    compact_decode_profile, compact_step_profile
                )
                decode_timings["decode_compact_total"] += compact_call_elapsed
                decode_timings["decode_compact_visible"] += float(
                    compact_step_profile.get("visible_sec", 0.0)
                )
                decode_timings["decode_compact_context"] += float(
                    compact_step_profile.get("context_sec", 0.0)
                )
                decode_timings["decode_compact_cache_lookup"] += float(
                    compact_step_profile.get("cache_lookup_sec", 0.0)
                )
                decode_timings["decode_compact_compile_miss"] += float(
                    compact_step_profile.get("compile_miss_sec", 0.0)
                )
                decode_timings["decode_compact_graph_call"] += float(
                    compact_step_profile.get("graph_call_sec", 0.0)
                )
                decode_timings["decode_compact_unpack"] += float(
                    compact_step_profile.get("unpack_sec", 0.0)
                )
                decode_timings["decode_compact_kv_apply"] += float(
                    compact_step_profile.get("kv_apply_sec", 0.0)
                )
                logits = compact_outputs["logits"]
                compact_kv_caches = compact_outputs["kv_caches"]
                compact_decode_report = compact_outputs["report"]
                compact_next_id = compact_outputs.get("next_id")
                decode_timings["token_select"] += token_select_elapsed_sec

                if compact_next_id is None:
                    if logits is None:
                        raise RuntimeError(
                            "native vLLM compute_logits returned None"
                        )
                    timing_started = time.perf_counter()
                    next_id = int(torch.argmax(logits[-1].float(), dim=-1).item())
                    decode_timings["token_select"] += (
                        time.perf_counter() - timing_started
                    )
                else:
                    next_id = compact_next_id
                output_ids.append(next_id)
                if eos_id is not None and next_id == eos_id:
                    break
        finally:
            decode_timings["decode_loop"] += time.perf_counter() - loop_started
            self._record_pa_swa_decode_timings(metrics, decode_timings)

        return {
            "output_ids": output_ids,
            "supported_envelope": self._pa_swa_supported_envelope(),
            "prefill_kv_nonzero": self._prefill_kv_nonzero(),
            "pa_swa_visibility": self._pa_swa_visibility_summary(
                prompt_len=prompt_len,
                generated_tokens=len(output_ids),
            ),
            "compiled_decode": compiled_decode,
            "compact_decode": compact_decode_report,
            "compact_decode_profile": self._compact_decode_profile_summary(
                compact_decode_profile
            ),
            "runner_state": self.runtime_cache_summary(),
            "native_runtime_recreated_this_run": self.native_runtime_recreated_this_run,
            "compact_ring_decode": True,
            "early_compact_decode": True,
            "prompt_len": prompt_len,
            "prompt_buckets": list(self.prompt_buckets),
            "compact_prompt_bucket": self._bucketed_prompt_len(prompt_len),
        }

    def _new_compact_decode_profile(self) -> dict[str, Any]:
        return {
            "invocations": 0,
            "cache_hits": 0,
            "compile_misses": 0,
            "first_compile_miss_sec": 0.0,
            "total_sec": 0.0,
            "visible_sec": 0.0,
            "context_sec": 0.0,
            "cache_lookup_sec": 0.0,
            "compile_miss_sec": 0.0,
            "graph_call_sec": 0.0,
            "unpack_sec": 0.0,
            "kv_apply_sec": 0.0,
            "token_select_sec": 0.0,
            "min_graph_call_sec": None,
            "max_graph_call_sec": 0.0,
        }

    def _accumulate_compact_decode_profile(
        self, aggregate: dict[str, Any], step_profile: dict[str, Any]
    ) -> None:
        aggregate["invocations"] += 1
        if bool(step_profile.get("cache_hit", False)):
            aggregate["cache_hits"] += 1
        else:
            aggregate["compile_misses"] += 1
        for key in (
            "total_sec",
            "visible_sec",
            "context_sec",
            "cache_lookup_sec",
            "compile_miss_sec",
            "graph_call_sec",
            "unpack_sec",
            "kv_apply_sec",
            "token_select_sec",
        ):
            aggregate[key] += float(step_profile.get(key, 0.0))
        compile_miss_sec = float(step_profile.get("compile_miss_sec", 0.0))
        if compile_miss_sec and not aggregate["first_compile_miss_sec"]:
            aggregate["first_compile_miss_sec"] = compile_miss_sec
        graph_call_sec = float(step_profile.get("graph_call_sec", 0.0))
        if graph_call_sec > 0:
            current_min = aggregate["min_graph_call_sec"]
            aggregate["min_graph_call_sec"] = (
                graph_call_sec
                if current_min is None
                else min(float(current_min), graph_call_sec)
            )
            aggregate["max_graph_call_sec"] = max(
                float(aggregate["max_graph_call_sec"]), graph_call_sec
            )

    def _compact_decode_profile_summary(
        self, aggregate: dict[str, Any]
    ) -> dict[str, Any]:
        invocations = int(aggregate.get("invocations", 0))
        cache_hits = int(aggregate.get("cache_hits", 0))
        graph_call_sec = float(aggregate.get("graph_call_sec", 0.0))
        total_sec = float(aggregate.get("total_sec", 0.0))
        return {
            "invocations": invocations,
            "cache_hits": cache_hits,
            "compile_misses": int(aggregate.get("compile_misses", 0)),
            "first_compile_miss_sec": float(
                aggregate.get("first_compile_miss_sec", 0.0)
            ),
            "total_sec": total_sec,
            "visible_sec": float(aggregate.get("visible_sec", 0.0)),
            "context_sec": float(aggregate.get("context_sec", 0.0)),
            "cache_lookup_sec": float(aggregate.get("cache_lookup_sec", 0.0)),
            "compile_miss_sec": float(aggregate.get("compile_miss_sec", 0.0)),
            "graph_call_sec": graph_call_sec,
            "unpack_sec": float(aggregate.get("unpack_sec", 0.0)),
            "kv_apply_sec": float(aggregate.get("kv_apply_sec", 0.0)),
            "token_select_sec": float(aggregate.get("token_select_sec", 0.0)),
            "avg_graph_call_sec": (graph_call_sec / cache_hits)
            if cache_hits
            else 0.0,
            "avg_total_sec": (total_sec / invocations) if invocations else 0.0,
            "min_graph_call_sec": float(aggregate.get("min_graph_call_sec") or 0.0),
            "max_graph_call_sec": float(aggregate.get("max_graph_call_sec", 0.0)),
        }

    def _record_pa_swa_decode_timings(
        self, metrics: RuntimeMetrics | None, timings: dict[str, float]
    ) -> None:
        if metrics is None:
            return
        measured_loop_children = (
            timings["decode_embed"]
            + timings["decode_compact_pack"]
            + timings["decode_compact_total"]
            + timings["token_select"]
        )
        loop_overhead = max(0.0, timings["decode_loop"] - measured_loop_children)
        rows = (
            ("language_prefill_model", "rbln", "pa_swa_prefill", timings["prefill_model"]),
            ("language_prefill_logits", "cpu", "lm_head", timings["prefill_logits"]),
            ("language_decode_loop_overhead", "cpu", "python_loop", loop_overhead),
            ("language_decode_embed", "rbln", "token_embedding", timings["decode_embed"]),
            ("language_decode_compact_pack", "cpu", "initial_pack", timings["decode_compact_pack"]),
            ("language_decode_compact_visible", "cpu", "visible_set", timings["decode_compact_visible"]),
            ("language_decode_compact_context", "cpu", "metadata_mask", timings["decode_compact_context"]),
            ("language_decode_compact_cache_lookup", "cpu", "compile_cache_lookup", timings["decode_compact_cache_lookup"]),
            ("language_decode_compact_compile_miss", "rbln", "compile_miss", timings["decode_compact_compile_miss"]),
            ("language_decode_compact_graph", "rbln", "compiled_graph_call", timings["decode_compact_graph_call"]),
            ("language_decode_compact_unpack", "cpu", "output_unpack", timings["decode_compact_unpack"]),
            ("language_decode_compact_kv_apply", "cpu", "kv_apply", timings["decode_compact_kv_apply"]),
            ("language_decode_token_select", "cpu", "argmax", timings["token_select"]),
        )
        for component, backend, detail, elapsed_sec in rows:
            if elapsed_sec > 0:
                metrics.add(component, backend, elapsed_sec, detail)

    def _ensure_native_model(self, max_model_len: int) -> None:
        if self.native_model is not None and self.vllm_config is not None:
            current_max_model_len = int(self.vllm_config.model_config.max_model_len)
            if current_max_model_len >= int(max_model_len):
                self._native_max_model_len = current_max_model_len
                return
            self._reset_shape_bound_runtime_state(clear_compiled=True)
        _ensure_unlimited_ocr_config_registered()
        _ensure_vllm_rbln_language_imports()
        _ensure_vllm_distributed_initialized()
        from vllm.config import (
            CacheConfig,
            CompilationConfig,
            DeviceConfig,
            LoadConfig,
            ModelConfig,
            ParallelConfig,
            SchedulerConfig,
            VllmConfig,
            set_current_vllm_config,
        )
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
        from vllm_rbln.model_executor.models.unlimited_ocr import (
            normalize_unlimited_ocr_config_for_vllm,
        )

        def _pa_swa_cached_overrides(config: Any) -> Any:
            # ModelConfig owns this private config object. Make the copy
            # boundary explicit before normalize_unlimited_ocr_config_for_vllm()
            # mutates it in place, and never mutate self.hf_model.config.
            #
            # This MUST keep returning the outer (Unlimited-OCR) config, not
            # config.text_config: vLLM's HFConfigParser.parse() probes
            # hf_overrides on a dummy config first to detect whether it
            # changes `model_type`, and uses the *result's* model_type to
            # decide whether our registered UnlimitedOCRConfig class (keyed
            # on "unlimited-ocr") applies. Returning text_config there would
            # report "deepseek_v2" instead, missing the registry lookup and
            # falling through to a real AutoConfig.from_pretrained() call --
            # which fails without trust_remote_code.
            config = copy.deepcopy(config)
            normalize_unlimited_ocr_config_for_vllm(config)
            config.rbln_prefill_aware_swa_backend_ready = True
            config.rbln_prefill_aware_swa_experimental = True
            config.prefill_aware_swa = True
            if hasattr(config, "text_config"):
                # This runner swaps ModelConfig.hf_config to config.text_config
                # right after construction (see below), since it builds a
                # standalone DeepseekV2ForCausalLM directly. The RBLN
                # attention backend's PA-SWA detection
                # (detect_prefill_aware_swa_status) reads these flags off
                # WHATEVER object ends up as vllm_config.model_config.hf_config
                # -- i.e. text_config here -- so every flag it checks
                # (particularly `requires_prefill_aware_swa`, which gates
                # `active` regardless of `backend_ready`) must be mirrored
                # onto text_config, not just a subset. Missing
                # `requires_prefill_aware_swa` here silently made PA-SWA
                # detection report inactive, which routed the compiled decode
                # step through the plain (non-compact) attention code path
                # instead -- and crashed the RBLN MLIR compiler
                # (`assert: Should be dram alloc`) on that mismatched
                # graph shape.
                config.text_config.requires_prefill_aware_swa = True
                config.text_config.rbln_prefill_aware_swa_backend_ready = True
                config.text_config.rbln_prefill_aware_swa_experimental = True
                config.text_config.prefill_aware_swa = True
            return config

        hf_overrides = _pa_swa_cached_overrides

        model_config = ModelConfig(
            model=self.model_name_or_path,
            trust_remote_code=False,
            dtype="bfloat16",
            max_model_len=max_model_len,
            enforce_eager=True,
            hf_overrides=hf_overrides,
        )
        cache_config = CacheConfig(block_size=max_model_len, enable_prefix_caching=False)
        parallel_config = ParallelConfig(tensor_parallel_size=1)
        scheduler_config = SchedulerConfig(
            max_model_len=max_model_len,
            is_encoder_decoder=False,
            max_num_batched_tokens=max_model_len,
            max_num_seqs=1,
            enable_chunked_prefill=False,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=DeviceConfig(device="cpu"),
            load_config=LoadConfig(),
            compilation_config=CompilationConfig(),
        )
        outer_config = hf_overrides(vllm_config.model_config.hf_config)
        # This runner constructs `DeepseekV2ForCausalLM` directly (not nested
        # inside a multimodal wrapper via init_vllm_registered_model), so it
        # needs a flat DeepseekV2Config as ModelConfig.hf_config, not the
        # outer Unlimited-OCR wrapper -- unlike Baidu's own remote config
        # class (which subclassed DeepseekV2Config directly and so was
        # already flat), our registered UnlimitedOCRConfig extends
        # DeepseekVLV2Config and only carries the flat language fields on
        # the nested text_config. Swap it in here, *after* ModelConfig has
        # already resolved model_type -> the registered config class above.
        vllm_config.model_config.hf_config = outer_config.text_config
        with set_current_vllm_config(vllm_config):
            _ensure_vllm_model_parallel_initialized()
            native_model = DeepseekV2ForCausalLM(vllm_config=vllm_config, prefix="")

        language_weights = [
            (name, tensor)
            for name, tensor in self.hf_model.items()
            if name.startswith("model.layers.")
            or name.startswith("model.norm.")
            or name.startswith("model.embed_tokens.")
            or name.startswith("lm_head.")
        ]
        loaded = native_model.load_weights(language_weights)
        param_names = set(dict(native_model.named_parameters()))
        missing = sorted(param_names - loaded)
        self.load_summary = {
            "weights_seen": len(language_weights),
            "native_parameters": len(param_names),
            "loaded_parameters": len(loaded),
            "missing_parameters": missing[:20],
            "missing_count": len(missing),
            "all_parameters_loaded": len(missing) == 0,
            "pa_swa_backend_ready_scope": "private_pa_swa_true_default_config_unchanged",
            "mode": self.mode,
            "rbln_device_ids": list(available_rbln_device_ids()),
            "rbln_num_devices": rbln_num_devices(),
            "rbln_language_compile_options": rbln_compile_options("language_decode"),
            "vllm_tensor_parallel_size": parallel_config.tensor_parallel_size,
        }
        # vllm_rbln's platform hook forces vllm_config.model_config.dtype to
        # float32 for graph compilation (see platform.py's "force model dtype
        # into fp32" override), which runs *before* this model is
        # constructed -- so freshly-initialized parameters default to
        # float32 regardless of the "bfloat16" requested above. Loading the
        # real (bf16) checkpoint weights on top does not necessarily change
        # each parameter's own dtype back (depends on whether the loader
        # does an in-place copy_ or a dtype-preserving assign), so cast
        # explicitly here to guarantee every parameter matches the
        # checkpoint's actual bf16 weights instead of silently mixing
        # dtypes across layers.
        native_model = native_model.eval().to(torch.bfloat16)
        self.native_model = native_model
        self.vllm_config = vllm_config
        self._native_max_model_len = int(max_model_len)
        self.native_runtime_rebuilds += 1
        self.native_runtime_recreated_this_run = True

    def _reset_shape_bound_runtime_state(self, *, clear_compiled: bool) -> None:
        self.reset_pa_swa_request_state()
        self._pa_swa_builders.clear()
        self.native_model = None
        self.vllm_config = None
        self._native_max_model_len = None
        if clear_compiled:
            self.compiled_compact_pa_swa_decode_by_key.clear()

    def reset_pa_swa_request_state(self) -> None:
        if self.native_model is not None:
            for layer in self.native_model.model.layers:
                # vLLM's Attention.kv_cache is a plain tensor (no longer a
                # per-virtual-engine list); match its own placeholder value.
                layer.self_attn.attn.kv_cache = torch.tensor([])
        self._pa_swa_kv_caches.clear()

    def runtime_cache_summary(self) -> dict[str, Any]:
        return {
            "generation_count": int(self.generation_count),
            "native_runtime_rebuilds": int(self.native_runtime_rebuilds),
            "native_runtime_recreated_this_run": bool(
                self.native_runtime_recreated_this_run
            ),
            "native_max_model_len": (
                int(self._native_max_model_len)
                if self._native_max_model_len is not None
                else None
            ),
            "compact_cache_entries": len(
                self.compiled_compact_pa_swa_decode_by_key
            ),
            "kv_cache_entries": len(self._pa_swa_kv_caches),
            "builder_entries": len(self._pa_swa_builders),
        }

    def _ensure_pa_swa_runtime_symbols(self) -> dict[str, Any]:
        if self._pa_swa_runtime_symbols is not None:
            return self._pa_swa_runtime_symbols

        _ensure_vllm_rbln_language_imports()
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm_rbln.torch_compile_backend import set_warmup_active
        from vllm_rbln.v1.attention.backends.flash_attention import (
            RBLNFlashAttentionMetadataBuilder,
        )

        self._pa_swa_runtime_symbols = {
            "set_current_vllm_config": set_current_vllm_config,
            "set_forward_context": set_forward_context,
            "CommonAttentionMetadata": CommonAttentionMetadata,
            "RBLNFlashAttentionMetadataBuilder": RBLNFlashAttentionMetadataBuilder,
            "set_warmup_active": set_warmup_active,
        }
        return self._pa_swa_runtime_symbols


    def _assert_pa_swa_cached_envelope(self) -> None:
        assert self.vllm_config is not None
        model_config = self.vllm_config.model_config
        cache_config = self.vllm_config.cache_config
        parallel_config = self.vllm_config.parallel_config
        scheduler_config = self.vllm_config.scheduler_config
        sliding_window = (
            getattr(model_config.hf_config, "sliding_window_size", None)
            or getattr(model_config.hf_config, "sliding_window", None)
        )
        checks = {
            "tensor_parallel_size": getattr(parallel_config, "tensor_parallel_size", None)
            == 1,
            "max_num_seqs": getattr(scheduler_config, "max_num_seqs", None) == 1,
            "single_partition": cache_config.block_size == model_config.max_model_len,
            "chunked_prefill_disabled": not scheduler_config.enable_chunked_prefill,
            "prefix_caching_disabled": not cache_config.enable_prefix_caching,
            "sliding_window_configured": sliding_window is not None
            and int(sliding_window) > 0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "PA-SWA cached probe envelope violation: "
                f"failed={failed}, checks={checks}"
            )

    def _pa_swa_supported_envelope(self) -> dict[str, Any]:
        assert self.vllm_config is not None
        model_config = self.vllm_config.model_config
        cache_config = self.vllm_config.cache_config
        parallel_config = self.vllm_config.parallel_config
        scheduler_config = self.vllm_config.scheduler_config
        sliding_window = (
            getattr(model_config.hf_config, "sliding_window_size", None)
            or getattr(model_config.hf_config, "sliding_window", None)
        )
        return {
            "num_reqs": 1,
            "batch_pad": 1,
            "block_table_shape": [1, 1],
            "block_size": cache_config.block_size,
            "max_model_len": model_config.max_model_len,
            "single_partition": cache_config.block_size == model_config.max_model_len,
            "enable_chunked_prefill": scheduler_config.enable_chunked_prefill,
            "enable_prefix_caching": cache_config.enable_prefix_caching,
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "max_num_seqs": scheduler_config.max_num_seqs,
            "sliding_window": int(sliding_window) if sliding_window is not None else None,
            "regular_swa_routing": False,
        }

    def _ensure_pa_swa_kv_caches(self) -> None:
        assert self.native_model is not None
        assert self.vllm_config is not None
        if self._pa_swa_kv_caches:
            return
        max_model_len = self.vllm_config.model_config.max_model_len
        num_kv_heads = self.vllm_config.model_config.get_num_kv_heads(
            self.vllm_config.parallel_config
        )
        for layer in self.native_model.model.layers:
            head_size = layer.self_attn.attn.impl.head_size
            kv_shape = (2, 1, num_kv_heads, 1, max_model_len, head_size)
            kv_cache = torch.zeros(kv_shape, dtype=torch.bfloat16)
            # get_attention_context() reads attn_layer.kv_cache directly as
            # the tensor (no per-virtual-engine list wrapping anymore).
            layer.self_attn.attn.kv_cache = kv_cache
            self._pa_swa_kv_caches.append(kv_cache)

    def _initial_compiled_decode_status(self, *, requested: bool) -> dict[str, Any]:
        return {
            "compiled_decode_status": "not_requested"
            if not requested
            else "compile_pending",
            "compiled_decode_backend": "none",
            "compiled_scope": "decoder_norm_lm_head",
            "full_decoder_compiled": False,
            "compiled_decode_no_fallback": False,
            "compiled_decode_invocations": 0,
            "compiled_decode_fingerprint": None,
            "compiled_decode_blocker": None,
        }

    def _pa_swa_decode_with_logits(
        self,
        token_embeds: torch.Tensor,
        positions: torch.Tensor,
        *kv_caches: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Compiled one-token decode unit: forward + logits + updated KV.

        The compact PA-SWA decode path compiles this callable. KV caches are
        passed as explicit graph inputs and the post-forward updated caches are
        returned so the compiled (functional) graph's writes can be carried into
        the next decode step.
        """

        from vllm.forward_context import get_forward_context

        assert self.native_model is not None
        forward_context = get_forward_context()
        for layer, kv_cache in zip(self.native_model.model.layers, kv_caches):
            attn = layer.self_attn.attn
            attn.kv_cache = kv_cache
            metadata = forward_context.attn_metadata[attn.layer_name]
            metadata.kv_caches = [kv_cache]

        hidden = self.native_model(None, positions, None, token_embeds.contiguous())
        logits = self.native_model.compute_logits(hidden)
        if logits is None:
            raise RuntimeError("native vLLM compute_logits returned None")
        updated_kv_caches = tuple(
            getattr(
                layer.self_attn.attn.impl,
                "_pa_swa_updated_kv_cache",
                layer.self_attn.attn.kv_cache[0],
            )
            for layer in self.native_model.model.layers
        )
        return (logits, *updated_kv_caches)

    def _run_pa_swa_model_step(
        self,
        inputs_embeds: torch.Tensor,
        *,
        start_position: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        symbols = self._ensure_pa_swa_runtime_symbols()
        set_current_vllm_config = symbols["set_current_vllm_config"]
        set_forward_context = symbols["set_forward_context"]
        CommonAttentionMetadata = symbols["CommonAttentionMetadata"]
        RBLNFlashAttentionMetadataBuilder = symbols[
            "RBLNFlashAttentionMetadataBuilder"
        ]
        set_warmup_active = symbols["set_warmup_active"]

        assert self.native_model is not None
        assert self.vllm_config is not None
        self._ensure_pa_swa_kv_caches()
        q_len = int(inputs_embeds.shape[0])
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        if start_position + q_len > max_model_len:
            raise RuntimeError(
                "PA-SWA cached probe exceeds single-partition cache: "
                f"start={start_position}, q_len={q_len}, max={max_model_len}"
            )
        positions = torch.arange(
            start_position, start_position + q_len, dtype=torch.long
        )
        seq_len_for_metadata = start_position if not is_prefill else q_len
        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, q_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len_for_metadata], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=q_len,
            max_query_len=q_len,
            max_seq_len=max_model_len,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=positions.to(torch.int64),
            _seq_lens_cpu=torch.tensor([seq_len_for_metadata], dtype=torch.int32),
        )
        ctx: dict[str, Any] = {}
        slot_ctx: dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.native_model.model.layers):
            attn = layer.self_attn.attn
            builder = self._pa_swa_builders.get(attn.layer_name)
            if builder is None:
                with set_current_vllm_config(self.vllm_config):
                    builder = RBLNFlashAttentionMetadataBuilder(
                        attn.get_kv_cache_spec(self.vllm_config),
                        [attn.layer_name],
                        self.vllm_config,
                        torch.device("cpu"),
                    )
                self._pa_swa_builders[attn.layer_name] = builder
            with set_current_vllm_config(self.vllm_config):
                meta = builder.build(
                    0,
                    common,
                    positions=positions,
                    batch_pad=1,
                    is_prefill=is_prefill,
                )
            meta.kv_caches = [self._pa_swa_kv_caches[layer_index]]
            ctx[attn.layer_name] = meta
            slot_ctx[attn.layer_name] = common.slot_mapping
        set_warmup_active(True)
        try:
            with torch.no_grad(), set_forward_context(
                ctx, self.vllm_config, num_tokens=q_len, slot_mapping=slot_ctx
            ):
                return self.native_model(
                    None, positions, None, inputs_embeds.contiguous()
                )
        finally:
            set_warmup_active(False)

    def _prefill_kv_nonzero(self) -> bool:
        if not self._pa_swa_kv_caches:
            return False
        return all(bool(cache.abs().sum().item() > 0) for cache in self._pa_swa_kv_caches)

    def _pa_swa_sliding_window(self) -> int:
        assert self.vllm_config is not None
        sliding_window = (
            getattr(self.vllm_config.model_config.hf_config, "sliding_window_size", None)
            or getattr(self.vllm_config.model_config.hf_config, "sliding_window", None)
        )
        if sliding_window is None:
            raise RuntimeError("PA-SWA compact probe requires sliding_window")
        return int(sliding_window)

    def _compact_pa_swa_visible_after(
        self, *, start_position: int, prompt_len: int
    ) -> list[int]:
        from vllm_rbln.v1.attention.prefill_aware_swa import (
            prefill_aware_visible_token_indices,
        )

        return prefill_aware_visible_token_indices(
            seq_len=start_position + 1,
            prefill_len=prompt_len,
            sliding_window=self._pa_swa_sliding_window(),
        )

    def _compact_pa_swa_slot_for_position(
        self, *, position: int, prompt_len: int
    ) -> int:
        """Map a logical token position to the compact PA-SWA physical slot.

        Prompt tokens keep stable dense slots [0, prompt_len). Generated
        tokens reuse the fixed sliding-window tail as a ring buffer. Attention
        for a single decode query is permutation-invariant across visible K/V
        slots after RoPE has already been applied to K, so this avoids the
        previous per-step full compact-cache shift without changing the visible
        token set.
        """

        if position < prompt_len:
            return int(position)
        return int(prompt_len + ((position - prompt_len) % self._pa_swa_sliding_window()))

    def _pack_compact_pa_swa_kv_from_full(
        self, *, start_position: int, prompt_len: int, fixed_len: int | None = None
    ) -> list[torch.Tensor]:
        visible_after = self._compact_pa_swa_visible_after(
            start_position=start_position, prompt_len=prompt_len
        )
        if not visible_after or visible_after[-1] != start_position:
            raise RuntimeError(
                "compact PA-SWA production path requires current token in visible set"
            )
        active_len = len(visible_after)
        compact_len = int(fixed_len) if fixed_len is not None else active_len
        if compact_len < active_len:
            raise RuntimeError(
                "fixed compact KV length is smaller than active PA-SWA visibility: "
                f"fixed={compact_len} active={active_len}"
            )
        compact_kv_caches: list[torch.Tensor] = []
        for full_cache in self._pa_swa_kv_caches:
            compact_cache = torch.zeros(
                (*full_cache.shape[:4], compact_len, full_cache.shape[-1]),
                dtype=full_cache.dtype,
                device=full_cache.device,
            )
            for position in visible_after[:-1]:
                compact_slot = self._compact_pa_swa_slot_for_position(
                    position=int(position), prompt_len=prompt_len
                )
                compact_cache[..., compact_slot, :] = full_cache[..., int(position), :]
            compact_kv_caches.append(compact_cache.contiguous())
        return compact_kv_caches

    def _build_compact_pa_swa_decode_context(
        self,
        *,
        compact_kv_caches: list[torch.Tensor],
        positions: torch.Tensor,
        compact_len: int,
        compact_slot: int,
        active_visible_count: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        assert self.native_model is not None
        assert self.vllm_config is not None
        from vllm.config import set_current_vllm_config
        from vllm.v1.attention.backend import CommonAttentionMetadata

        symbols = self._ensure_pa_swa_runtime_symbols()
        RBLNFlashAttentionMetadataBuilder = symbols[
            "RBLNFlashAttentionMetadataBuilder"
        ]
        active_visible_count = int(active_visible_count or compact_len)
        if active_visible_count > compact_len:
            raise RuntimeError(
                "active visible count exceeds compact KV length: "
                f"active={active_visible_count} compact={compact_len}"
            )
        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([active_visible_count - 1], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=compact_len,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.tensor([compact_slot], dtype=torch.int64),
            _seq_lens_cpu=torch.tensor([active_visible_count - 1], dtype=torch.int32),
        )

        ctx: dict[str, Any] = {}
        slot_ctx: dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.native_model.model.layers):
            attn = layer.self_attn.attn
            builder = self._pa_swa_builders.get(attn.layer_name)
            if builder is None:
                with set_current_vllm_config(self.vllm_config):
                    builder = RBLNFlashAttentionMetadataBuilder(
                        attn.get_kv_cache_spec(self.vllm_config),
                        [attn.layer_name],
                        self.vllm_config,
                        torch.device("cpu"),
                    )
                self._pa_swa_builders[attn.layer_name] = builder
            with set_current_vllm_config(self.vllm_config):
                meta = builder.build(
                    0,
                    common,
                    positions=positions,
                    batch_pad=1,
                    is_prefill=False,
                )
            meta.kv_caches = [compact_kv_caches[layer_index]]
            meta.attn_masks = torch.zeros(
                1, 1, 1, 1, compact_len, dtype=torch.float16
            )
            meta.attn_masks[:, :, :, :, :active_visible_count] = 1
            meta.slot_update_mask = torch.zeros(
                1, 1, 1, compact_len, 1, dtype=torch.float16
            )
            meta.slot_update_mask[:, :, :, compact_slot, :] = 1
            meta.block_tables = torch.tensor([[0]], dtype=torch.int32)
            meta.slot_mapping = torch.tensor([compact_slot], dtype=torch.int64)
            meta.single_block_id = 0
            meta.single_slot_id = compact_slot
            ctx[attn.layer_name] = meta
            slot_ctx[attn.layer_name] = meta.slot_mapping
        return ctx, slot_ctx

    def _run_compiled_compact_pa_swa_decode_step(
        self,
        *,
        token_embeds: torch.Tensor,
        start_position: int,
        prompt_len: int,
        compact_kv_caches: list[torch.Tensor],
        compiled_decode: dict[str, Any],
        fail_on_compile_blocker: bool,
    ) -> dict[str, Any]:
        assert self.native_model is not None
        assert self.vllm_config is not None
        from vllm.forward_context import set_forward_context
        from vllm_rbln.torch_compile_backend import logged_rbln_backend

        step_total_started = time.perf_counter()
        profile: dict[str, Any] = {
            "total_sec": 0.0,
            "visible_sec": 0.0,
            "context_sec": 0.0,
            "cache_lookup_sec": 0.0,
            "compile_miss_sec": 0.0,
            "graph_call_sec": 0.0,
            "unpack_sec": 0.0,
            "kv_apply_sec": 0.0,
            "token_select_sec": 0.0,
            "cache_hit": False,
        }
        timing_started = time.perf_counter()
        visible_after = self._compact_pa_swa_visible_after(
            start_position=start_position, prompt_len=prompt_len
        )
        active_visible_count = len(visible_after)
        compact_len = int(compact_kv_caches[0].shape[4])
        compact_slot = self._compact_pa_swa_slot_for_position(
            position=start_position, prompt_len=prompt_len
        )
        if compact_len < active_visible_count:
            raise RuntimeError(
                "compact KV shape is smaller than PA-SWA active visibility: "
                f"cache={compact_len} active={active_visible_count}"
            )
        positions = torch.tensor([start_position], dtype=torch.long)
        profile["visible_sec"] = time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        ctx, slot_ctx = self._build_compact_pa_swa_decode_context(
            compact_kv_caches=compact_kv_caches,
            positions=positions,
            compact_len=compact_len,
            compact_slot=compact_slot,
            active_visible_count=active_visible_count,
        )
        profile["context_sec"] = time.perf_counter() - timing_started
        compile_options = rbln_compile_options("language_decode")
        first_cache = compact_kv_caches[0]
        cache_key = (
            "compact_pa_swa_decode_production",
            tuple(int(x) for x in token_embeds.shape),
            tuple(int(x) for x in positions.shape),
            tuple(int(x) for x in first_cache.shape),
            tuple(int(x) for x in first_cache.stride()),
            str(first_cache.dtype),
            str(first_cache.device),
            len(compact_kv_caches),
            int(self._pa_swa_sliding_window()),
            tuple(sorted(compile_options.items())),
            getattr(self.hf_config, "_name_or_path", "baidu/Unlimited-OCR"),
        )
        backend_state = {"completed": False}

        def compile_backend(gm: torch.fx.GraphModule, example_inputs, **kwargs):
            compiled_forward = logged_rbln_backend(gm, example_inputs, **kwargs)
            backend_state["completed"] = True
            return compiled_forward

        try:
            timing_started = time.perf_counter()
            compiled = self.compiled_compact_pa_swa_decode_by_key.get(cache_key)
            cache_hit = compiled is not None
            profile["cache_hit"] = cache_hit
            profile["cache_lookup_sec"] = time.perf_counter() - timing_started
            if compiled is None:
                timing_started = time.perf_counter()
                compiled = torch.compile(
                    self._pa_swa_decode_with_logits,
                    dynamic=False,
                    fullgraph=True,
                    backend=compile_backend,
                    options=compile_options,
                )
                self.compiled_compact_pa_swa_decode_by_key[cache_key] = compiled
                profile["compile_miss_sec"] += time.perf_counter() - timing_started

            def run_compiled_decode() -> Any:
                with _temporary_rotary_no_empty_cat_patch():
                    return compiled(
                        token_embeds.contiguous(),
                        positions.contiguous(),
                        *tuple(cache.contiguous() for cache in compact_kv_caches),
                    )

            if cache_hit:
                timing_started = time.perf_counter()
                with torch.no_grad(), set_forward_context(
                    ctx, self.vllm_config, num_tokens=1, slot_mapping=slot_ctx
                ):
                    compiled_outputs = run_compiled_decode()
                profile["graph_call_sec"] += time.perf_counter() - timing_started
                compile_log = ""
            else:
                compile_stdout = io.StringIO()
                compile_stderr = io.StringIO()
                compile_logging = io.StringIO()
                log_handler = logging.StreamHandler(compile_logging)
                root_logger = logging.getLogger()
                root_logger.addHandler(log_handler)
                timing_started = time.perf_counter()
                with torch.no_grad(), set_forward_context(
                    ctx, self.vllm_config, num_tokens=1, slot_mapping=slot_ctx
                ), contextlib.redirect_stdout(compile_stdout), contextlib.redirect_stderr(
                    compile_stderr
                ):
                    try:
                        compiled_outputs = run_compiled_decode()
                    finally:
                        root_logger.removeHandler(log_handler)
                compile_miss_and_first_call_sec = time.perf_counter() - timing_started
                profile["compile_miss_sec"] += compile_miss_and_first_call_sec
                compile_log = (
                    compile_stdout.getvalue()
                    + compile_stderr.getvalue()
                    + compile_logging.getvalue()
                )

            fallback_markers = (
                "Fallback to eager execution",
                "RBLN compilation failure",
                "torch.cpu",
                "fallback to CPU",
            )
            fallback = any(
                marker.lower() in compile_log.lower() for marker in fallback_markers
            )
            completed = bool(cache_hit or backend_state["completed"])
            timing_started = time.perf_counter()
            logits = compiled_outputs[0]
            updated_kv_caches = [cache.detach() for cache in compiled_outputs[1:]]
            profile["unpack_sec"] += time.perf_counter() - timing_started
            clean_compiled_decode = (
                not fallback
                and completed
                and len(updated_kv_caches) == len(compact_kv_caches)
            )
            if not clean_compiled_decode:
                blocker = "RBLN compact production decode did not complete cleanly"
                compiled_decode.update(
                    {
                        "compiled_decode_status": "compile_blocked",
                        "compiled_decode_backend": "none",
                        "compiled_scope": "compact_decoder_norm_lm_head",
                        "full_decoder_compiled": False,
                        "compiled_decode_no_fallback": False,
                        "compiled_decode_blocker": blocker,
                        "compact_decode_log_excerpt": compile_log[-4000:],
                    }
                )
                raise RuntimeError(blocker)
            timing_started = time.perf_counter()
            for layer, cache in zip(self.native_model.model.layers, updated_kv_caches):
                layer.self_attn.attn.kv_cache = cache
            profile["kv_apply_sec"] += time.perf_counter() - timing_started
            token_select_started = time.perf_counter()
            compiled_next_id = int(torch.argmax(logits[-1].float(), dim=-1).item())
            token_select_elapsed_sec = time.perf_counter() - token_select_started
            profile["token_select_sec"] += token_select_elapsed_sec
            profile["total_sec"] = time.perf_counter() - step_total_started
            compact_invocations = int(compiled_decode.get("compact_decode_invocations", 0)) + 1
            compact_status = {
                "compiled_decode_invocations": int(
                    compiled_decode.get("compiled_decode_invocations", 0)
                )
                + 1,
                "compiled_decode_compiled_next_id": compiled_next_id,
                "compiled_decode_output_used_for_token": True,
                "compiled_decode_eager_compare_uses_cloned_kv": False,
                "compact_decode_active": True,
                "compact_decode_invocations": compact_invocations,
                "compact_decode_cache_hit": cache_hit,
                "compact_decode_cache_entries": len(
                    self.compiled_compact_pa_swa_decode_by_key
                ),
                "compact_decode_compiled_this_call": bool(backend_state["completed"]),
                "compact_decode_kv_shape": [int(x) for x in first_cache.shape],
                "compact_decode_visible_count": active_visible_count,
                "compact_decode_physical_len": compact_len,
                "compact_decode_slot": compact_slot,
                "compact_decode_update_mode": "ring",
                "compact_decode_ring_slots": True,
            }
            if clean_compiled_decode:
                compact_status.update(
                    {
                        "compiled_decode_status": "pass_compact_runtime",
                        "compiled_decode_backend": "rbln",
                        "compiled_scope": "compact_decoder_norm_lm_head",
                        "full_decoder_compiled": True,
                        "compiled_decode_no_fallback": True,
                        "compiled_decode_blocker": None,
                    }
                )
            compiled_decode.update(compact_status)
            return {
                "logits": logits.detach(),
                "kv_caches": updated_kv_caches,
                "next_id": compiled_next_id,
                "token_select_elapsed_sec": token_select_elapsed_sec,
                "profile": profile,
                "report": {
                    "active": True,
                    "backend": "rbln",
                    "compiled": completed,
                    "no_fallback": not fallback,
                    "start_position": start_position,
                    "visible_count": active_visible_count,
                    "physical_len": compact_len,
                    "compact_slot": compact_slot,
                    "update_mode": "ring",
                    "ring_slots": True,
                    "invocations": compact_invocations,
                    "cache_hit": cache_hit,
                    "cache_entries": len(self.compiled_compact_pa_swa_decode_by_key),
                    "compact_kv_shape": [int(x) for x in first_cache.shape],
                    "compiled_next_id": compiled_next_id,
                },
            }
        except Exception as exc:
            compiled_decode.update(
                {
                    "compiled_decode_status": "compile_blocked",
                    "compiled_decode_backend": "none",
                    "compiled_scope": "compact_decoder_norm_lm_head",
                    "full_decoder_compiled": False,
                    "compiled_decode_no_fallback": False,
                    "compiled_decode_blocker": f"{exc.__class__.__name__}: {exc}",
                }
            )
            raise

    def _pa_swa_visibility_summary(
        self, *, prompt_len: int, generated_tokens: int
    ) -> dict[str, Any]:
        assert self.vllm_config is not None
        from vllm_rbln.v1.attention.prefill_aware_swa import (
            prefill_aware_visible_token_indices,
        )

        sliding_window = (
            getattr(self.vllm_config.model_config.hf_config, "sliding_window_size", None)
            or getattr(self.vllm_config.model_config.hf_config, "sliding_window", None)
        )
        if sliding_window is None:
            return {"pass": False, "reason": "missing_sliding_window"}
        seq_len = prompt_len + max(0, generated_tokens - 1)
        visible = prefill_aware_visible_token_indices(
            seq_len=seq_len,
            prefill_len=prompt_len,
            sliding_window=int(sliding_window),
        )
        expected_prefix = list(range(prompt_len))
        pass_visibility = visible[:prompt_len] == expected_prefix
        return {
            "pass": pass_visibility,
            "seq_len_before_last_decode": seq_len,
            "prefill_len": prompt_len,
            "sliding_window": int(sliding_window),
            "visible_count": len(visible),
            "visible_head": visible[: min(8, len(visible))],
            "visible_tail": visible[-min(8, len(visible)) :] if visible else [],
        }


_VLLM_RBLN_LANGUAGE_IMPORTS_READY = False


def _ensure_vllm_rbln_language_imports() -> None:
    global _VLLM_RBLN_LANGUAGE_IMPORTS_READY
    if _VLLM_RBLN_LANGUAGE_IMPORTS_READY:
        return

    import vllm_rbln
    from vllm_rbln.platform import RblnPlatform

    vllm_rbln.register_model()
    vllm_rbln.register_ops()
    import vllm_rbln.model_executor.layers.fused_moe.layer  # noqa: F401
    from vllm_rbln.torch_compile_backend import logged_rbln_backend  # noqa: F401

    def _pa_swa_check_and_update_config(cls, vllm_config: Any) -> None:
        # check_and_update_config() encodes assumptions from RBLN's
        # standard vLLM-engine path (real Scheduler/Worker, compiled
        # decode, chunked prefill always on, eager mode implying a
        # specific device-tensor + no-sliding-window setup, etc). This
        # runtime never goes through that path -- it builds
        # DeepseekV2ForCausalLM directly and feeds attn_metadata itself --
        # and several of those assumptions directly conflict with the
        # PA-SWA cached probe's actual, deliberate design:
        #   * enable_chunked_prefill=False is required here (see
        #     _validate_pa_swa_envelope's "chunked_prefill_disabled" check)
        #     to control prefill/decode step boundaries manually.
        #   * sliding_window is intentionally set (PA-SWA), which the
        #     upstream eager-mode assert forbids.
        #   * the upstream eager-mode branch also force-overwrites
        #     model_config.dtype to float16, which would fight this
        #     runtime's explicit bf16 weights/compute.
        # On the vLLM version this experimental runtime was originally
        # built against, this hook was never invoked for a bare
        # VllmConfig(...) construction outside the real engine, so this
        # probe never had to satisfy it; newer vLLM now calls it eagerly
        # from VllmConfig.__post_init__ regardless. Skip it here to
        # restore that behavior for this runtime specifically -- this
        # does not affect the standard-engine (LLM()/vllm serve) path,
        # which never imports this module.
        if scheduler_config := getattr(vllm_config, "scheduler_config", None):
            if scheduler_config.async_scheduling:
                scheduler_config.async_scheduling = False

    RblnPlatform.check_and_update_config = classmethod(
        _pa_swa_check_and_update_config
    )

    _VLLM_RBLN_LANGUAGE_IMPORTS_READY = True


@contextlib.contextmanager
def _temporary_rotary_no_empty_cat_patch():
    """Avoid RoPE zero-width concatenation only while compiling decode.

    Unlimited-OCR uses full-dimension RoPE (`rotary_dim == head_size`).  Upstream
    vLLM still builds `torch.cat((rotated, empty_pass), dim=-1)`, which can lower
    to a Relay concatenate with problematic dtype inference.  Keep the eager
    production/probe path untouched and patch only the compile attempt.
    """

    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    original_forward_static = RotaryEmbedding.__dict__["forward_static"]

    @staticmethod
    def forward_static_no_empty_cat(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        def apply_rotary_without_neox_cat(x: torch.Tensor) -> torch.Tensor:
            return ApplyRotaryEmb.forward_static(x, cos, sin, is_neox_style)

        query_shape = query.shape
        query = query.view(num_tokens, -1, head_size)
        query_rot = apply_rotary_without_neox_cat(query[..., :rotary_dim])
        if rotary_dim == head_size:
            query = query_rot.reshape(query_shape)
        else:
            query = torch.cat((query_rot, query[..., rotary_dim:]), dim=-1).reshape(
                query_shape
            )

        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, head_size)
            key_rot = apply_rotary_without_neox_cat(key[..., :rotary_dim])
            if rotary_dim == head_size:
                key = key_rot.reshape(key_shape)
            else:
                key = torch.cat((key_rot, key[..., rotary_dim:]), dim=-1).reshape(
                    key_shape
                )
        return query, key

    RotaryEmbedding.forward_static = forward_static_no_empty_cat
    try:
        yield
    finally:
        RotaryEmbedding.forward_static = original_forward_static


def _ensure_vllm_distributed_initialized() -> None:
    import torch.distributed as dist
    from vllm.distributed.parallel_state import init_distributed_environment

    if not dist.is_initialized():
        tmp = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp}",
            local_rank=0,
            backend="gloo",
        )


def _ensure_vllm_model_parallel_initialized() -> None:
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
        initialize_model_parallel,
    )

    try:
        get_tensor_model_parallel_world_size()
    except Exception:
        initialize_model_parallel(1, 1)

def load_unlimited_ocr(
    model_name_or_path: str = "baidu/Unlimited-OCR",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> LoadedUnlimitedOCR:
    """Load Unlimited-OCR entirely via vLLM-native classes, no remote code.

    Baidu's own checkpoint only exposes a config/model shape through remote
    code (``trust_remote_code=True``), and that remote code unconditionally
    imports a transformers-4.x-era modeling file that breaks on transformers
    5.x. vLLM's own ``DeepseekOCRForCausalLM`` already reimplements the exact
    same vision towers (SAM ViT-B + CLIP-L) + projector + DeepSeek-V2 MoE
    language backbone natively, so this builds that class directly and loads
    the checkpoint's raw safetensors weights into it -- verified to load all
    584 parameters with zero missing/unexpected keys, and to produce
    numerically-equivalent (not bit-identical -- see vision parity notes)
    vision features versus Baidu's remote code.
    """

    _ensure_unlimited_ocr_config_registered()
    _ensure_vllm_rbln_language_imports()
    _ensure_vllm_distributed_initialized()
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm_rbln.model_executor.models.unlimited_ocr import (
        UnlimitedOCRForCausalLM,
        normalize_unlimited_ocr_config_for_vllm,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=False
    )

    def hf_overrides(config: Any) -> Any:
        normalize_unlimited_ocr_config_for_vllm(config)
        # Enable the (fail-closed-by-default) PA-SWA backend for this one
        # model instance. This custom runtime's compact PA-SWA decode path is
        # the extensively-validated production path here (unlike the
        # still-in-progress standard-engine serving route, which stays
        # fail-closed unless UOCR_STANDARD_ENGINE_PA_SWA=1). Setting this at
        # construction time -- rather than rebuilding a second, separately
        # PA-SWA-flagged DeepseekV2ForCausalLM later in
        # NativeRBLNLanguagePrefillRunner -- avoids initializing the RBLN
        # attention backend/distributed state a second time in one process,
        # which crashes the RBLN runtime (native abort in librbln.so).
        config.rbln_prefill_aware_swa_backend_ready = True
        config.rbln_prefill_aware_swa_experimental = True
        config.prefill_aware_swa = True
        if hasattr(config, "text_config"):
            config.text_config.rbln_prefill_aware_swa_backend_ready = True
            config.text_config.prefill_aware_swa = True
        return config

    max_model_len = 32768
    model_config = ModelConfig(
        model=model_name_or_path,
        trust_remote_code=False,
        dtype=torch_dtype,
        max_model_len=max_model_len,
        enforce_eager=True,
        hf_overrides=hf_overrides,
    )
    cache_config = CacheConfig(block_size=max_model_len, enable_prefix_caching=False)
    parallel_config = ParallelConfig(tensor_parallel_size=1)
    scheduler_config = SchedulerConfig(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        enable_chunked_prefill=False,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )
    with set_current_vllm_config(vllm_config):
        _ensure_vllm_model_parallel_initialized()
        native_model = UnlimitedOCRForCausalLM(vllm_config=vllm_config, prefix="")

    raw_weights = _load_unlimited_ocr_raw_weights(model_name_or_path)
    loaded = native_model.load_weights(raw_weights.items())
    param_names = set(dict(native_model.named_parameters()))
    missing = sorted(param_names - loaded)
    if missing:
        raise RuntimeError(
            f"Unlimited-OCR native load is missing {len(missing)} parameters "
            f"(first 20): {missing[:20]}"
        )

    native_model = native_model.eval().to(torch_dtype)
    facade = UnlimitedOCRFacade(
        native_model,
        tokenizer,
        raw_weights=raw_weights,
        model_name_or_path=model_name_or_path,
        vllm_config=vllm_config,
    )
    return LoadedUnlimitedOCR(
        tokenizer=tokenizer, native_model=native_model, facade=facade
    )


def _image_shape_tuple(images: torch.Tensor) -> tuple[int, int, int, int]:
    return (
        int(images.shape[0]),
        int(images.shape[1]),
        int(images.shape[2]),
        int(images.shape[3]),
    )


def _projector_shape_from_image_tensor(images: torch.Tensor) -> tuple[int, int, int]:
    batch = int(images.shape[0])
    height = int(images.shape[-2])
    patch_size = 16
    downsample_ratio = 4
    seq = (height // patch_size // downsample_ratio) ** 2
    return (batch, seq, 2048)


def _sam_output_shape_from_image_shape(
    image_shape: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    batch, _, height, width = image_shape
    patch_size = 16
    downsample_ratio = 4
    return (
        batch,
        1024,
        height // patch_size // downsample_ratio,
        width // patch_size // downsample_ratio,
    )


def _module_info(module: Any) -> dict[str, Any]:
    params = list(module.parameters()) if hasattr(module, "parameters") else []
    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    first_param = params[0] if params else None
    return {
        "class": module.__class__.__name__,
        "parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "dtype": str(first_param.dtype) if first_param is not None else None,
        "device": str(first_param.device) if first_param is not None else None,
    }


def _raise_rbln_compile_error(
    component: str, input_summary: dict[str, Any], exc: Exception
) -> NoReturn:
    raise RuntimeError(
        "RBLN component compile failed: "
        f"component={component}, input_summary={input_summary}, "
        f"reason={exc.__class__.__name__}: {exc}"
    ) from exc


def _set_runtime_attr(module: Any, name: str, value: Any) -> None:
    modules = getattr(module, "_modules", None)
    if isinstance(modules, dict) and name in modules:
        del modules[name]
    setattr(module, name, value)


@contextlib.contextmanager
def _patch_tensor_cuda_for_cpu():
    original_cuda = torch.Tensor.cuda

    def _cuda_noop(
        tensor: torch.Tensor,
        device: Any = None,
        non_blocking: bool = False,
        memory_format: Any = None,
    ) -> torch.Tensor:
        # The production-safe fallback path is intentionally CPU-only even on
        # CUDA-capable hosts.  HF remote code may call Tensor.cuda() directly;
        # returning the tensor here keeps the advertised "language_decode=cpu"
        # runtime contract true instead of silently moving decode to CUDA.
        return tensor

    torch.Tensor.cuda = cast(Any, _cuda_noop)
    try:
        yield
    finally:
        torch.Tensor.cuda = cast(Any, original_cuda)


@contextlib.contextmanager
def _maybe_measure(
    metrics: RuntimeMetrics | None, component: str, backend: str, detail: str = ""
):
    if metrics is None:
        yield
    else:
        with metrics.measure(component, backend, detail):
            yield


@contextlib.contextmanager
def _without_sliding_window(config: Any, keep_ring_window: bool = False):
    original = getattr(config, "sliding_window", None)
    original_size = getattr(config, "sliding_window_size", None)
    original_ring = getattr(config, "_ring_window", None)
    config._ring_window = (original_size or original) if keep_ring_window else None
    config.sliding_window = None
    try:
        yield
    finally:
        config.sliding_window = original
        config._ring_window = original_ring


def _patch_remote_interpolate_ops(vision_model: Any) -> None:
    """Avoid currently unsupported RBLN upsample ops in HF remote code.

    Unlimited-OCR remote code resizes CLIP/SAM positional embeddings with
    antialiased bicubic and 1D linear F.interpolate.  On ATOM/RBLN those lower to
    aten::_upsample_bicubic2d_aa and aten::upsample_linear1d and can force eager
    CPU fallback.  This experimental runner patches only the loaded remote module:
    - bicubic positional resize keeps bicubic but disables antialiasing;
    - 1D relative-position resize uses explicit gather/lerp instead of
      F.interpolate(mode="linear").
    """
    module = inspect.getmodule(vision_model.__class__)
    if module is None:
        return
    if not getattr(
        module.__dict__.get("get_abs_pos"), "_unlimited_ocr_rbln_patched", False
    ):
        module.__dict__["get_abs_pos"] = _make_get_abs_pos_no_antialias()
    if not getattr(
        module.__dict__.get("get_abs_pos_sam"), "_unlimited_ocr_rbln_patched", False
    ):
        module.__dict__["get_abs_pos_sam"] = _make_get_abs_pos_sam_no_antialias()
    if not getattr(
        module.__dict__.get("get_rel_pos"), "_unlimited_ocr_rbln_patched", False
    ):
        module.__dict__["get_rel_pos"] = _get_rel_pos_without_interpolate


def _make_get_abs_pos_no_antialias():
    def get_abs_pos_no_antialias(abs_pos: torch.Tensor, tgt_size: int) -> torch.Tensor:
        dim = abs_pos.size(-1)
        abs_pos_new = abs_pos.squeeze(0)
        cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]
        src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
        tgt_size_int = int(math.sqrt(tgt_size))
        dtype = abs_pos.dtype

        if src_size == tgt_size_int:
            return abs_pos

        old_pos_embed = (
            old_pos_embed.view(1, src_size, src_size, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(torch.float32)
        )
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size_int, tgt_size_int),
            mode="bicubic",
            antialias=False,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size_int * tgt_size_int, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        return vision_pos_embed.view(1, tgt_size_int * tgt_size_int + 1, dim)

    get_abs_pos_no_antialias._unlimited_ocr_rbln_patched = True  # type: ignore[attr-defined]
    return get_abs_pos_no_antialias


def _make_get_abs_pos_sam_no_antialias():
    def get_abs_pos_sam_no_antialias(
        abs_pos: torch.Tensor, tgt_size: int
    ) -> torch.Tensor:
        dtype = abs_pos.dtype
        src_size = abs_pos.size(1)
        if src_size == tgt_size:
            return abs_pos
        old_pos_embed = abs_pos.permute(0, 3, 1, 2).to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=False,
            align_corners=False,
        ).to(dtype)
        return new_pos_embed.permute(0, 2, 3, 1)

    get_abs_pos_sam_no_antialias._unlimited_ocr_rbln_patched = True  # type: ignore[attr-defined]
    return get_abs_pos_sam_no_antialias


def _get_rel_pos_without_interpolate(
    q_size: int, k_size: int, rel_pos: torch.Tensor
) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        dtype = rel_pos.dtype
        old_len = int(rel_pos.shape[0])
        scale = (old_len - 1) / (max_rel_dist - 1) if max_rel_dist > 1 else 0.0
        pos = (
            torch.arange(max_rel_dist, device=rel_pos.device, dtype=torch.float32)
            * scale
        )
        left = torch.floor(pos).to(torch.long)
        right = torch.clamp(left + 1, max=old_len - 1)
        weight = (pos - left.to(pos.dtype)).unsqueeze(-1)
        rel_pos_float = rel_pos.to(torch.float32)
        rel_pos_resized = (
            rel_pos_float[left] * (1.0 - weight) + rel_pos_float[right] * weight
        )
        rel_pos_resized = rel_pos_resized.to(dtype)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(
        k_size / q_size, 1.0
    )
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(
        q_size / k_size, 1.0
    )
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


_get_rel_pos_without_interpolate._unlimited_ocr_rbln_patched = True  # type: ignore[attr-defined]
