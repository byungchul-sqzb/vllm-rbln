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

"""Native vLLM-RBLN model surface for Baidu Unlimited-OCR.

This adapter moves the language path away from Hugging Face remote-code
attention/cache execution and toward a native vLLM DeepSeek runtime, matching the
structure used by Baidu's custom SGLang wheel.  The inherited upstream vLLM
``DeepseekOCRForCausalLM`` already supplies the closest available composition:

- SAM + CLIP vision towers;
- MLP projector plus newline/view-separator layout;
- native vLLM DeepSeek language backbone;
- multimodal embedding merge through vLLM's multimodal interfaces.

Unlimited-OCR's HF config differs from Deepseek-OCR's vLLM config shape: language
fields live at top level, while ``vision_config`` and ``projector_config`` are raw
dicts.  ``normalize_unlimited_ocr_config_for_vllm`` patches those differences in
place before the inherited initializer reads the config.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from transformers import BatchFeature
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from vllm.logger import init_logger
from vllm.model_executor.models.deepseek_ocr import (
    DeepseekOCRDummyInputsBuilder,
    DeepseekOCRForCausalLM,
    DeepseekOCRMultiModalProcessor,
    DeepseekOCRProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import MultiModalProcessingInfo
from vllm.transformers_utils.configs.deepseek_vl2 import (
    MlpProjectorConfig,
    VisionEncoderConfig,
)
from vllm.transformers_utils.processors.deepseek_ocr import (
    DeepseekOCRProcessor,
    count_tiles,
)

logger = init_logger(__name__)

_IMAGE_TOKEN = "<image>"
_UNLIMITED_OCR_IMAGE_MODES: dict[str, tuple[int, int, bool]] = {
    "gundam": (1024, 640, True),
    "base": (1024, 1024, False),
    "small": (640, 640, False),
    "tiny": (512, 512, False),
    "large": (1280, 1280, False),
}
_UNLIMITED_OCR_MULTI_IMAGE_ALLOWED_MODES = frozenset({"tiny", "small", "base"})
# Unlimited-OCR's own spec allows up to 32 local crops (vs DeepSeek-OCR's 6);
# see official vLLM's `unlimited_ocr.py` (`_UNLIMITED_OCR_MAX_CROPS`). The
# *predicted* token count below always honors this value, but the real
# `DeepseekOCRProcessor` we reuse for actual tiling is the version installed
# with this vLLM, whose `tokenize_with_images` calls `dynamic_preprocess`
# without a `max_num=` override and so is hard-capped at its own module
# default of 6 -- see the docstring on `UnlimitedOCRProcessingInfo` below.
_UNLIMITED_OCR_MAX_CROPS = 32
# Only "gundam" has crop_mode=True; multi-image + crop is what breaks vLLM's
# per-item processing cache invariant (see official vLLM's
# `UnlimitedOCRProcessor.tokenize_with_images` / `_cached_apply_hf_processor`),
# so it is auto-fallen-back to non-crop instead of rejected. "large" is
# already crop_mode=False -- Baidu's own reference still rejects it for
# multi-image (see SGLang's `_MULTI_IMAGE_ALLOWED`), for a presumably
# unrelated vision-token-budget reason, so it keeps the hard rejection below.
_UNLIMITED_OCR_AUTO_FALLBACK_MODE = "gundam"


_CONFIG_KEYS_NOT_LANGUAGE = {
    "architectures",
    "auto_map",
    "model_type",
    "vision_config",
    "projector_config",
    "tile_tag",
    "global_view_pos",
    "candidate_resolutions",
    "transformers_version",
    # Baidu's raw config also carries the (already-flattened) language fields
    # under this nested key; excluding it keeps that raw dict from leaking
    # into `DeepseekV2Config(**language_kwargs)` as a spurious, unused
    # `text_config.language_config` attribute.
    "language_config",
}


def normalize_unlimited_ocr_config_for_vllm(config: Any) -> Any:
    """Patch Baidu Unlimited-OCR HF config into vLLM DeepseekOCR shape.

    The function mutates and returns ``config`` so it can be used directly on
    ``vllm_config.model_config.hf_config`` immediately before calling the
    inherited Deepseek-OCR initializer.
    """

    if not hasattr(config, "text_config"):
        language_kwargs = _language_kwargs_from_top_level_config(config)
        # `kv_lora_rank` is an MLA-only field Unlimited-OCR's raw config sets
        # to None (this checkpoint has no MLA low-rank projection at all).
        # `DeepseekV2Config.kv_lora_rank` is declared `int` (not optional),
        # so newer transformers releases with strict dataclass-based config
        # validation reject `None` outright; drop it so the class's own
        # default (512) applies instead. `deepseek_v2.py`'s own MHA-vs-MLA
        # model-class selection only reads `qk_nope_head_dim`/
        # `qk_rope_head_dim` (forced to 0 below), so that part is unaffected.
        if language_kwargs.get("kv_lora_rank") is None:
            language_kwargs.pop("kv_lora_rank", None)
        config.text_config = DeepseekV2Config(**language_kwargs)
        # However, vLLM's own `ModelConfig.use_mla` / `is_deepseek_mla()`
        # (transformers_utils/model_arch_config_convertor.py) infers MLA
        # from `model_type == "deepseek_v2" and kv_lora_rank is not None` --
        # it doesn't look at qk_nope/qk_rope_head_dim at all. Since
        # kv_lora_rank can no longer be the None sentinel this checkpoint
        # wants (see above), that heuristic now misfires and reports
        # use_mla=True for a model that has no MLA weights, which trips
        # our own PA-SWA-sliding-window code's `assert not
        # vllm_config.model_config.use_mla` (MLA + PA-SWA sliding window
        # isn't implemented). Change model_type so vLLM's heuristic no
        # longer matches; nothing else in vllm/vllm_rbln keys off the
        # "deepseek_v2" model_type string.
        config.text_config.model_type = "unlimited-ocr-language"

    # `init_vllm_registered_model` resolves the language backbone's model
    # class from `text_config.architectures`.  This must be forced
    # unconditionally (not just when unset): when `config` already carries a
    # nested `language_config` dict (e.g. when built via a config class that
    # extracts it directly, such as `DeepseekVLV2Config`), that nested dict's
    # own `architectures` field names the *outer* multimodal wrapper Baidu
    # used at export time (`DeepseekOCRForCausalLM`), which has vision/
    # projector fields `text_config` doesn't carry -- resolving it would
    # construct the wrong (multimodal) class for what must be a text-only
    # backbone.
    config.text_config.architectures = ["DeepseekV2ForCausalLM"]

    # Unlimited-OCR's checkpoint is plain multi-head attention (separate
    # q_proj/k_proj/v_proj, no MLA low-rank compression); its config never
    # sets the MLA-only fields (qk_nope_head_dim, qk_rope_head_dim). But
    # `DeepseekV2Config.__init__` defaults those to non-zero MLA values
    # (128/64) regardless, and vLLM's `DeepseekV2DecoderLayer` picks its
    # attention class by checking `all(dim == 0 for dim in
    # (qk_nope_head_dim, qk_rope_head_dim))`. Force them to 0 here so vLLM
    # selects its existing plain-MHA `DeepseekAttention` class instead of an
    # MLA class the checkpoint has no weights for.
    config.text_config.qk_nope_head_dim = 0
    config.text_config.qk_rope_head_dim = 0

    # The nested `language_config` dict in Baidu's raw checkpoint only
    # carries `sliding_window_size`, not `sliding_window` -- unlike the
    # top-level dict, which has both (`sliding_window` is the standard
    # vLLM/transformers attention-backend field name; `get_kv_cache_spec`/
    # RBLNFlashAttentionMetadataBuilder read `sliding_window` directly, not
    # `sliding_window_size`). When `text_config` is built from that nested
    # dict (e.g. via DeepseekVLV2Config), `text_config.sliding_window` ends
    # up None even though the top-level config has 128, silently disabling
    # PA-SWA windowing in the attention backend and corrupting the compact
    # KV cache the runner packs assuming a real window size.
    if getattr(config.text_config, "sliding_window", None) is None:
        top_level_sliding_window = getattr(config, "sliding_window", None) or getattr(
            config, "sliding_window_size", None
        )
        if top_level_sliding_window is not None:
            config.text_config.sliding_window = top_level_sliding_window
        elif getattr(config.text_config, "sliding_window_size", None) is not None:
            config.text_config.sliding_window = config.text_config.sliding_window_size

    if isinstance(getattr(config, "vision_config", None), dict):
        config.vision_config = VisionEncoderConfig(**config.vision_config)

    if isinstance(getattr(config, "projector_config", None), dict):
        config.projector_config = MlpProjectorConfig(**config.projector_config)

    # vLLM's DeepseekOCR adapter expects these DeepseekVLV2-style fields.
    if not hasattr(config, "tile_tag"):
        config.tile_tag = "2D"
    if not hasattr(config, "global_view_pos"):
        config.global_view_pos = "head"
    if not hasattr(config, "candidate_resolutions"):
        config.candidate_resolutions = ((384, 384),)

    # Keep top-level vocab in sync with native language config.
    if hasattr(config.text_config, "vocab_size"):
        config.vocab_size = config.text_config.vocab_size

    # HF config alone does not expose Baidu's prefill-aware SWA contract.
    # vLLM-RBLN handles it with a full-KV cache plus a PA-SWA visibility mask:
    # decode attends to all prefill tokens and only the latest decode window.
    config.requires_prefill_aware_swa = True
    # Production PA-SWA decode KV-cache/runtime through the standard vLLM
    # engine (LLM()/`vllm serve`) is still being validated end-to-end. Keep it
    # fail-closed by default so no existing behavior changes; set
    # UOCR_STANDARD_ENGINE_PA_SWA=1 to opt into the in-progress standard-engine
    # serving path for validation. The bounded experimental probe runner in
    # examples/experimental/unlimited_ocr always overrides a private copied
    # config directly and does not depend on this env var.
    config.rbln_prefill_aware_swa_backend_ready = os.environ.get(
        "UOCR_STANDARD_ENGINE_PA_SWA", "0"
    ).strip().lower() not in {"0", "false", "no", "off", ""}
    config.rbln_prefill_aware_swa_experimental = False
    config.prefill_aware_swa = True

    return config


def _get_unlimited_ocr_image_mode(mm_kwargs: Mapping[str, object] | None) -> str:
    mode = "gundam"
    if mm_kwargs is not None:
        raw_mode = mm_kwargs.get("image_mode", mode)
        if isinstance(raw_mode, str):
            mode = raw_mode
    if mode not in _UNLIMITED_OCR_IMAGE_MODES:
        raise ValueError(
            f"unknown Unlimited-OCR image_mode={mode!r}; "
            f"choices={sorted(_UNLIMITED_OCR_IMAGE_MODES)}"
        )
    return mode


def _get_unlimited_ocr_mode_params(image_mode: str) -> tuple[int, int, bool]:
    if image_mode not in _UNLIMITED_OCR_IMAGE_MODES:
        raise ValueError(
            f"unknown Unlimited-OCR image_mode={image_mode!r}; "
            f"choices={sorted(_UNLIMITED_OCR_IMAGE_MODES)}"
        )
    return _UNLIMITED_OCR_IMAGE_MODES[image_mode]


def _validate_unlimited_ocr_image_count(image_mode: str, image_count: int) -> None:
    _get_unlimited_ocr_mode_params(image_mode)
    if image_count <= 0:
        raise ValueError(f"image_count must be positive, got {image_count}")
    if (
        image_count > 1
        and image_mode not in _UNLIMITED_OCR_MULTI_IMAGE_ALLOWED_MODES
        and image_mode != _UNLIMITED_OCR_AUTO_FALLBACK_MODE
    ):
        raise ValueError(
            "Unlimited-OCR multi-image prompts are only supported for "
            f"{sorted(_UNLIMITED_OCR_MULTI_IMAGE_ALLOWED_MODES)} (or "
            f"{_UNLIMITED_OCR_AUTO_FALLBACK_MODE!r}, which falls back to "
            "non-crop processing); got "
            f"image_mode={image_mode!r} with image_count={image_count}"
        )


def _get_unlimited_ocr_effective_crop_mode(image_mode: str, image_count: int) -> bool:
    """Return whether crop mode is actually active for this request.

    Mirrors official vLLM's ``effective_cropping = CROP_MODE and len(images)
    == 1``: "gundam" silently disables cropping for multi-image requests
    instead of being rejected, since crop_mode + multi-image is what breaks
    vLLM's per-item processing cache invariant, not multi-image itself.
    """
    _, _, crop_mode = _get_unlimited_ocr_mode_params(image_mode)
    if image_mode == _UNLIMITED_OCR_AUTO_FALLBACK_MODE and image_count > 1:
        return False
    return crop_mode


def _get_unlimited_ocr_num_image_tokens(
    *,
    image_width: int,
    image_height: int,
    image_mode: str,
    cropping: bool | None = None,
) -> int:
    base_size, image_size, mode_crop_mode = _get_unlimited_ocr_mode_params(image_mode)
    # `cropping=None` means "use this image_mode's own default"; callers that
    # already know the real per-request image count (e.g. `_get_prompt_updates`
    # below) pass an explicit override computed via
    # `_get_unlimited_ocr_effective_crop_mode`.
    crop_mode = mode_crop_mode if cropping is None else cropping
    patch_size = 16
    downsample_ratio = 4
    if crop_mode:
        if image_width <= image_size and image_height <= image_size:
            num_width_tiles = num_height_tiles = 1
        else:
            num_width_tiles, num_height_tiles = count_tiles(
                image_width,
                image_height,
                max_num=_UNLIMITED_OCR_MAX_CROPS,
                image_size=image_size,
            )
    else:
        num_width_tiles = num_height_tiles = 1

    h = w = math.ceil((base_size // patch_size) / downsample_ratio)
    h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)
    global_views_tokens = h * (w + 1)
    if num_width_tiles > 1 or num_height_tiles > 1:
        local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
    else:
        local_views_tokens = 0
    return global_views_tokens + local_views_tokens + 1


def _language_kwargs_from_top_level_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        raw = config.to_dict()
    else:
        raw = dict(getattr(config, "__dict__", {}))
    raw = deepcopy(raw)
    return {
        key: value
        for key, value in raw.items()
        if not key.startswith("_") and key not in _CONFIG_KEYS_NOT_LANGUAGE
    }


class UnlimitedOCRProcessingInfo(DeepseekOCRProcessingInfo):
    """Unlimited-OCR processing info with Baidu image-mode presets.

    Real per-image tiling still goes through the installed vLLM's
    ``DeepseekOCRProcessor``, whose ``tokenize_with_images`` calls
    ``dynamic_preprocess`` without a configurable ``max_num=`` (that
    parameter only exists in a newer upstream version of this file, gated
    behind a ``DeepseekOCRProcessor.__init__(max_crops=...)`` we don't have
    here). So the real crop count stays capped at that module's
    ``MAX_CROPS`` (6), even though ``_get_unlimited_ocr_num_image_tokens``
    below predicts token counts assuming Unlimited-OCR's true
    ``max_crops=32``. Closing this gap needs either duplicating
    ``tokenize_with_images`` or a vLLM upgrade.
    """

    def get_hf_processor(self, **kwargs: object):
        image_mode = _get_unlimited_ocr_image_mode(kwargs)
        base_size, image_size, _ = _get_unlimited_ocr_mode_params(image_mode)
        processor_config = dict(
            image_size=image_size,
            base_size=base_size,
            strategy="v1",
        )
        processor_kwargs = {**kwargs, **processor_config}
        processor_kwargs.pop("image_mode", None)
        # `crop_mode` is intentionally NOT set here: `DeepseekOCRProcessor`
        # (this installed vLLM's version) has no `crop_mode` constructor
        # parameter, so it would be silently dropped as an unused kwarg.
        # The real per-request crop toggle is instead passed as a *call-time*
        # `crop_mode` kwarg by `UnlimitedOCRMultiModalProcessor._call_hf_processor`,
        # which `DeepseekOCRProcessor.__call__`/`process_one` do accept.
        processor_kwargs.pop("crop_mode", None)
        return self.ctx.get_hf_processor(DeepseekOCRProcessor, **processor_kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Baidu allows multi-image only in tiny/small/base (gundam falls back
        # to non-crop instead of being rejected).  The concrete mode is an
        # mm_processor kwarg, so per-request validation/fallback is performed
        # by the processor while the global modality limit remains unbounded.
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        cropping: bool | None = None,
        image_mode: str = "gundam",
    ) -> int:
        return _get_unlimited_ocr_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
            image_mode=image_mode,
            cropping=cropping,
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        # The worst legal Baidu/SGLang crop ratio is 1x6 for gundam mode.
        # This produces more tokens than square 1280x1280, so use a tall
        # image to keep vLLM multimodal profiling/prompt budgeting safe.
        return ImageSize(width=640, height=640 * 6)


class UnlimitedOCRMultiModalProcessor(DeepseekOCRMultiModalProcessor):
    """DeepseekOCR processor with Unlimited-OCR image_mode plumbing.

    Multi-image + crop mode ("gundam") auto-falls-back to non-crop instead of
    being rejected, matching official vLLM's ``UnlimitedOCRProcessor`` /
    ``_get_prompt_updates``. Both the actual processor call below and the
    placeholder-token-count prediction in ``_get_prompt_updates`` must agree
    on this same effective crop flag, or the predicted prompt token count
    would diverge from what the processor actually produces.
    """

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        image_mode = _get_unlimited_ocr_image_mode(mm_kwargs)
        images = mm_data.get("images") or mm_data.get("image") if mm_data else None
        image_count = len(images) if isinstance(images, (list, tuple)) else 1
        _validate_unlimited_ocr_image_count(image_mode, image_count)

        effective_cropping = _get_unlimited_ocr_effective_crop_mode(
            image_mode, image_count
        )
        if effective_cropping != _get_unlimited_ocr_mode_params(image_mode)[2]:
            logger.warning_once(
                "Unlimited-OCR: crop mode is not supported for multi-image "
                "input. Falling back to cropping=False."
            )
        # `DeepseekOCRProcessor.__call__`/`process_one` accept `crop_mode` as
        # a call-time keyword (unlike the constructor, see `get_hf_processor`
        # above), so this is where the per-request crop toggle -- including
        # the multi-image fallback -- actually takes effect.
        mm_kwargs = {**mm_kwargs, "crop_mode": effective_cropping}
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_mode = _get_unlimited_ocr_image_mode(hf_processor_mm_kwargs)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_unlimited_ocr(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )
            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                size = images.get_image_size(item_idx)
                # Must agree with the effective crop flag `_call_hf_processor`
                # actually uses, or the predicted placeholder token count
                # diverges from what the processor really emits.
                effective_cropping = _get_unlimited_ocr_effective_crop_mode(
                    image_mode, len(images)
                )
                num_image_tokens = _get_unlimited_ocr_num_image_tokens(
                    image_width=size.width,
                    image_height=size.height,
                    image_mode=image_mode,
                    cropping=effective_cropping,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_unlimited_ocr,
            )
        ]

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        # The gundam auto-fallback above makes per-item processor output
        # depend on how many images are in the request, which breaks the
        # per-item processing cache's invariance assumption. Bypass the cache
        # for multi-image requests (recomputed fresh each time) and only
        # cache the single-image case, matching official vLLM's
        # `UnlimitedOCRMultiModalProcessor`/`DeepseekVL2MultiModalProcessor`.
        if inputs.mm_data_items.get_count("image", strict=False) > 1:
            return self._apply_hf_processor(inputs, timing_ctx)
        return super()._cached_apply_hf_processor(inputs, timing_ctx)


@MULTIMODAL_REGISTRY.register_processor(
    UnlimitedOCRMultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):
    """Unlimited-OCR native adapter scaffold with Baidu PA-SWA contract."""

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        # vLLM's model loader detects "new-style" model classes by inspecting
        # __init__ for literal `vllm_config`/`prefix` parameters (see
        # vllm.model_executor.model_loader.utils.initialize_model); a
        # *args/**kwargs signature is treated as old-style and vllm_config is
        # never passed in, so this signature must name them explicitly.
        normalize_unlimited_ocr_config_for_vllm(vllm_config.model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError("Only image modality is supported")

    def requires_prefill_aware_swa(self) -> bool:
        """Unlimited-OCR requires Baidu PA-SWA for exact sliding-window decode."""

        return True

    def is_prefill_aware_swa_backend_ready(self) -> bool:
        """Return whether production RBLN attention/KV-cache PA-SWA is wired."""

        return bool(getattr(self.config, "rbln_prefill_aware_swa_backend_ready", False))

    def get_attention_sliding_window_size(self) -> int | None:
        """Return SWA size only when the PA-SWA backend is production-ready.

        Failing closed avoids silently running regular sliding-window attention,
        which would evict prefill KV and diverge from Baidu's PA-SWA contract.
        """

        if (
            self.requires_prefill_aware_swa()
            and not self.is_prefill_aware_swa_backend_ready()
        ):
            return None
        return getattr(self.config, "sliding_window_size", None) or getattr(
            self.config, "sliding_window", None
        )

    def get_required_attention_sliding_window_size(self) -> int | None:
        """Return the Baidu PA-SWA window required by the checkpoint."""

        return getattr(self.config, "sliding_window_size", None) or getattr(
            self.config, "sliding_window", None
        )

    def is_prefill_aware_swa(self) -> bool:
        """Report active PA-SWA support, not merely the model requirement."""

        return self.is_prefill_aware_swa_backend_ready()


EntryClass = [UnlimitedOCRForCausalLM]
