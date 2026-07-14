"""In-process Unlimited-OCR service: load once, warm buckets, serve many.

This is the production-style reuse surface requested for Unlimited-OCR on ATOM:
a long-lived object that loads the model and compiles the RBLN graphs a single
time, then serves many OCR requests reusing warmed compiled artifacts. It does
not start an HTTP/socket server; callers embed it in their own process.

Two compile-cost sources are amortized here:

1. Compact PA-SWA decode graph: ``warmup_decode_buckets()`` pre-compiles one
   graph per configured prompt-length bucket. With buckets configured (see
   ``prompt_buckets`` / ``UOCR_PROMPT_BUCKETS``), every request whose true
   prompt length maps to a warmed bucket decodes with zero compile miss.
2. RBLN vision/frontend graphs: compiled lazily on the first ``generate()``
   call and reused. Frontend graphs are keyed by input tile shapes; for a fixed
   ``image_mode`` these are typically stable across requests.

The supported envelope matches the underlying runtime: batch=1, single request,
serialized runner access. Concurrent multi-request serving is out of scope and
requires request-scoped runners (see milestone notes).
"""

from __future__ import annotations

import os
from typing import Any

import torch

from examples.experimental.unlimited_ocr.metrics import RuntimeMetrics
from examples.experimental.unlimited_ocr.runtime import load_unlimited_ocr


def _normalize_buckets(prompt_buckets: list[int] | None) -> list[int]:
    if not prompt_buckets:
        return []
    return sorted({int(b) for b in prompt_buckets})


def _apply_bucket_env(buckets: list[int]) -> None:
    """Publish buckets to the env the facade reads at construction time.

    ``NativeRBLNLanguagePrefillRunner.__init__`` reads ``UOCR_PROMPT_BUCKETS``,
    so this must run before the facade/runner is built.
    """

    if buckets:
        os.environ["UOCR_PROMPT_BUCKETS"] = ",".join(str(b) for b in buckets)


class UnlimitedOCRService:
    """Load Unlimited-OCR once and serve many OCR requests with warm graphs."""

    def __init__(
        self,
        model: str = "baidu/Unlimited-OCR",
        *,
        rbln_components: str = "all",
        prompt_buckets: list[int] | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.prompt_buckets = _normalize_buckets(prompt_buckets)
        _apply_bucket_env(self.prompt_buckets)

        # Imported here so applying the bucket env happens before any facade is
        # constructed, and to avoid a circular import at module load time.
        from examples.experimental.unlimited_ocr.infer import (
            _component_compile_set,
            _language_backend_from_components,
            _parse_components,
        )

        self.rbln_components = _parse_components(rbln_components)
        self.compile_components = _component_compile_set(self.rbln_components)
        self.language_backend = _language_backend_from_components(
            self.rbln_components
        )

        self.loaded = load_unlimited_ocr(model, torch_dtype=torch_dtype)
        self.facade = self.loaded.facade
        # Frontend RBLN graphs are shape-specific (embed_tokens compiles per
        # prompt length; vision/projector per image tile shape). Track which
        # input signatures have been compiled and grow the set as new shapes
        # appear, recompiling the frontend from the original modules for the
        # full accumulated set.
        self._frontend_inputs: list[Any] = []
        self._frontend_keys: set[Any] = set()

    def _frontend_key(self, inputs: Any) -> Any:
        image_shapes = tuple(self.facade._image_shapes_for_inputs(inputs))
        return (int(inputs.prompt_length), image_shapes)

    def _ensure_frontend(self, inputs_iterable, *, metrics=None) -> bool:
        """Compile frontend for any not-yet-seen input signatures.

        Returns True if a (re)compile happened. Passing several inputs at once
        compiles them in a single enable call (no repeated rebuilds).
        """
        if not self.compile_components:
            return False
        items = (inputs_iterable if isinstance(inputs_iterable, (list, tuple))
                 else [inputs_iterable])
        new = False
        for inp in items:
            key = self._frontend_key(inp)
            if key not in self._frontend_keys:
                self._frontend_keys.add(key)
                self._frontend_inputs.append(inp)
                new = True
        if new:
            self.facade.enable_rbln_components(
                self.compile_components, self._frontend_inputs, metrics=metrics
            )
        return new

    def warmup_decode_buckets(
        self,
        buckets: list[int] | None = None,
        *,
        warmup_tokens: int = 3,
    ) -> dict[str, Any]:
        """Pre-compile the compact decode graph for each prompt bucket.

        Call once after construction. Returns per-bucket compile evidence; a
        correctly warmed bucket reports ``compile_misses == 1`` here (the single
        warmup compile) and lets later real requests report ``0``.

        IMPORTANT — partial warmup. This compiles (and reuses) the decode graph
        so real requests never recompile, but it runs on a synthetic prompt and
        does NOT execute the RBLN vision frontend. Measured on ATOM, the first
        real request after only this synthetic warmup still pays a large one-time
        device-load cost on its first decode token (~33s observed), because
        running the vision frontend for the first time evicts the warmed decode
        program from the device. To make the first real request fully warm, also
        call :meth:`warmup` with one representative image per bucket, which runs
        the full real path and absorbs that device-load cost into warmup.
        """

        targets = self.prompt_buckets if buckets is None else _normalize_buckets(
            buckets
        )
        if not targets:
            raise ValueError(
                "no prompt buckets configured; pass prompt_buckets to the "
                "service or to warmup_decode_buckets()"
            )
        metrics = RuntimeMetrics()
        summary = self.facade.warmup_compact_decode_buckets(
            targets, warmup_tokens=warmup_tokens, metrics=metrics
        )
        summary["stage_timings"] = metrics.stages_to_list()
        return summary

    def warmup(
        self,
        images: list[str],
        *,
        image_mode: str = "gundam",
        max_length: int = 32768,
        prompt: str = "<image>document parsing.",
        pdf_page: int = 0,
        pdf_dpi: int = 200,
        warmup_tokens: int = 3,
    ) -> dict[str, Any]:
        """Fully warm the runtime so the first real request is already optimal.

        Recommended production warmup. For each representative image it runs one
        full real generation (output discarded), which compiles the vision
        frontend and the decode graph for that image's prompt bucket AND absorbs
        the one-time RBLN device-load cost (vision-frontend-first-execution) that
        synthetic decode warmup alone cannot. Provide one image per prompt bucket
        you expect to serve so every bucket's full path is loaded before traffic.

        Buckets configured on the service still drive decode-graph reuse, so two
        images of slightly different prompt length that map to the same bucket
        share one warmed graph. If you configured buckets that no warm image
        covers, call :meth:`warmup_decode_buckets` for those to at least
        pre-compile them (compile-only; see its note on residual device-load).
        """

        # Pre-enable the frontend for ALL warm images in one pass so each
        # component is compiled once for the full shape union (avoids O(N^2)
        # rebuilds that per-image lazy enable would cause).
        if self.compile_components:
            prepared = [
                self.facade.prepare_inputs(
                    image, prompt=prompt, image_mode=image_mode,
                    pdf_page=pdf_page, pdf_dpi=pdf_dpi,
                )
                for image in images
            ]
            self._ensure_frontend(prepared)

        per_image: list[dict[str, Any]] = []
        for image in images:
            result = self.generate(
                image,
                image_mode=image_mode,
                max_length=max_length,
                prompt=prompt,
                pdf_page=pdf_page,
                pdf_dpi=pdf_dpi,
            )
            profile = result.get("compact_decode_profile") or {}
            per_image.append(
                {
                    "image": image,
                    "prompt_length": result.get("prompt_length"),
                    "compact_prompt_bucket": result.get("compact_prompt_bucket"),
                    "compile_misses": int(profile.get("compile_misses", 0)),
                    "max_graph_call_sec": float(
                        profile.get("max_graph_call_sec", 0.0)
                    ),
                    "elapsed_sec": float(result.get("elapsed_sec", 0.0)),
                }
            )
        return {"warmed_images": per_image}

    def generate(
        self,
        image: str,
        *,
        image_mode: str = "gundam",
        max_length: int = 32768,
        prompt: str = "<image>document parsing.",
        pdf_page: int = 0,
        pdf_dpi: int = 200,
    ) -> dict[str, Any]:
        """Run OCR on one image/PDF, reusing warmed model and compiled graphs."""

        metrics = RuntimeMetrics()
        inputs = self.facade.prepare_inputs(
            image,
            prompt=prompt,
            image_mode=image_mode,
            pdf_page=pdf_page,
            pdf_dpi=pdf_dpi,
            metrics=metrics,
        )
        self._ensure_frontend(inputs, metrics=metrics)

        prompt_length = int(inputs.prompt_length)
        max_new_tokens = max_length - prompt_length
        if max_new_tokens < 1:
            raise ValueError(
                f"max_length ({max_length}) must exceed prompt length "
                f"({prompt_length})"
            )

        result = self.facade.generate_rbln_pa_swa_cached(
            inputs, max_new_tokens=max_new_tokens, metrics=metrics
        )
        result["input_summary"] = inputs.shape_summary()
        result["stage_timings"] = metrics.stages_to_list()
        result["component_timings"] = metrics.to_list()
        return result
