"""Production-facing CLI for Unlimited-OCR inference experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.experimental.unlimited_ocr.metrics import RuntimeMetrics
from examples.experimental.unlimited_ocr.rbln_compile import (
    available_rbln_device_ids,
    rbln_compile_options,
    rbln_num_devices,
)
from examples.experimental.unlimited_ocr.runtime import (
    ExecutionPlan,
    load_unlimited_ocr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baidu/Unlimited-OCR")
    parser.add_argument("--image", help="image or PDF path for OCR execution")
    parser.add_argument(
        "--pdf-page",
        type=int,
        default=1,
        help="1-based PDF page to render when --image points to a PDF",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help="PDF render DPI when --image points to a PDF",
    )
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--image-mode", default="gundam")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="total generation runs including warmup runs",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="initial generation runs excluded from component timings",
    )
    parser.add_argument(
        "--rbln-components",
        default="all",
        help=(
            "comma-separated OCR components to run on RBLN, e.g. "
            "all, quality_safe, vision_frontend, language_decode, or "
            "sam_model,vision_model,projector,embed_tokens. "
            "The all alias means every NPU-compilable component "
            "(sam_model, vision_model, projector, embed_tokens, "
            "language_decode). Language prefill stays eager."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help=(
            "path for the full JSON execution report; defaults to "
            "<input_stem>_result.json next to the input image/PDF"
        ),
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        help=(
            "path for the plain OCR text; defaults to the JSON path with "
            "a .txt suffix"
        ),
    )
    parser.add_argument(
        "--detail-timings",
        action="store_true",
        help="also print low-level component timings after the stage summary",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="load model and print module boundaries without running OCR generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_benchmark_args(args.benchmark_runs, args.warmup_runs)
    _validate_generation_args(args.max_length)
    metrics = RuntimeMetrics()
    try:
        rbln_components = _parse_components(args.rbln_components)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"error: {exc}") from exc
    language_backend = _language_backend_from_components(rbln_components)
    compile_components = _component_compile_set(rbln_components)
    with metrics.measure("model_load", "cpu", args.model):
        loaded = load_unlimited_ocr(args.model, torch_dtype=torch.bfloat16)
    facade = loaded.facade
    execution_plan = ExecutionPlan.from_cli(compile_components, language_backend=language_backend)
    backend_summary = facade.backend_summary().to_dict()
    if language_backend == "rbln-pa-swa-cached":
        backend_summary["language_decode"] = language_backend

    result = {
        "model": args.model,
        "runtime_contract": {
            "quality_reference": (
                "none_runtime_direct"
                if language_backend == "rbln-pa-swa-cached"
                else "hf_generate_cache"
            ),
            "language_decode_component_requested": "language_decode" in rbln_components,
            "language_decode": (
                "rbln-pa-swa-cached"
                if language_backend == "rbln-pa-swa-cached"
                else "cpu"
            ),
            "actual_language_decode": (
                "rbln-pa-swa-cached"
                if language_backend == "rbln-pa-swa-cached"
                else "cpu"
            ),
            "full_decoder_npu": language_backend == "rbln-pa-swa-cached",
            "rbln_prefill_backend": (
                "native_vllm_rbln_prefill_compile"
                if language_backend == "rbln-pa-swa-cached"
                else None
            ),
            "rbln_decode_backend": (
                "native_vllm_rbln_pa_swa_cached"
                if language_backend == "rbln-pa-swa-cached"
                else "blocked"
            ),
            "production_pa_swa_kv_cache": (
                "production_style_single_request"
                if language_backend == "rbln-pa-swa-cached"
                else "blocked"
            ),
            "rbln_components_are_frontend_only": False,
            "rbln_components_requested": sorted(rbln_components),
            "rbln_quality_profile": _rbln_quality_profile(rbln_components),
            "rbln_all_components": sorted(_all_rbln_components()),
            "rbln_cpu_only_components": [],
            "rbln_device_ids": list(available_rbln_device_ids()),
            "rbln_num_devices": rbln_num_devices(),
            "rbln_compile_options": {
                "default": rbln_compile_options(),
                "vision_model": rbln_compile_options("vision_model"),
                "language_decode": rbln_compile_options("language_decode"),
            },
        },
        "execution_plan": execution_plan.to_dict(),
        "backend_summary": backend_summary,
        "module_summary": facade.module_summary(),
        "blockers": [],
    }

    if args.image:
        inputs = facade.prepare_inputs(
            args.image,
            prompt=args.prompt,
            image_mode=args.image_mode,
            pdf_page=args.pdf_page,
            pdf_dpi=args.pdf_dpi,
            metrics=metrics,
        )
        result["input_summary"] = inputs.shape_summary()
        if compile_components:
            facade.enable_rbln_components(compile_components, inputs, metrics=metrics)
            result["rbln_components_compiled"] = sorted(compile_components)
            backend_summary = facade.backend_summary().to_dict()
            if language_backend == "rbln-pa-swa-cached":
                backend_summary["language_decode"] = language_backend
            result["backend_summary"] = backend_summary
        if not args.inspect_only:
            benchmark = _run_generation_benchmark(
                facade=facade,
                inputs=inputs,
                max_length=args.max_length,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                total_runs=args.benchmark_runs,
                warmup_runs=args.warmup_runs,
                metrics=metrics,
                language_backend=language_backend,
            )
            result["generation"] = benchmark["last_measured"]
            result["generation_benchmark"] = benchmark
            result["rbln_components_executed_in_last_generation"] = sorted(
                benchmark["last_measured"].get("rbln_generation_calls", {})
            )
    result["stage_timings"] = metrics.stages_to_list()
    result["component_timings"] = metrics.to_list()
    print(metrics.render_stage_table())
    if args.detail_timings:
        print()
        print(metrics.render_table())
    print()
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    _write_outputs(
        result,
        json_text=text,
        image_path=args.image,
        output_json=args.output_json,
        output_txt=args.output_txt,
    )


def _write_outputs(
    result: dict,
    *,
    json_text: str,
    image_path: str | None,
    output_json: Path | None,
    output_txt: Path | None,
) -> None:
    json_path = output_json or _default_output_json_path(image_path)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_text + "\n", encoding="utf-8")

    ocr_text = result.get("generation", {}).get("text")
    if not isinstance(ocr_text, str):
        return

    txt_path = output_txt
    if txt_path is None:
        if json_path is None:
            return
        txt_path = json_path.with_suffix(".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(ocr_text + "\n", encoding="utf-8")


def _default_output_json_path(image_path: str | None) -> Path | None:
    if not image_path:
        return None
    path = Path(image_path)
    return path.with_name(f"{path.stem}_result.json")


def _run_generation_benchmark(
    facade,
    inputs,
    max_length: int,
    no_repeat_ngram_size: int,
    total_runs: int,
    warmup_runs: int,
    metrics: RuntimeMetrics,
    language_backend: str = "cpu",
) -> dict:
    prompt_length = int(inputs.prompt_length)
    max_new_tokens = _max_new_tokens_from_max_length(max_length, prompt_length)
    measured = []
    warmup = []
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup_runs
        if language_backend == "rbln-pa-swa-cached":
            generation = facade.generate_rbln_pa_swa_cached(
                inputs,
                max_new_tokens=max_new_tokens,
                metrics=None if is_warmup else metrics,
            )
        else:
            generation = facade.generate_hf(
                inputs,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                metrics=None if is_warmup else metrics,
            )
        generation["max_length"] = max_length
        generation["effective_max_new_tokens"] = max_new_tokens
        generation["run_index"] = run_idx + 1
        generation["warmup"] = is_warmup
        if is_warmup:
            warmup.append(generation)
        else:
            measured.append(generation)

    elapsed_values = [item["elapsed_sec"] for item in measured]
    return {
        "total_runs": total_runs,
        "warmup_runs": warmup_runs,
        "measured_runs": len(measured),
        "warmup": warmup,
        "measured": measured,
        "last_measured": measured[-1],
        "measured_elapsed_sec_avg": sum(elapsed_values) / len(elapsed_values),
        "measured_elapsed_sec_min": min(elapsed_values),
        "measured_elapsed_sec_max": max(elapsed_values),
    }


def _validate_benchmark_args(total_runs: int, warmup_runs: int) -> None:
    if total_runs < 1:
        raise argparse.ArgumentTypeError("--benchmark-runs must be >= 1")
    if warmup_runs < 0:
        raise argparse.ArgumentTypeError("--warmup-runs must be >= 0")
    if warmup_runs >= total_runs:
        raise argparse.ArgumentTypeError(
            "--warmup-runs must be smaller than --benchmark-runs so at least "
            "one measured run remains"
        )


def _validate_generation_args(max_length: int) -> None:
    if max_length < 1:
        raise argparse.ArgumentTypeError("--max-length must be >= 1")


def _max_new_tokens_from_max_length(max_length: int, prompt_length: int) -> int:
    max_new_tokens = max_length - prompt_length
    if max_new_tokens < 1:
        raise argparse.ArgumentTypeError(
            f"--max-length ({max_length}) must be greater than prompt length "
            f"({prompt_length})"
        )
    return max_new_tokens


def _parse_components(value: str) -> set[str]:
    components = {part.strip() for part in value.split(",") if part.strip()}
    quality_safe = _quality_safe_components()
    vision_frontend = {
        "vision_model",
        "projector",
    }
    aliases = {
        "quality_safe": quality_safe,
        "vision_frontend": vision_frontend,
        "llm_decoder": {"language_decode"},
        "language": {"language_decode"},
        "all": _all_rbln_components(),
    }
    expanded: set[str] = set()
    for component in components:
        expanded.update(aliases.get(component, {component}))
    components = expanded
    supported = {
        "sam_model",
        "vision_model",
        "projector",
        "embed_tokens",
        "language_decode",
        "llm_decoder",
        "language",
        "quality_safe",
        "vision_frontend",
        "all",
    }
    unknown = components - supported
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported RBLN components: {sorted(unknown)}; "
            f"supported: {sorted(supported)}"
        )
    return components


def _language_backend_from_components(components: set[str]) -> str:
    if "language_decode" in components:
        return "rbln-pa-swa-cached"
    return "cpu"


def _component_compile_set(components: set[str]) -> set[str]:
    return components - {"language_decode"}


def _rbln_quality_profile(components: set[str]) -> str:
    quality_safe = _quality_safe_components()
    if not components:
        return "cpu_reference"
    if components <= quality_safe:
        return "quality_safe_exact_text_validated_on_hf_generate_path"
    if "language_decode" in components and components <= _all_rbln_components():
        return "npu_frontend_and_decode_runtime"
    return "experimental_vision_frontend_may_change_ocr_coordinates"


def _quality_safe_components() -> set[str]:
    return {"embed_tokens"}


def _all_rbln_components() -> set[str]:
    return {
        "sam_model",
        "vision_model",
        "projector",
        "embed_tokens",
        "language_decode",
    }


if __name__ == "__main__":
    main()
