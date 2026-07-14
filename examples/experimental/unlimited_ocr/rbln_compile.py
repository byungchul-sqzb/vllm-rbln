"""RBLN compile helpers for Unlimited-OCR experimental components."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from vllm_rbln.torch_compile_backend import logged_rbln_backend, set_warmup_active


@dataclass(frozen=True)
class CompiledComponent:
    """A compiled callable plus its backend label."""

    name: str
    backend: str
    module: Any


class SignatureCompiledModule:
    """Dispatch to one compiled module per static input signature."""

    def __init__(
        self, modules_by_signature: dict[tuple[tuple[int, ...], ...], Any]
    ) -> None:
        self._modules_by_signature = modules_by_signature

    def __call__(self, *inputs: torch.Tensor) -> Any:
        signature = tuple(tuple(input_tensor.shape) for input_tensor in inputs)
        if signature not in self._modules_by_signature:
            raise ValueError(
                f"no RBLN compiled module for signature={signature}; "
                f"compiled={sorted(self._modules_by_signature)}"
            )
        return self._modules_by_signature[signature](*inputs)


class ShapeCompiledModule:
    """Backward-compatible single-input shape dispatcher."""

    def __init__(self, modules_by_shape: dict[tuple[int, ...], Any]) -> None:
        self._dispatcher = SignatureCompiledModule(
            {(shape,): module for shape, module in modules_by_shape.items()}
        )

    def __call__(self, input_tensor: torch.Tensor) -> Any:
        return self._dispatcher(input_tensor)


def compile_sam_shapes(
    sam_model: torch.nn.Module,
    image_shapes: list[tuple[int, int, int, int]],
    dtype: torch.dtype,
) -> CompiledComponent:
    """Compile the SAM encoder for one or more static image shapes."""
    return _compile_single_input_shapes(
        name="sam_model",
        module=sam_model,
        shapes=image_shapes,
        dtype=dtype,
    )


def compile_vision_signatures(
    vision_model: torch.nn.Module,
    signatures: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    dtype: torch.dtype,
) -> CompiledComponent:
    """Compile the CLIP-like vision encoder for image/SAM feature signatures."""
    vision_model.eval()
    modules_by_signature: dict[tuple[tuple[int, ...], ...], Any] = {}
    device = _module_device(vision_model)
    for image_shape, sam_shape in dict.fromkeys(signatures):
        example_images = torch.zeros(image_shape, dtype=dtype, device=device)
        example_sam = torch.zeros(sam_shape, dtype=dtype, device=device)
        modules_by_signature[(image_shape, sam_shape)] = _compile_one(
            vision_model, example_images, example_sam, component="vision_model"
        )
    return CompiledComponent(
        name="vision_model",
        backend="rbln",
        module=SignatureCompiledModule(modules_by_signature),
    )


def compile_projector_shapes(
    projector: torch.nn.Module, shapes: list[tuple[int, int, int]], dtype: torch.dtype
) -> CompiledComponent:
    """Compile the Unlimited-OCR projector for one or more static input shapes."""
    return _compile_single_input_shapes(
        name="projector",
        module=projector,
        shapes=shapes,
        dtype=dtype,
    )


def compile_embed_tokens_shapes(
    embed_tokens: torch.nn.Module, shapes: list[tuple[int, int]]
) -> CompiledComponent:
    """Compile token embedding lookup for prompt and decode token shapes."""
    embed_tokens.eval()
    modules_by_shape: dict[tuple[int, ...], Any] = {}
    device = _module_device(embed_tokens)
    for shape in dict.fromkeys(shapes):
        example_input = torch.zeros(shape, dtype=torch.long, device=device)
        modules_by_shape[shape] = _compile_one(
            embed_tokens, example_input, component="embed_tokens"
        )
    return CompiledComponent(
        name="embed_tokens",
        backend="rbln",
        module=ShapeCompiledModule(modules_by_shape),
    )


def _compile_single_input_shapes(
    name: str,
    module: torch.nn.Module,
    shapes: list[tuple[int, ...]],
    dtype: torch.dtype,
) -> CompiledComponent:
    module.eval()
    modules_by_shape: dict[tuple[int, ...], Any] = {}
    device = _module_device(module)
    for shape in dict.fromkeys(shapes):
        example_input = torch.zeros(shape, dtype=dtype, device=device)
        modules_by_shape[shape] = _compile_one(module, example_input, component=name)
    return CompiledComponent(
        name=name,
        backend="rbln",
        module=ShapeCompiledModule(modules_by_shape),
    )


def available_rbln_device_ids() -> tuple[int, ...]:
    """Return visible RBLN device ids, honoring RBLN_DEVICES when set.

    The standalone Unlimited-OCR example does not go through the vLLM worker
    that normally prepares RBLN_DEVICES/VLLM_RBLN_TP_SIZE.  This helper mirrors
    that intent locally: use the visible device list if the user constrained it,
    otherwise ask rebel for the installed NPU count.
    """

    explicit = os.environ.get("RBLN_DEVICES")
    if explicit:
        ids: list[int] = []
        for raw in explicit.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                ids.append(int(raw))
            except ValueError as exc:
                raise ValueError(f"invalid RBLN_DEVICES entry: {raw!r}") from exc
        return tuple(ids) or (0,)

    try:
        import rebel

        count = int(rebel.device_count())
        if count > 0:
            return tuple(range(count))
    except Exception:
        pass

    # Last-resort probe for older rebel builds without device_count().
    try:
        import rebel

        ids = []
        for device_id in range(16):
            try:
                if rebel.get_npu_name(device_id):
                    ids.append(device_id)
            except Exception:
                break
        if ids:
            return tuple(ids)
    except Exception:
        pass
    return (0,)


def rbln_num_devices() -> int:
    """Number of NPUs to request from torch-rbln for compiled graphs."""

    return max(1, len(available_rbln_device_ids()))


def rbln_compile_options(component: str | None = None) -> dict[str, Any]:
    """Compile options shared by Unlimited-OCR RBLN graphs.

    vLLM-RBLN uses RBLN_DEVICES plus an RBLN device count to avoid pinning
    every compiled graph to device 0.  torch-rbln's current option name is
    ``num_devices`` (``tensor_parallel_size`` is deprecated), so use it here.

    Component policy is intentionally conservative for the vision frontend:
    CLIP/vision, projector, and token embedding graphs are small or currently
    safer as single-device compiled graphs.  LLM decode uses all visible NPUs.
    """

    num_devices = _component_num_devices(component)
    options: dict[str, Any] = {
        "num_devices": num_devices,
        "use_global_ctx": True,
    }
    if num_devices == 1:
        # Keep the previous stable single-device placement for components that
        # cannot safely compile across multiple devices yet.
        options["global_device_id"] = available_rbln_device_ids()[0]
    return options


def _component_num_devices(component: str | None) -> int:
    requested = rbln_num_devices()
    if component in {"vision_model", "projector", "embed_tokens"}:
        return 1
    return requested


def _compile_one(
    module: torch.nn.Module,
    *example_inputs: torch.Tensor,
    component: str | None = None,
) -> Any:
    options = rbln_compile_options(component)
    compiled = torch.compile(
        module,
        dynamic=False,
        fullgraph=True,
        backend=logged_rbln_backend,
        options=options,
    )
    set_warmup_active(True)
    try:
        with torch.no_grad():
            compiled(*example_inputs)
    finally:
        set_warmup_active(False)
    return compiled


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cpu")
