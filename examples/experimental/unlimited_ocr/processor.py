"""Runtime input preparation for Unlimited-OCR execution.

This module intentionally mirrors the Hugging Face remote-code preprocessing while
keeping tensors on the caller-selected device.  That gives the runtime a stable
preprocessing boundary before individual modules are executed with RBLN-compiled
backends.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision import transforms

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
BOS_TOKEN_ID = 0


@dataclass(frozen=True)
class ImageMode:
    """Image preprocessing preset used by Unlimited-OCR/SGLang."""

    name: str
    base_size: int
    image_size: int
    crop_mode: bool


IMAGE_MODES: dict[str, ImageMode] = {
    "gundam": ImageMode("gundam", base_size=1024, image_size=640, crop_mode=True),
    "base": ImageMode("base", base_size=1024, image_size=1024, crop_mode=False),
    "small": ImageMode("small", base_size=640, image_size=640, crop_mode=False),
    "tiny": ImageMode("tiny", base_size=512, image_size=512, crop_mode=False),
    "large": ImageMode("large", base_size=1280, image_size=1280, crop_mode=False),
}

MULTI_IMAGE_ALLOWED_MODES = frozenset({"tiny", "small", "base"})

# Unlimited-OCR's own spec allows up to 32 local crops (vs DeepSeek-OCR's 6);
# see official vLLM's `unlimited_ocr.py` (`_UNLIMITED_OCR_MAX_CROPS`).
MAX_CROPS = 32


def validate_image_count_for_mode(image_mode: str, image_count: int) -> None:
    """Validate Baidu Unlimited-OCR's image-mode/multi-image contract."""

    if image_mode not in IMAGE_MODES:
        raise ValueError(
            f"unknown image_mode={image_mode!r}; choices={sorted(IMAGE_MODES)}"
        )
    if image_count <= 0:
        raise ValueError(f"image_count must be positive, got {image_count}")
    if image_count > 1 and image_mode not in MULTI_IMAGE_ALLOWED_MODES:
        raise ValueError(
            "Unlimited-OCR multi-image prompts are only supported for "
            f"{sorted(MULTI_IMAGE_ALLOWED_MODES)}; got image_mode={image_mode!r} "
            f"with image_count={image_count}"
        )


@dataclass
class OCRInputs:
    """Prepared model inputs for one-image OCR."""

    input_ids: torch.Tensor
    images: list[tuple[torch.Tensor, torch.Tensor]]
    images_seq_mask: torch.Tensor
    images_spatial_crop: torch.Tensor
    prompt: str
    image_path: str
    mode: ImageMode

    def to_generation_kwargs(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids.unsqueeze(0),
            "images": self.images,
            "images_seq_mask": self.images_seq_mask.unsqueeze(0),
            "images_spatial_crop": self.images_spatial_crop,
        }

    @property
    def prompt_length(self) -> int:
        return int(self.input_ids.numel())

    def shape_summary(self) -> dict[str, Any]:
        image_crop, image_ori = self.images[0]
        return {
            "input_ids": list(self.input_ids.shape),
            "images_seq_mask": list(self.images_seq_mask.shape),
            "images_seq_true": int(self.images_seq_mask.sum().item()),
            "images_crop": list(image_crop.shape),
            "images_ori": list(image_ori.shape),
            "images_spatial_crop": self.images_spatial_crop.tolist(),
            "mode": self.mode.__dict__,
        }


class BasicImageTransform:
    """Equivalent to the HF remote-code BasicImageTransform."""

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ) -> None:
        self.mean = mean
        self.std = std
        pipeline: list[Any] = [transforms.ToTensor()]
        if normalize:
            pipeline.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(pipeline)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"image transform returned {type(tensor)!r}")
        return tensor


def load_image(
    image_path: str | Path,
    *,
    pdf_page: int = 1,
    pdf_dpi: int = 200,
) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"input image/PDF path does not exist: {path}")
    if path.suffix.lower() == ".pdf":
        return load_pdf_page(path, page=pdf_page, dpi=pdf_dpi)
    image = Image.open(path)
    return ImageOps.exif_transpose(image).convert("RGB")


def load_pdf_page(pdf_path: Path, *, page: int = 1, dpi: int = 200) -> Image.Image:
    if page < 1:
        raise ValueError(f"pdf_page must be >= 1, got {page}")
    if dpi <= 0:
        raise ValueError(f"pdf_dpi must be positive, got {dpi}")
    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF input requires PyMuPDF (`fitz`) in this environment. "
            "Install pymupdf or convert the PDF page to an image first."
        ) from exc

    document = fitz.open(pdf_path)
    try:
        page_index = page - 1
        if page_index >= document.page_count:
            raise ValueError(
                f"pdf_page={page} is out of range for {pdf_path} "
                f"with {document.page_count} pages"
            )
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pixmap = document[page_index].get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return ImageOps.exif_transpose(image).convert("RGB")
    finally:
        document.close()


def text_encode(
    tokenizer: Any, text: str, bos: bool = True, eos: bool = False
) -> list[int]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if bos:
        tokens = [BOS_TOKEN_ID, *tokens]
    if eos:
        tokens = [*tokens, int(tokenizer.eos_token_id)]
    return tokens


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 2,
    max_num: int = MAX_CROPS,
    image_size: int = 640,
    use_thumbnail: bool = False,
) -> tuple[list[Image.Image], tuple[int, int]]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images, target_aspect_ratio


def _format_plain_prompt(prompt: str) -> str:
    """Approximate the HF plain conversation path without importing remote code."""
    return prompt.strip()


def prepare_single_image_inputs(
    tokenizer: Any,
    image_path: str | Path,
    prompt: str = "<image>document parsing.",
    image_mode: str = "gundam",
    dtype: torch.dtype = torch.bfloat16,
    pdf_page: int = 1,
    pdf_dpi: int = 200,
) -> OCRInputs:
    """Prepare CPU tensors for one-image Unlimited-OCR generation."""
    validate_image_count_for_mode(image_mode, 1)

    mode = IMAGE_MODES[image_mode]
    image = load_image(image_path, pdf_page=pdf_page, pdf_dpi=pdf_dpi)
    formatted_prompt = _format_plain_prompt(prompt)
    transform = BasicImageTransform()

    patch_size = 16
    downsample_ratio = 4
    text_splits = formatted_prompt.split(IMAGE_TOKEN)
    if len(text_splits) < 2:
        text_splits = [formatted_prompt, ""]

    tokenized_str: list[int] = []
    images_seq_mask: list[bool] = []
    images_list: list[torch.Tensor] = []
    images_crop_list: list[torch.Tensor] = []
    images_spatial_crop: list[list[int]] = []

    tokenized_sep = text_encode(tokenizer, text_splits[0], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    if mode.crop_mode:
        if image.size[0] <= mode.image_size and image.size[1] <= mode.image_size:
            crop_ratio = (1, 1)
            images_crop_raw: list[Image.Image] = []
        else:
            images_crop_raw, crop_ratio = dynamic_preprocess(
                image, max_num=MAX_CROPS, image_size=mode.image_size
            )

        global_view = ImageOps.pad(
            image,
            (mode.base_size, mode.base_size),
            color=tuple(int(x * 255) for x in transform.mean),
        )
        images_list.append(transform(global_view).to(dtype))
        width_crop_num, height_crop_num = crop_ratio
        images_spatial_crop.append([width_crop_num, height_crop_num])
        if width_crop_num > 1 or height_crop_num > 1:
            images_crop_list.extend(transform(img).to(dtype) for img in images_crop_raw)

        num_queries = math.ceil((mode.image_size // patch_size) / downsample_ratio)
        num_queries_base = math.ceil((mode.base_size // patch_size) / downsample_ratio)
        tokenized_image = (
            [IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]
        ) * num_queries_base
        tokenized_image += [IMAGE_TOKEN_ID]
        if width_crop_num > 1 or height_crop_num > 1:
            tokenized_image += (
                [IMAGE_TOKEN_ID] * (num_queries * width_crop_num) + [IMAGE_TOKEN_ID]
            ) * (num_queries * height_crop_num)
    else:
        if mode.image_size <= 640:
            image = image.resize((mode.image_size, mode.image_size))
        global_view = ImageOps.pad(
            image,
            (mode.image_size, mode.image_size),
            color=tuple(int(x * 255) for x in transform.mean),
        )
        images_list.append(transform(global_view).to(dtype))
        images_spatial_crop.append([1, 1])
        num_queries = math.ceil((mode.image_size // patch_size) / downsample_ratio)
        tokenized_image = (
            [IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]
        ) * num_queries
        tokenized_image += [IMAGE_TOKEN_ID]

    tokenized_str += tokenized_image
    images_seq_mask += [True] * len(tokenized_image)

    tail_text = text_splits[-1] if len(text_splits) > 1 else ""
    tokenized_sep = text_encode(tokenizer, tail_text, bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    tokenized_str = [BOS_TOKEN_ID, *tokenized_str]
    images_seq_mask = [False, *images_seq_mask]

    input_ids = torch.tensor(tokenized_str, dtype=torch.long)
    seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)
    images_ori = torch.stack(images_list, dim=0)
    spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
    if images_crop_list:
        images_crop = torch.stack(images_crop_list, dim=0)
    else:
        images_crop = torch.zeros((1, 3, mode.base_size, mode.base_size), dtype=dtype)

    return OCRInputs(
        input_ids=input_ids,
        images=[(images_crop, images_ori)],
        images_seq_mask=seq_mask,
        images_spatial_crop=spatial_crop,
        prompt=formatted_prompt,
        image_path=str(image_path),
        mode=mode,
    )
