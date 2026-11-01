#!/usr/bin/env python3
"""Spatially aligned image metrics for SEVA outputs.

SEVA evaluates and generates on a resized canvas.  Reference images and masks must
be transformed to that same canvas; resizing a generated image back to the raw
camera image changes its field of view and invalidates pixel metrics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


ALIGNMENT_VERSION = "seva_canvas_v1"


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _resize(array: np.ndarray, height: int, width: int, is_mask: bool) -> np.ndarray:
    tensor = torch.from_numpy(array).float()
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    else:
        tensor = tensor.permute(2, 0, 1)[None]
    if is_mask:
        resized = torch.nn.functional.interpolate(tensor, (height, width), mode="nearest")
    else:
        resized = torch.nn.functional.interpolate(
            tensor, (height, width), mode="area", antialias=False
        )
    result = resized[0]
    if array.ndim == 2:
        return result[0].numpy()
    return result.permute(1, 2, 0).numpy()


def align_array_to_canvas(
    array: np.ndarray,
    canvas_size: tuple[int, int],
    mode: str = "crop",
    *,
    is_mask: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replicate SEVA's area-resize plus centered crop/pad preprocessing.

    ``canvas_size`` is ``(width, height)`` in PIL convention.
    """
    if mode not in {"crop", "pad", "stretch"}:
        raise ValueError(f"Unknown alignment mode: {mode}")
    source_h, source_w = array.shape[:2]
    target_w, target_h = canvas_size
    if source_w == target_w and source_h == target_h:
        return array.copy(), {
            "source_size": [source_w, source_h],
            "resized_size": [source_w, source_h],
            "canvas_size": [target_w, target_h],
            "mode": mode,
            "crop_left_top": [0, 0],
            "pad_left_top": [0, 0],
        }

    if mode == "stretch":
        resized_w, resized_h = target_w, target_h
    else:
        if mode == "crop":
            factor = max(target_w / source_w, target_h / source_h)
        else:
            factor = min(target_w / source_w, target_h / source_h)
        resized_h = int(math.ceil(factor * source_h))
        resized_w = int(math.ceil(factor * source_w))
    resized = _resize(array, resized_h, resized_w, is_mask)

    crop_left = crop_top = pad_left = pad_top = 0
    if mode != "pad":
        center_y = int(0.5 * resized_h)
        center_x = int(0.5 * resized_w)
        crop_top = max(0, center_y - target_h // 2)
        crop_left = max(0, center_x - target_w // 2)
        crop_top = min(crop_top, resized_h - target_h)
        crop_left = min(crop_left, resized_w - target_w)
        aligned = resized[
            crop_top : crop_top + target_h,
            crop_left : crop_left + target_w,
            ...,
        ]
    else:
        center_y = int(0.5 * resized_h)
        center_x = int(0.5 * resized_w)
        pad_top = max(0, target_h // 2 - center_y)
        pad_left = max(0, target_w // 2 - center_x)
        shape = (target_h, target_w) + (() if array.ndim == 2 else (array.shape[2],))
        aligned = np.zeros(shape, dtype=resized.dtype)
        aligned[
            pad_top : pad_top + resized_h,
            pad_left : pad_left + resized_w,
            ...,
        ] = resized

    return aligned, {
        "source_size": [source_w, source_h],
        "resized_size": [resized_w, resized_h],
        "canvas_size": [target_w, target_h],
        "mode": mode,
        "crop_left_top": [crop_left, crop_top],
        "pad_left_top": [pad_left, pad_top],
    }


def align_path_to_canvas(
    path: Path, canvas_size: tuple[int, int], mode: str = "crop"
) -> tuple[np.ndarray, dict[str, Any]]:
    return align_array_to_canvas(read_rgb(path), canvas_size, mode)


def psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    squared = (reference - prediction) ** 2
    if mask is not None:
        squared = squared[mask]
        if squared.size == 0:
            return float("nan")
    mse = float(squared.mean())
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def evaluate_pair(
    reference_path: Path,
    prediction_path: Path,
    mask_path: Path | None = None,
    alignment_mode: str = "crop",
) -> tuple[dict[str, float | None], dict[str, Any]]:
    prediction = read_rgb(prediction_path)
    canvas_size = (prediction.shape[1], prediction.shape[0])
    reference, geometry = align_path_to_canvas(reference_path, canvas_size, alignment_mode)
    if reference.shape != prediction.shape:
        raise RuntimeError(
            f"Aligned reference {reference.shape} != prediction {prediction.shape}: "
            f"{reference_path} vs {prediction_path}"
        )
    metrics: dict[str, float | None] = {
        "psnr": psnr(reference, prediction),
        "ssim": float(
            structural_similarity(reference, prediction, channel_axis=2, data_range=1.0)
        ),
        "object_psnr": None,
        "background_psnr": None,
    }
    if mask_path is not None and mask_path.is_file():
        mask, _ = align_array_to_canvas(
            read_mask(mask_path), canvas_size, alignment_mode, is_mask=True
        )
        foreground = mask > 0.5
        metrics["object_psnr"] = psnr(reference, prediction, foreground)
        metrics["background_psnr"] = psnr(reference, prediction, ~foreground)
    return metrics, geometry


def evaluation_summary(
    scene_id: str,
    mode: str,
    rows: list[dict[str, Any]],
    alignment_mode: str,
) -> dict[str, Any]:
    canvas_sizes = sorted(
        {tuple(row["alignment"]["canvas_size"]) for row in rows}
    )
    return {
        "scene_id": scene_id,
        "mode": mode,
        "target_count": len(rows),
        "spatial_alignment": {
            "version": ALIGNMENT_VERSION,
            "reference_transform": (
                "Reference RGB and masks are area/nearest resized and center-"
                f"{alignment_mode}ped to the native generated-image canvas."
            ),
            "prediction_transform": "none",
            "canvas_sizes": [list(size) for size in canvas_sizes],
            "invalid_legacy_method": (
                "Generated images were resized to the uncropped raw reference size; "
                "those legacy absolute image metrics must not be compared with these values."
            ),
        },
        "mean_psnr": finite_mean([row["psnr"] for row in rows]),
        "mean_ssim": finite_mean([row["ssim"] for row in rows]),
        "mean_object_psnr": finite_mean([row["object_psnr"] for row in rows]),
        "mean_background_psnr": finite_mean([row["background_psnr"] for row in rows]),
        "views": rows,
    }


def lpips_tensor_from_array(array: np.ndarray, short_side: int) -> torch.Tensor:
    height, width = array.shape[:2]
    scale = short_side / min(height, width)
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))
    resized = _resize(array, target_h, target_w, False)
    tensor = torch.from_numpy(resized * 2.0 - 1.0).permute(2, 0, 1)
    return tensor

