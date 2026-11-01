#!/usr/bin/env python3
"""独立计算两张图片的 PSNR。

使用方法：只需修改下面“用户输入”区域的两个图片路径，然后运行本脚本。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# ============================== 用户输入 ==============================
IMAGE_PATH_1 = "/path/to/image_1.png"
IMAGE_PATH_2 = "/path/to/image_2.png"

# None：以图片1经过方向校正后的尺寸作为统一评测尺寸。
# 也可以显式设置为 (宽, 高)，例如 (518, 518)。
TARGET_SIZE = None
# =====================================================================


def load_as_rgb(path: str | Path) -> tuple[Image.Image, dict[str, object]]:
    """读取图片，校正 EXIF 方向并统一为 8-bit sRGB/RGB。"""
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    with Image.open(image_path) as source:
        original_mode = source.mode
        original_size = source.size
        image = ImageOps.exif_transpose(source)
        oriented_size = image.size

        # 对带透明通道的图片使用白色背景，避免直接丢弃 alpha 后产生黑底。
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")

    metadata = {
        "path": str(image_path.resolve()),
        "original_mode": original_mode,
        "original_size": list(original_size),
        "oriented_size": list(oriented_size),
    }
    return image, metadata


def resize_for_evaluation(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """使用同一双三次插值规则缩放到评测尺寸。"""
    if image.size == size:
        return image
    return image.resize(size, resample=Image.Resampling.BICUBIC)


def calculate_psnr(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """按 8-bit RGB、data_range=255 计算全图像素 PSNR。"""
    if first.shape != second.shape:
        raise ValueError(f"预处理后尺寸仍不一致: {first.shape} vs {second.shape}")

    difference = first.astype(np.float64) - second.astype(np.float64)
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    if mse == 0.0:
        return math.inf, mse
    psnr = 10.0 * math.log10((255.0**2) / mse)
    return psnr, mse


def main() -> None:
    image_1, metadata_1 = load_as_rgb(IMAGE_PATH_1)
    image_2, metadata_2 = load_as_rgb(IMAGE_PATH_2)

    if TARGET_SIZE is None:
        evaluation_size = image_1.size
        size_policy = "image_2_resized_to_image_1"
    else:
        evaluation_size = (int(TARGET_SIZE[0]), int(TARGET_SIZE[1]))
        if evaluation_size[0] <= 0 or evaluation_size[1] <= 0:
            raise ValueError(f"TARGET_SIZE 必须为正整数尺寸，当前值: {TARGET_SIZE}")
        size_policy = "both_resized_to_explicit_target"

    processed_1 = resize_for_evaluation(image_1, evaluation_size)
    processed_2 = resize_for_evaluation(image_2, evaluation_size)
    array_1 = np.asarray(processed_1, dtype=np.uint8)
    array_2 = np.asarray(processed_2, dtype=np.uint8)
    psnr, mse = calculate_psnr(array_1, array_2)

    result = {
        "metric": "PSNR",
        "psnr_db": "inf" if math.isinf(psnr) else psnr,
        "mse": mse,
        "data_range": 255,
        "channels": "RGB",
        "evaluation_size_wh": list(evaluation_size),
        "size_policy": size_policy,
        "resize_interpolation": "bicubic",
        "alpha_background": "white",
        "image_1": metadata_1,
        "image_2": metadata_2,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
