#!/usr/bin/env python3
"""Compare generated target views with held-out real images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    squared = (reference - prediction) ** 2
    if mask is not None:
        selected = squared[mask]
        if selected.size == 0:
            return float("nan")
        mse = float(selected.mean())
    else:
        mse = float(squared.mean())
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("generated_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if a manifest target is absent; otherwise evaluate generated files present.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    views = manifest["views"]
    rows = []
    for index in manifest["target_indices"]:
        view = views[index]
        generated_path = args.generated_dir / f"view_{int(view['index']):02d}.png"
        ground_truth_path = Path(view["ground_truth"])
        if not generated_path.is_file():
            if args.require_all:
                raise FileNotFoundError(generated_path)
            continue
        gt_image = Image.open(ground_truth_path).convert("RGB")
        gt = np.asarray(gt_image, dtype=np.float32) / 255.0
        generated = read_rgb(generated_path, gt_image.size)
        row = {
            "view": int(view["index"]),
            "pose_source": view["pose_source"],
            "psnr": psnr(gt, generated),
            "ssim": float(
                structural_similarity(gt, generated, channel_axis=2, data_range=1.0)
            ),
            "object_psnr": None,
            "background_psnr": None,
        }
        mask_path = Path(view["mask"]) if view.get("mask") else None
        if mask_path is not None and mask_path.is_file():
            mask = Image.open(mask_path).convert("L").resize(
                gt_image.size, Image.Resampling.NEAREST
            )
            foreground = np.asarray(mask) > 127
            row["object_psnr"] = psnr(gt, generated, foreground)
            row["background_psnr"] = psnr(gt, generated, ~foreground)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No generated target images found in {args.generated_dir}")

    summary = {
        "scene_id": manifest["scene_id"],
        "mode": manifest["mode"],
        "target_count": len(rows),
        "mean_psnr": finite_mean([row["psnr"] for row in rows]),
        "mean_ssim": finite_mean([row["ssim"] for row in rows]),
        "mean_object_psnr": finite_mean(
            [row["object_psnr"] for row in rows if row["object_psnr"] is not None]
        ),
        "mean_background_psnr": finite_mean(
            [row["background_psnr"] for row in rows if row["background_psnr"] is not None]
        ),
        "views": rows,
    }
    output = args.output or args.generated_dir.parent / "image_metrics.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
