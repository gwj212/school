#!/usr/bin/env python3
"""Paired native-size evaluation of the consistency adapter on four unseen scenes."""

from __future__ import annotations

import csv
import argparse
import json
import math
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageChops
from scipy.stats import wilcoxon


SCENES = [
    "chair_620_101264_203062",
    "laptop_373_41558_82973",
    "plant_598_92218_184789",
    "toybus_121_14153_28986",
]
MANIFESTS = Path("/root/seva_vggt/config/test30_mvgen")
BASE = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/mvgen")
ADAPTED = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/mvgen_consistency_adapter_test4")
OUTPUT = ADAPTED / "evaluation"


def write_csv(path: Path, rows: list[dict]):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def fitted(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    resized = image.copy()
    resized.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def make_sheet(scene: str, visual: list[dict], path: Path):
    cell_w, cell_h, label_w, header_h = 260, 205, 90, 60
    titles = ("Real GT", "Original MVGen", "Adapted MVGen", "|adapted-original|")
    sheet = Image.new("RGB", (label_w + 4 * cell_w, header_h + 7 * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), scene, fill="black")
    for col, title in enumerate(titles):
        draw.text((label_w + col * cell_w + 8, 32), title, fill="black")
    for idx, item in enumerate(visual):
        y = header_h + idx * cell_h
        draw.multiline_text((5, y + 7), f"view {item['slot']:02d}\nΔP {item['delta_psnr']:+.2f}\nΔL {item['delta_lpips']:+.3f}", fill="black")
        diff = ImageChops.difference(item["base"], item["adapted"])
        diff = diff.point(lambda p: min(255, p * 4))
        for col, image in enumerate((item["gt"], item["base"], item["adapted"], diff)):
            sheet.paste(fitted(image, cell_w, cell_h - 10), (label_w + col * cell_w, y))
    sheet.save(path, quality=95)


def bootstrap_ci(values: np.ndarray, seed: int = 23):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(10000)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    global ADAPTED, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-root", type=Path, default=ADAPTED)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    ADAPTED = args.adapted_root
    OUTPUT = args.output or (ADAPTED / "evaluation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "comparisons").mkdir(exist_ok=True)
    sys.path[:0] = ["/root/seva_vggt/bridge", "/root/vggt"]
    import evaluate_render_test30_native as native

    device = torch.device("cuda:0")
    perceptual = lpips.LPIPS(net="alex").eval().to(device)
    rows = []
    for scene in SCENES:
        manifest = json.loads((MANIFESTS / f"{scene}.json").read_text())
        visual = []
        for slot in range(1, 8):
            gt_path = Path(manifest["views"][slot]["image"])
            base_path = BASE / scene / "images" / f"view_{slot:02d}_generated.png"
            adapted_path = ADAPTED / scene / "images" / f"view_{slot:02d}_generated.png"
            gt_pil, gt_tensor = native.open_rgb(str(gt_path))
            base_pil, base_tensor = native.open_rgb(str(base_path))
            adapted_pil, adapted_tensor = native.open_rgb(str(adapted_path))
            width, height = gt_pil.size
            base_aligned = native.resize_tensor(base_tensor.to(device), height, width)
            adapted_aligned = native.resize_tensor(adapted_tensor.to(device), height, width)
            gt_device = gt_tensor.to(device)
            base_metrics = native.image_metrics(gt_device, base_aligned, perceptual)
            adapted_metrics = native.image_metrics(gt_device, adapted_aligned, perceptual)
            row = {
                "scene": scene, "category": scene.split("_")[0], "slot": slot,
                "gt_path": str(gt_path), "original_path": str(base_path), "adapted_path": str(adapted_path),
                "gt_width": width, "gt_height": height,
                "original_width": base_pil.width, "original_height": base_pil.height,
                "adapted_width": adapted_pil.width, "adapted_height": adapted_pil.height,
            }
            for key in ("psnr", "ssim", "lpips", "mse"):
                row[f"original_{key}"] = finite(base_metrics[key])
                row[f"adapted_{key}"] = finite(adapted_metrics[key])
                row[f"delta_{key}"] = finite(adapted_metrics[key] - base_metrics[key])
            rows.append(row)
            visual.append({
                "slot": slot, "delta_psnr": row["delta_psnr"], "delta_lpips": row["delta_lpips"],
                "gt": gt_pil,
                "base": native.tensor_to_pil(base_aligned),
                "adapted": native.tensor_to_pil(adapted_aligned),
            })
        make_sheet(scene, visual, OUTPUT / "comparisons" / f"{scene}.jpg")

    write_csv(OUTPUT / "metrics_per_view.csv", rows)
    summary = {"protocol": {
        "scenes": SCENES, "views": len(rows), "seed": 23, "steps": 50, "cfg": 2,
        "comparison": "Each generated image is bicubic-resized once to the paired native GT width/height; RGB [0,1].",
        "training_categories": ["apple", "ball", "bicycle", "car", "couch", "cup", "teddybear", "vase"],
        "test_categories": ["chair", "laptop", "plant", "toybus"],
    }, "metrics": {}, "by_scene": {}}
    for metric in ("psnr", "ssim", "lpips", "mse"):
        original = np.array([r[f"original_{metric}"] for r in rows], dtype=float)
        adapted = np.array([r[f"adapted_{metric}"] for r in rows], dtype=float)
        delta = adapted - original
        higher = metric in ("psnr", "ssim")
        improvements = delta > 0 if higher else delta < 0
        try:
            pvalue = float(wilcoxon(delta).pvalue)
        except ValueError:
            pvalue = 1.0
        summary["metrics"][metric] = {
            "original_mean": float(original.mean()), "adapted_mean": float(adapted.mean()),
            "delta_mean": float(delta.mean()), "delta_median": float(np.median(delta)),
            "delta_ci95_bootstrap": bootstrap_ci(delta), "wilcoxon_p": pvalue,
            "improved_views": int(improvements.sum()), "total_views": len(rows),
        }
    for scene in SCENES:
        subset = [r for r in rows if r["scene"] == scene]
        summary["by_scene"][scene] = {
            metric: {
                "original_mean": float(np.mean([r[f"original_{metric}"] for r in subset])),
                "adapted_mean": float(np.mean([r[f"adapted_{metric}"] for r in subset])),
                "delta_mean": float(np.mean([r[f"delta_{metric}"] for r in subset])),
            } for metric in ("psnr", "ssim", "lpips")
        }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    m = summary["metrics"]
    report = "# MVGen 一致性适配器小样本验证\n\n"
    report += "测试集为4个训练未见类别、28个生成视角；原模型与适配器使用相同相机、seed=23、50步、CFG=2。图像仅一次双三次缩放到配对GT原始宽高后计算。\n\n"
    report += "|指标|原模型|适配后|变化|改善视角|95% bootstrap CI|Wilcoxon p|\n|---|---:|---:|---:|---:|---:|---:|\n"
    for key, label in (("psnr","PSNR↑"),("ssim","SSIM↑"),("lpips","LPIPS↓")):
        x=m[key]; ci=x["delta_ci95_bootstrap"]
        report += f"|{label}|{x['original_mean']:.5f}|{x['adapted_mean']:.5f}|{x['delta_mean']:+.5f}|{x['improved_views']}/{x['total_views']}|[{ci[0]:+.5f},{ci[1]:+.5f}]|{x['wilcoxon_p']:.4g}|\n"
    report += "\n注意：这是生成图层面的验证；只有继续通过冻结VGGT-G重建并渲染后，才能判断是否缓解漂浮Gaussian和多重结构。\n"
    (OUTPUT / "report_zh.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
