#!/usr/bin/env python3
"""Summarize the held-out MVGenMaster 3D-adapter -> frozen VGGT-G pilot."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import wilcoxon


SCENES = [
    "chair_620_101264_203062",
    "laptop_373_41558_82973",
    "bicycle_62_4318_10726",
    "car_336_34863_64261",
    "bottle_601_93006_187080",
]
BASE = Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
ROOT = Path("/root/seva_vggt/outputs/joint_finetune_small")
ADAPT_GEN = ROOT / "adapted_mvgen_test5"
ADAPT_RECON = ROOT / "adapted_vggt_test5"
ADAPT_EVAL = ROOT / "adapted_render_test5"
OUT = ROOT / "analysis_test5"
MANIFESTS = Path("/root/seva_vggt/config/joint_small_test5")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def filtered_xyz(path: Path) -> np.ndarray:
    reconstruction = torch.load(path, map_location="cpu")
    gaussian = reconstruction["gaussians"]
    xyz = gaussian["xyz"][0].reshape(-1, 3).float().cpu().numpy()
    opacity = gaussian["opacity"][0].reshape(-1).float().cpu().numpy()
    scale = gaussian["scale"][0].reshape(-1, 3).float().cpu().numpy()
    finite = np.isfinite(xyz).all(1) & np.isfinite(opacity) & np.isfinite(scale).all(1)
    pre = finite & (scale > 0).all(1) & (opacity >= 0.10)
    scale_max = scale.max(1)
    limit = float(np.percentile(scale_max[pre], 99.9))
    return xyz[pre & (scale_max <= limit)]


def bootstrap_ci(deltas: np.ndarray, seed=23, n=10000):
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(deltas, size=(n, len(deltas)), replace=True), axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def fitted(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    copy = image.convert("RGB").copy(); copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(copy, ((width-copy.width)//2, (height-copy.height)//2))
    return canvas


def make_sheet(scene: str, output: Path):
    manifest = json.loads((MANIFESTS / f"{scene}.json").read_text())
    cell_w, cell_h, label_w, header_h = 220, 170, 72, 58
    titles = ["GT", "Original MVGen", "Adapted MVGen", "Original VGGT render", "Adapted VGGT render"]
    sheet = Image.new("RGB", (label_w + cell_w*len(titles), header_h + cell_h*7), "white")
    draw = ImageDraw.Draw(sheet); draw.text((5, 5), scene, fill="black")
    for j, title in enumerate(titles): draw.text((label_w+j*cell_w+5, 31), title, fill="black")
    for row, slot in enumerate(range(1, 8)):
        y = header_h + row*cell_h; draw.text((5, y+10), f"view {slot:02d}", fill="black")
        paths = [
            Path(manifest["views"][slot]["image"]),
            BASE / "mvgen" / scene / "images" / f"view_{slot:02d}_generated.png",
            ADAPT_GEN / scene / "images" / f"view_{slot:02d}_generated.png",
            BASE / "evaluation_native_v2" / "scenes" / scene / "mvgen_to_gt_native" / f"view_{slot:02d}_render.png",
            ADAPT_EVAL / "scenes" / scene / "mvgen_to_gt_native" / f"view_{slot:02d}_render.png",
        ]
        for col, path in enumerate(paths):
            with Image.open(path) as im: tile = fitted(im, cell_w, cell_h-18)
            sheet.paste(tile, (label_w+col*cell_w, y))
    sheet.save(output, quality=95)


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "comparisons").mkdir(exist_ok=True)
    sys.path[:0] = ["/root/seva_vggt/bridge"]
    import run_pipeline as rp

    base_rows = read_csv(BASE / "evaluation_native_v2" / "metrics_per_view_native.csv")
    adapt_rows = read_csv(ADAPT_EVAL / "metrics_per_view_native.csv")
    key = lambda r: (r["scene"], int(r["slot"]))
    base = {key(r): r for r in base_rows if r["scene"] in SCENES and int(r["slot"]) > 0}
    adapt = {key(r): r for r in adapt_rows if r["scene"] in SCENES and int(r["slot"]) > 0}
    if set(base) != set(adapt) or len(base) != 35:
        raise RuntimeError(f"paired rows mismatch baseline={len(base)} adapted={len(adapt)}")

    metrics = {
        "image_psnr": ("mv_generated_to_real_gt_psnr", 1),
        "image_ssim": ("mv_generated_to_real_gt_ssim", 1),
        "image_lpips": ("mv_generated_to_real_gt_lpips", -1),
        "render_to_input_psnr": ("mv_render_to_generated_input_psnr", 1),
        "render_to_input_ssim": ("mv_render_to_generated_input_ssim", 1),
        "render_to_input_lpips": ("mv_render_to_generated_input_lpips", -1),
        "render_to_gt_psnr": ("mv_render_to_real_gt_psnr", 1),
        "render_to_gt_ssim": ("mv_render_to_real_gt_ssim", 1),
        "render_to_gt_lpips": ("mv_render_to_real_gt_lpips", -1),
    }
    summary_rows, paired_rows = [], []
    for name, (field, direction) in metrics.items():
        old = np.array([float(base[k][field]) for k in sorted(base)], dtype=float)
        new = np.array([float(adapt[k][field]) for k in sorted(adapt)], dtype=float)
        delta = new-old
        try: p = float(wilcoxon(delta).pvalue)
        except ValueError: p = 1.0
        lo, hi = bootstrap_ci(delta)
        summary_rows.append({
            "metric": name, "higher_is_better": int(direction > 0), "count": len(old),
            "original_mean": float(old.mean()), "adapted_mean": float(new.mean()),
            "adapted_minus_original": float(delta.mean()), "improved_fraction": float(np.mean(direction*delta > 0)),
            "paired_delta_ci95_low": lo, "paired_delta_ci95_high": hi, "wilcoxon_p": p,
        })
        for k, a, b, d in zip(sorted(base), old, new, delta):
            paired_rows.append({"scene": k[0], "slot": k[1], "metric": name, "original": a, "adapted": b, "delta": d})

    geometry_rows = []
    for scene in SCENES:
        real_pt = filtered_xyz(BASE / "reconstruction" / "scenes" / scene / "real8" / "vggt_reconstruction.pt")
        original_pt = filtered_xyz(BASE / "reconstruction" / "scenes" / scene / "mvgen_1real7gen" / "vggt_reconstruction.pt")
        adapted_pt = filtered_xyz(ADAPT_RECON / "scenes" / scene / "mvgen_1real7gen" / "vggt_reconstruction.pt")
        for method, points in (("original", original_pt), ("adapted", adapted_pt)):
            g = rp.compare_pointclouds(real_pt, points, 50000, 23, "real8", method)
            geometry_rows.append({
                "scene": scene, "method": method,
                "normalized_chamfer_l1": g["symmetric_chamfer_l1_normalized"],
                "normalized_chamfer_l2": g["symmetric_chamfer_l2_normalized"],
                "hausdorff_p95_normalized": g["hausdorff_p95_normalized"],
                "fscore_0p005": g["f_scores_at_normalized_threshold"]["0.005"]["f_score"],
                "fscore_0p010": g["f_scores_at_normalized_threshold"]["0.010"]["f_score"],
                "fscore_0p020": g["f_scores_at_normalized_threshold"]["0.020"]["f_score"],
                "point_count": len(points),
            })
        make_sheet(scene, OUT / "comparisons" / f"{scene}_original_vs_adapter.jpg")
        del real_pt, original_pt, adapted_pt

    geo_summary = []
    for metric, direction in (("normalized_chamfer_l1", -1), ("normalized_chamfer_l2", -1),
                              ("hausdorff_p95_normalized", -1), ("fscore_0p005", 1),
                              ("fscore_0p010", 1), ("fscore_0p020", 1), ("point_count", 0)):
        old = np.array([r[metric] for r in geometry_rows if r["method"] == "original"], float)
        new = np.array([r[metric] for r in geometry_rows if r["method"] == "adapted"], float)
        geo_summary.append({"metric": "geometry_"+metric, "higher_is_better": int(direction>0) if direction else "",
                            "count": len(old), "original_mean": old.mean(), "adapted_mean": new.mean(),
                            "adapted_minus_original": (new-old).mean(),
                            "improved_fraction": np.mean(direction*(new-old)>0) if direction else ""})

    timing_rows = []
    for scene in SCENES:
        bgen = json.loads((BASE / "mvgen" / scene / "generation_metadata.json").read_text())
        agen = json.loads((ADAPT_GEN / scene / "generation_metadata.json").read_text())
        brec = json.loads((BASE / "reconstruction" / "scenes" / scene / "mvgen_1real7gen" / "metrics.json").read_text())
        arec = json.loads((ADAPT_RECON / "scenes" / scene / "mvgen_1real7gen" / "metrics.json").read_text())
        timing_rows.append({"scene": scene,
            "original_generation_seconds": bgen["timing_seconds"]["diffusion_generation"],
            "adapted_generation_seconds": agen["generation_seconds"],
            "original_vggt_inference_seconds": brec["timing_seconds"]["model_inference"],
            "adapted_vggt_inference_seconds": arec["timing_seconds"]["model_inference"],
            "original_generation_peak_gib": bgen["peak_allocated_gib"],
            "adapted_generation_peak_gib": agen["peak_allocated_gib"],
            "original_vggt_peak_gib": brec["peak_allocated_gib"],
            "adapted_vggt_peak_gib": arec["peak_allocated_gib"],
        })

    train = json.loads((ROOT / "adapter_4train_2val" / "training_summary.json").read_text())
    payload = {"protocol": {
        "design": "MVGenMaster frozen UNet backbone; train add_conv_in 3D-prior adapter only; frozen VGGT-G downstream",
        "train_scenes": train["train_scenes"], "validation_scenes": train["validation_scenes"], "test_scenes": SCENES,
        "disjoint_split": True, "steps": train["steps"], "lr": train["lr"], "seed": train["seed"],
        "generation": {"views": "1 real + 7 generated", "steps": 50, "cfg": 2.0},
        "render": "native-size intrinsics after exact removal of VGGT 518x518 virtual padding",
    }, "training": train, "image_render_summary": summary_rows,
       "geometry_summary": geo_summary, "geometry_per_scene": geometry_rows, "timing_per_scene": timing_rows}
    (OUT / "joint_adapter_test5_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUT / "ablation_image_render.csv", summary_rows)
    write_csv(OUT / "paired_view_metrics.csv", paired_rows)
    write_csv(OUT / "geometry_per_scene.csv", geometry_rows)
    write_csv(OUT / "ablation_geometry.csv", geo_summary)
    write_csv(OUT / "timing_per_scene.csv", timing_rows)
    print(json.dumps({"image_render": summary_rows, "geometry": geo_summary}, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
