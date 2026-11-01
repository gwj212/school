#!/usr/bin/env python3
"""Fair image-level comparison on each MVGenMaster native canvas."""

import argparse
import csv
import json
import math
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

import metric_alignment as ma


TARGETS = [2, 5, 7, 10, 12, 15, 17]


def finite_mean(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", required=True, type=Path)
    p.add_argument("--mv-root", required=True, type=Path)
    p.add_argument("--seva-root", required=True, type=Path)
    p.add_argument("--hybrid-root", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--scenes", nargs="+", required=True)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    rows = []
    with torch.inference_mode():
        for scene in args.scenes:
            manifest = json.load(open(args.manifests / f"{scene}.json", encoding="utf-8"))
            for idx in TARGETS:
                view = manifest["views"][idx]
                mv_path = args.mv_root / scene / "images" / f"view_{idx:02d}_generated.png"
                seva_path = args.seva_root / scene / "seva_1real_7gen" / "images" / f"view_{idx:02d}_generated.png"
                mv = ma.read_rgb(mv_path)
                canvas = (mv.shape[1], mv.shape[0])
                gt, _ = ma.align_path_to_canvas(Path(view["ground_truth"]), canvas, "crop")
                mask, _ = ma.align_array_to_canvas(ma.read_mask(Path(view["mask"])), canvas, "crop", is_mask=True)
                seva, _ = ma.align_array_to_canvas(ma.read_rgb(seva_path), canvas, "stretch", is_mask=False)
                fg = mask > 0.5
                methods = [("SEVA", seva), ("MVGenMaster", mv)]
                if args.hybrid_root is not None:
                    hybrid_path = args.hybrid_root / scene / "images" / f"view_{idx:02d}_generated.png"
                    methods.append(("GeometryHybrid", ma.read_rgb(hybrid_path)))
                for method, pred in methods:
                    gt_t = ma.lpips_tensor_from_array(gt, 256)[None].to(device)
                    pred_t = ma.lpips_tensor_from_array(pred, 256)[None].to(device)
                    rows.append({
                        "scene": scene,
                        "view": idx,
                        "method": method,
                        "psnr": ma.psnr(gt, pred),
                        "ssim": float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0)),
                        "lpips": float(perceptual(gt_t, pred_t).item()),
                        "object_psnr": ma.psnr(gt, pred, fg),
                        "background_psnr": ma.psnr(gt, pred, ~fg),
                        "canvas_width": canvas[0],
                        "canvas_height": canvas[1],
                    })
    summary = {}
    method_names = ["SEVA", "MVGenMaster"] + (["GeometryHybrid"] if args.hybrid_root is not None else [])
    for method in method_names:
        subset = [r for r in rows if r["method"] == method]
        summary[method] = {
            "count": len(subset),
            **{f"mean_{key}": finite_mean([r[key] for r in subset]) for key in ("psnr", "ssim", "lpips", "object_psnr", "background_psnr")},
        }
    paired = {}
    for key in ("psnr", "ssim", "lpips", "object_psnr", "background_psnr"):
        a = [r[key] for r in rows if r["method"] == "SEVA"]
        b = [r[key] for r in rows if r["method"] == "MVGenMaster"]
        paired[f"mvgen_minus_seva_{key}"] = float(np.mean(np.asarray(b) - np.asarray(a)))
        if args.hybrid_root is not None:
            c = [r[key] for r in rows if r["method"] == "GeometryHybrid"]
            paired[f"hybrid_minus_seva_{key}"] = float(np.nanmean(np.asarray(c) - np.asarray(a)))
            paired[f"hybrid_minus_mvgen_{key}"] = float(np.nanmean(np.asarray(c) - np.asarray(b)))
    result = {"protocol": "same 5 scenes, same 1 source + 7 exact target cameras, common MVGenMaster canvas", "summary": summary, "paired_differences": paired, "rows": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output / "image_comparison.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    with open(args.output / "image_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"summary": summary, "paired_differences": paired}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
