#!/usr/bin/env python3
"""Compare SEVA, MVGenMaster and fused VGGT depth against paired GT views.

CO3Dv2 does not provide dense sensor ground-truth depth for these images.  Two
clearly labelled protocols are therefore reported:

1. estimator_consistency: same monocular estimator on generated and GT RGB;
2. real8_vggt_teacher: mixed/fused VGGT depth versus VGGT depth from eight real
   GT images, with per-view median scale alignment inside the object mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def depth_tensor(record: dict) -> torch.Tensor:
    value = record["depth"]
    if value.ndim == 5:
        value = value[..., 0]
    if value.ndim == 4:
        value = value[0]
    return value.float()


def padded_mask(path: str, size: int = 518) -> torch.Tensor:
    image = Image.open(path).convert("L")
    w, h = image.size
    if w >= h:
        rw, rh = size, round(h * size / w / 14) * 14
    else:
        rh, rw = size, round(w * size / h / 14) * 14
    image = image.resize((rw, rh), Image.Resampling.NEAREST)
    out = np.zeros((size, size), dtype=bool)
    top, left = (size - rh) // 2, (size - rw) // 2
    out[top : top + rh, left : left + rw] = np.asarray(image) > 127
    return torch.from_numpy(out)


def metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    valid = mask & torch.isfinite(pred) & torch.isfinite(target) & (pred > 1e-5) & (target > 1e-5)
    p, t = pred[valid].double(), target[valid].double()
    if p.numel() < 100:
        return {k: math.nan for k in ("absrel", "rmse_norm", "silog", "bad20", "scale")}
    scale = torch.median(t) / torch.median(p).clamp_min(1e-9)
    p = p * scale
    rel = (p - t).abs() / t.clamp_min(1e-9)
    log_error = torch.log(p.clamp_min(1e-9)) - torch.log(t.clamp_min(1e-9))
    silog = torch.sqrt((log_error.square().mean() - log_error.mean().square()).clamp_min(0))
    return {
        "absrel": float(rel.mean()),
        "rmse_norm": float(torch.sqrt((p - t).square().mean()) / torch.median(t)),
        "silog": float(silog),
        "bad20": float((rel > 0.20).double().mean()),
        "scale": float(scale),
        "pixels": int(p.numel()),
    }


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    result = {
        "comparison_rows": len(rows),
        "views_per_method": len({(r["scene"], r["slot"]) for r in rows}),
        "scenes": len({r["scene"] for r in rows}),
    }
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        result[method] = {
            key: float(np.nanmean([r[key] for r in subset]))
            for key in ("absrel", "rmse_norm", "silog", "bad20")
        }
    if {"mvgen_baseline", "geodiff_token_fusion"} <= set(result):
        result["fusion_delta_vs_mvgen"] = {
            key: result["geodiff_token_fusion"][key] - result["mvgen_baseline"][key]
            for key in ("absrel", "rmse_norm", "silog", "bad20")
        }
        result["fusion_relative_percent_vs_mvgen"] = {
            key: 100.0 * result["fusion_delta_vs_mvgen"][key] / max(abs(result["mvgen_baseline"][key]), 1e-12)
            for key in ("absrel", "rmse_norm", "silog", "bad20")
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/root/seva_vggt/outputs/test30_mvgen_vggt"))
    p.add_argument("--manifests", type=Path, default=Path("/root/seva_vggt/config/test30_mvgen"))
    p.add_argument("--strategy-root", type=Path, default=Path("/root/seva_vggt/outputs/model_strategy_test5"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    # Existing paired estimator-depth comparison on the identical five scenes.
    estimator = {}
    columns = {
        "object_absrel": "mean.gt_vs_seva_depth.object.absrel",
        "object_rmse": "mean.gt_vs_seva_depth.object.rmse",
        "bad20": "mean.gt_vs_seva_depth.object.inconsistent_rate_0.20",
        "boundary_absrel": "mean.gt_vs_seva_depth.boundary.absrel",
    }
    for label, folder in (("seva", "depth_ghost_seva"), ("mvgenmaster", "depth_ghost_mvgen")):
        frame = pd.read_csv(args.strategy_root / folder / "per_scene_metrics.csv")
        estimator[label] = {name: float(frame[column].mean()) for name, column in columns.items()}
        estimator[label]["scenes"] = int(len(frame))
    estimator["mvgen_delta_vs_seva"] = {
        key: estimator["mvgenmaster"][key] - estimator["seva"][key] for key in columns
    }
    estimator["mvgen_relative_percent_vs_seva"] = {
        key: 100.0 * estimator["mvgen_delta_vs_seva"][key] / max(abs(estimator["seva"][key]), 1e-12)
        for key in columns
    }

    fused_root = args.root / "mvgen_vggt_geometry_token_fusion/reconstruction_test4/scenes"
    rows = []
    for scene_dir in sorted(fused_root.iterdir()):
        scene = scene_dir.name
        manifest = json.loads((args.manifests / f"{scene}.json").read_text())
        teacher_path = args.root / "reconstruction/scenes" / scene / "real8/vggt_reconstruction.pt"
        baseline_path = args.root / "reconstruction/scenes" / scene / "mvgen_1real7gen/vggt_reconstruction.pt"
        fused_path = scene_dir / "mvgen_geometry_token_fusion_1real7gen/vggt_reconstruction.pt"
        if not (teacher_path.is_file() and baseline_path.is_file() and fused_path.is_file()):
            continue
        teacher = depth_tensor(torch.load(teacher_path, map_location="cpu"))
        candidates = {
            "mvgen_baseline": depth_tensor(torch.load(baseline_path, map_location="cpu")),
            "geodiff_token_fusion": depth_tensor(torch.load(fused_path, map_location="cpu")),
        }
        for slot in range(1, min(8, teacher.shape[0])):
            mask = padded_mask(manifest["views"][slot]["mask"])
            for method, depth in candidates.items():
                row = {"scene": scene, "category": scene.split("_", 1)[0], "slot": slot, "method": method}
                row.update(metrics(depth[slot], teacher[slot], mask))
                rows.append(row)

    write_csv(args.output / "depth_gap_per_view.csv", rows)
    summary = {
        "ground_truth_warning": "CO3Dv2 subset has no dense sensor GT depth. estimator_consistency and real8_vggt_teacher are proxy protocols, not metric GT depth.",
        "estimator_consistency_test5": estimator,
        "real8_vggt_teacher_test4": summarize(rows),
    }
    (args.output / "depth_gap_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
