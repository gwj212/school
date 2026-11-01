#!/usr/bin/env python3
"""Aggregate aligned image metrics for the Mode-A 1-real + 7-SEVA benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import lpips
import torch

import aggregate_benchmark30 as common
import metric_alignment as image_eval


ROOT = Path("/root/seva_vggt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT / "outputs/benchmark30_mode_a_single"
    )
    parser.add_argument(
        "--manifest-root", type=Path, default=ROOT / "config/benchmark30"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lpips-size", type=int, default=256)
    args = parser.parse_args()

    selection = common.read_json(args.root / "benchmark30_scenes.json")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
    scene_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    missing = []
    for number, scene in enumerate(selection["scenes"], start=1):
        print(f"METRICS {number}/{len(selection['scenes'])} {scene}", flush=True)
        scene_root = args.root / scene
        branch = scene_root / "seva_1real_7gen"
        metadata_path = scene_root / "run_metadata_seva_1real_7gen.json"
        image_path = branch / "image_metrics.json"
        if not metadata_path.is_file() or not image_path.is_file():
            missing.append(scene)
            continue
        manifest = common.read_json(args.manifest_root / f"{scene}.json")
        metadata = common.read_json(metadata_path)
        image = common.read_json(image_path)
        image, rows = common.enrich_image_metrics(
            manifest, metadata, branch, image, model, device, args.lpips_size, "crop"
        )
        for row in rows:
            row.update({"scene": scene, "combination": "1real_7seva_mode_a"})
        view_rows.extend(rows)
        result: dict[str, Any] = {
            "scene": scene,
            "combination": "1real_7seva_mode_a",
        }
        for key in (
            "mean_psnr",
            "mean_ssim",
            "mean_lpips",
            "mean_object_psnr",
            "mean_background_psnr",
        ):
            common.put(result, f"image.{key}", image.get(key))
        for key, value in image.get("view_quality_consistency", {}).items():
            common.put(result, f"consistency.view_quality_{key}", value)
        for key, value in image.get("adjacent_view_transition_consistency", {}).items():
            common.put(result, f"consistency.{key}", value)
        for key, value in metadata.get("timing_seconds", {}).items():
            common.put(result, f"speed.{key}_seconds", value)
        common.put(result, "resource.peak_cuda_memory_mb", metadata.get("peak_cuda_memory_mb"))
        scene_rows.append(result)

    metrics: dict[str, Any] = {}
    keys = sorted({key for row in scene_rows for key in row if key not in {"scene", "combination"}})
    aggregate_rows = []
    for key in keys:
        stats = common.finite_stats([float(row[key]) for row in scene_rows if key in row])
        metrics[key] = stats
        aggregate_rows.append({"combination": "1real_7seva_mode_a", "metric": key, **stats})
    payload = {
        "protocol": {
            "generation": "Mode A: 1 real + 7 target cameras, T=21 with padding, short-side 576, 50 steps, nearest-gt, CFG=4",
            "image_alignment": (
                f"{image_eval.ALIGNMENT_VERSION}: raw reference RGB/mask transformed to the "
                "native 576x768 SEVA canvas before PSNR/SSIM/LPIPS."
            ),
            "lpips": f"AlexNet, aspect-preserving short side {args.lpips_size} after alignment",
            "statistics": "count, mean, min, max, and sum across scenes",
        },
        "scene_count": len(scene_rows),
        "missing": missing,
        "metrics": metrics,
    }
    (args.root / "mode_a_image_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    common.write_csv(args.root / "mode_a_scene_image_metrics.csv", scene_rows)
    common.write_csv(args.root / "mode_a_aggregate_image_metrics.csv", aggregate_rows)
    common.write_csv(args.root / "mode_a_per_view_image_metrics.csv", view_rows)
    print(json.dumps({"scene_count": len(scene_rows), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
