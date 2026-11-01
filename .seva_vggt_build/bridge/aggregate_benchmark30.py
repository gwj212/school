#!/usr/bin/env python3
"""Add LPIPS/transition consistency and aggregate the 30-scene benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch
from PIL import Image


ROOT = Path("/root/seva_vggt")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def image_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def lpips_pairs(
    model: torch.nn.Module,
    pairs: list[tuple[Path, Path]],
    device: torch.device,
    size: int,
) -> list[float]:
    if not pairs:
        return []
    left = torch.stack([image_tensor(pair[0], size) for pair in pairs]).to(device)
    right = torch.stack([image_tensor(pair[1], size) for pair in pairs]).to(device)
    values = []
    with torch.inference_mode():
        for start in range(0, len(pairs), 8):
            result = model(left[start : start + 8], right[start : start + 8])
            values.extend(float(value) for value in result.flatten().detach().cpu())
    return values


def branch_image(branch: Path, index: int) -> Path:
    candidates = sorted((branch / "images").glob(f"view_{index:02d}_*"))
    if not candidates:
        raise FileNotFoundError(f"No input image for view {index} in {branch / 'images'}")
    return candidates[0]


def finite_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"count": 0, "mean": None, "min": None, "max": None, "sum": None}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "sum": float(finite.sum()),
    }


def put(row: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        row[key] = float(value)


def enrich_image_metrics(
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    branch: Path,
    image_metrics: dict[str, Any],
    lpips_model: torch.nn.Module,
    device: torch.device,
    lpips_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    view_by_index = {int(view["index"]): view for view in manifest["views"]}
    metric_by_index = {int(item["view"]): item for item in image_metrics["views"]}
    target_indices = [int(index) for index in metadata["target_indices"]]
    target_pairs = [
        (Path(view_by_index[index]["ground_truth"]), branch / "generated_only" / f"view_{index:02d}.png")
        for index in target_indices
    ]
    target_lpips = lpips_pairs(lpips_model, target_pairs, device, lpips_size)
    per_view_rows = []
    for index, value in zip(target_indices, target_lpips):
        metric_by_index[index]["lpips"] = value
        per_view_rows.append({"view": index, **metric_by_index[index]})

    baseline_indices = [int(index) for index in metadata["baseline_indices"]]
    gt_paths = [Path(view_by_index[index]["ground_truth"]) for index in baseline_indices]
    mixed_paths = [branch_image(branch, index) for index in baseline_indices]
    gt_transition = lpips_pairs(
        lpips_model, list(zip(gt_paths[:-1], gt_paths[1:])), device, lpips_size
    )
    mixed_transition = lpips_pairs(
        lpips_model, list(zip(mixed_paths[:-1], mixed_paths[1:])), device, lpips_size
    )
    transition_error = [abs(left - right) for left, right in zip(gt_transition, mixed_transition)]
    correlation = None
    if len(gt_transition) > 1 and np.std(gt_transition) > 1e-12 and np.std(mixed_transition) > 1e-12:
        correlation = float(np.corrcoef(gt_transition, mixed_transition)[0, 1])

    psnr_values = [float(item["psnr"]) for item in image_metrics["views"]]
    ssim_values = [float(item["ssim"]) for item in image_metrics["views"]]
    image_metrics.update(
        {
            "lpips_network": "alex",
            "lpips_resize": lpips_size,
            "mean_lpips": float(np.mean(target_lpips)),
            "min_lpips": float(np.min(target_lpips)),
            "max_lpips": float(np.max(target_lpips)),
            "view_quality_consistency": {
                "psnr_std": float(np.std(psnr_values)),
                "ssim_std": float(np.std(ssim_values)),
                "lpips_std": float(np.std(target_lpips)),
            },
            "adjacent_view_transition_consistency": {
                "definition": "MAE between adjacent-view LPIPS transitions in mixed inputs and the same all-real camera sequence; lower is better.",
                "real8_transition_lpips_mean": float(np.mean(gt_transition)),
                "mixed_transition_lpips_mean": float(np.mean(mixed_transition)),
                "transition_lpips_mae_vs_real8": float(np.mean(transition_error)),
                "transition_lpips_max_error_vs_real8": float(np.max(transition_error)),
                "transition_lpips_correlation_vs_real8": correlation,
            },
        }
    )
    (branch / "image_metrics.json").write_text(json.dumps(image_metrics, indent=2) + "\n")
    return image_metrics, per_view_rows


def mixed_row(
    scene: str,
    real_count: int,
    metadata: dict[str, Any],
    image: dict[str, Any],
    geometry: dict[str, Any],
    generated_metrics: dict[str, Any],
    real_metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene": scene,
        "combination": f"{real_count}real_{8-real_count}seva",
        "real_count": real_count,
        "generated_count": 8 - real_count,
    }
    for key in ("mean_psnr", "mean_ssim", "mean_lpips", "mean_object_psnr", "mean_background_psnr"):
        put(row, f"image.{key}", image.get(key))
    shared_view = next((item for item in image.get("views", []) if int(item["view"]) == 17), None)
    if shared_view is not None:
        for key in ("psnr", "ssim", "lpips", "object_psnr", "background_psnr"):
            put(row, f"image.shared_view17_{key}", shared_view.get(key))
    quality = image.get("view_quality_consistency", {})
    for key, value in quality.items():
        put(row, f"consistency.view_quality_{key}", value)
    transition = image.get("adjacent_view_transition_consistency", {})
    for key, value in transition.items():
        put(row, f"consistency.{key}", value)

    diagonal = float(geometry.get("reference_bbox_diagonal") or 1.0)
    geometry_values = {
        "accuracy_pred_to_real8_mean": geometry.get("prediction_to_reference_mean"),
        "completeness_real8_to_pred_mean": geometry.get("reference_to_prediction_mean"),
        "chamfer_l1": geometry.get("symmetric_chamfer_l1"),
        "chamfer_l1_normalized": geometry.get("symmetric_chamfer_l1_normalized"),
        "chamfer_l2": geometry.get("symmetric_chamfer_l2"),
        "chamfer_l2_normalized": geometry.get("symmetric_chamfer_l2_normalized"),
        "hausdorff_max": geometry.get("hausdorff_max"),
        "hausdorff_max_normalized": geometry.get("hausdorff_max_normalized"),
        "hausdorff_p95": geometry.get("hausdorff_p95"),
        "hausdorff_p95_normalized": geometry.get("hausdorff_p95_normalized"),
    }
    if geometry.get("prediction_to_reference_mean") is not None:
        geometry_values["accuracy_pred_to_real8_mean_normalized"] = geometry["prediction_to_reference_mean"] / diagonal
    if geometry.get("reference_to_prediction_mean") is not None:
        geometry_values["completeness_real8_to_pred_mean_normalized"] = geometry["reference_to_prediction_mean"] / diagonal
    for key, value in geometry_values.items():
        put(row, f"geometry.{key}", value)
    for threshold, values in geometry.get("f_scores_at_normalized_threshold", {}).items():
        for key, value in values.items():
            put(row, f"geometry.{key}_at_{threshold}", value)

    source_aware = geometry.get("source_aware", {})
    for source_name, source_key in (
        ("generated_source", "generated_slots_vs_real8_same_slots"),
        ("real_source", "real_slots_vs_real8_same_slots"),
        ("shared_view17_source", "shared_view17_vs_real8_same_slot"),
    ):
        source = source_aware.get(source_key, {})
        for key in (
            "symmetric_chamfer_l1_normalized",
            "symmetric_chamfer_l2_normalized",
            "hausdorff_p95_normalized",
        ):
            put(row, f"geometry.{source_name}_{key}", source.get(key))
        for threshold, values in source.get("f_scores_at_normalized_threshold", {}).items():
            for key, value in values.items():
                put(row, f"geometry.{source_name}_{key}_at_{threshold}", value)

    generated_recon = generated_metrics["reconstruction"]
    real_recon = real_metrics["reconstruction"]
    put(row, "geometry.bbox_diagonal_ratio", generated_recon.get("bbox_diagonal") / real_recon.get("bbox_diagonal"))
    put(row, "geometry.opacity_mean_abs_gap", abs(generated_recon["opacity"]["mean"] - real_recon["opacity"]["mean"]))
    put(row, "geometry.opacity_p50_abs_gap", abs(generated_recon["opacity"]["p50"] - real_recon["opacity"]["p50"]))

    for key, value in metadata.get("timing_seconds", {}).items():
        put(row, f"speed.{key}_seconds", value)
    timing = metadata.get("timing_seconds", {})
    seva_load = timing.get("seva_model_load", timing.get("seva_model_load_shared_per_scene"))
    vggt_load = timing.get("vggt_model_load", timing.get("vggt_model_load_shared_per_scene"))
    warm = timing.get("warm_end_to_end")
    cold = timing.get("cold_end_to_end", timing.get("standalone_cold_end_to_end_estimate"))
    amortized = timing.get("seven_ratio_amortized_end_to_end")
    if amortized is None and warm is not None and seva_load is not None and vggt_load is not None:
        amortized = warm + (seva_load + vggt_load) / 7.0
    put(row, "speed.canonical_seva_model_load_seconds", seva_load)
    put(row, "speed.canonical_vggt_model_load_seconds", vggt_load)
    put(row, "speed.canonical_warm_end_to_end_seconds", warm)
    put(row, "speed.canonical_standalone_cold_end_to_end_seconds", cold)
    put(row, "speed.canonical_seven_ratio_amortized_end_to_end_seconds", amortized)
    recon_timing = generated_recon.get("timing_seconds", {})
    for key, value in recon_timing.items():
        put(row, f"speed.mixed_vggt_{key}_seconds", value)
    put(row, "resource.peak_cuda_memory_mb", metadata.get("peak_cuda_memory_mb"))
    return row


def real8_row(scene: str, real_metrics: dict[str, Any]) -> dict[str, Any]:
    recon = real_metrics["reconstruction"]
    row: dict[str, Any] = {"scene": scene, "combination": "8real", "real_count": 8, "generated_count": 0}
    put(row, "speed.real8_vggt_total_seconds", recon.get("elapsed_seconds"))
    for key, value in recon.get("timing_seconds", {}).items():
        put(row, f"speed.real8_vggt_{key}_seconds", value)
    put(row, "geometry.real8_bbox_diagonal", recon.get("bbox_diagonal"))
    put(row, "geometry.real8_opacity_mean", recon.get("opacity", {}).get("mean"))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/benchmark30_fixed8")
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "config/benchmark30")
    parser.add_argument("--lpips-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    selection = read_json(args.root / "benchmark30_scenes.json")
    scenes = selection["scenes"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
    scene_rows: list[dict[str, Any]] = []
    per_view_rows: list[dict[str, Any]] = []
    missing = []

    for scene_index, scene in enumerate(scenes, start=1):
        print(f"METRICS {scene_index}/{len(scenes)} {scene}", flush=True)
        manifest = read_json(args.manifest_root / f"{scene}.json")
        scene_root = args.root / scene
        real_metrics_path = scene_root / "real8/metrics.json"
        if not real_metrics_path.exists():
            missing.append(f"{scene}/real8")
            continue
        real_metrics = read_json(real_metrics_path)
        scene_rows.append(real8_row(scene, real_metrics))
        for real_count in range(1, 8):
            branch_name = f"seva_{real_count}real_{8-real_count}gen"
            branch = scene_root / branch_name
            paths = {
                "metadata": scene_root / f"run_metadata_{branch_name}.json",
                "image": branch / "image_metrics.json",
                "geometry": scene_root / f"comparison_real8_vs_{branch_name}.json",
                "generated": branch / "metrics.json",
            }
            if not all(path.exists() for path in paths.values()):
                missing.append(f"{scene}/{branch_name}")
                continue
            metadata = read_json(paths["metadata"])
            image = read_json(paths["image"])
            geometry = read_json(paths["geometry"])
            generated = read_json(paths["generated"])
            image, view_rows = enrich_image_metrics(
                manifest, metadata, branch, image, lpips_model, device, args.lpips_size
            )
            for view_row in view_rows:
                view_row.update({"scene": scene, "combination": f"{real_count}real_{8-real_count}seva"})
            per_view_rows.extend(view_rows)
            scene_rows.append(mixed_row(scene, real_count, metadata, image, geometry, generated, real_metrics))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        groups[row["combination"]].append(row)
    aggregate: dict[str, Any] = {}
    aggregate_rows = []
    for combination, rows in sorted(groups.items(), key=lambda item: item[0]):
        metric_keys = sorted({key for row in rows for key in row if key not in {"scene", "combination", "real_count", "generated_count"}})
        aggregate[combination] = {"scene_count": len(rows), "metrics": {}}
        for metric in metric_keys:
            stats = finite_stats([float(row[metric]) for row in rows if metric in row])
            aggregate[combination]["metrics"][metric] = stats
            aggregate_rows.append({"combination": combination, "metric": metric, **stats})

    payload = {
        "protocol": {
            "scene_selection": selection,
            "camera_slots": [0, 2, 5, 7, 10, 12, 15, 17],
            "combinations": ["8real"] + [f"{value}real_{8-value}seva" for value in range(1, 8)],
            "lpips": f"AlexNet LPIPS after resizing both images to {args.lpips_size}x{args.lpips_size}",
            "geometry_reference": "VGGT-G real8 reconstruction in the same raw coordinate frame; this is a paired pseudo-reference, not ground-truth surface geometry.",
            "statistics": "count, mean, min, max (peak), and sum across scenes",
        },
        "missing": missing,
        "aggregate": aggregate,
    }
    (args.root / "benchmark30_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(args.root / "benchmark30_scene_metrics.csv", scene_rows)
    write_csv(args.root / "benchmark30_aggregate_metrics.csv", aggregate_rows)
    write_csv(args.root / "benchmark30_per_view_image_metrics.csv", per_view_rows)
    print(json.dumps({"scene_rows": len(scene_rows), "per_view_rows": len(per_view_rows), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
