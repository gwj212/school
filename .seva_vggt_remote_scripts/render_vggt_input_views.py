#!/usr/bin/env python3
"""Render saved VGGT Gaussian parameters at their eight predicted input cameras.

The evaluation is resume-safe and does not rerun SEVA or VGGT.  For every
mixed real/SEVA branch it compares the rendered RGB against both the exact
preprocessed mixed input and the paired all-real input at the same camera slot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


ROOT = Path("/root/seva_vggt")
OUTPUT_ROOT = ROOT / "outputs/cfg_sweep30_t8_s12"
CFG_VALUES = [2, 3, 4, 5, 8]
REAL_COUNTS = list(range(1, 8))
PROTOCOL_VERSION = "vggt_merged_input_render_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cfg", type=int, nargs="*", default=CFG_VALUES)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def image_metrics(reference: torch.Tensor, prediction: torch.Tensor, lpips_model: torch.nn.Module) -> dict[str, float | None]:
    reference = reference.float().clamp(0, 1)
    prediction = prediction.float().clamp(0, 1)
    mse = torch.mean((reference - prediction) ** 2).item()
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    ref_np = reference.permute(1, 2, 0).detach().cpu().numpy()
    pred_np = prediction.permute(1, 2, 0).detach().cpu().numpy()
    ssim = structural_similarity(ref_np, pred_np, channel_axis=2, data_range=1.0)
    with torch.inference_mode():
        lpips_value = lpips_model(reference[None] * 2 - 1, prediction[None] * 2 - 1)
    return {
        "psnr": finite(psnr),
        "ssim": finite(float(ssim)),
        "lpips": finite(float(lpips_value.flatten()[0].detach().cpu())),
    }


def mean_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    for metric in ("psnr", "ssim", "lpips"):
        values = [float(row[f"{prefix}.{metric}"]) for row in rows if row.get(f"{prefix}.{metric}") is not None]
        result[metric] = float(np.mean(values)) if values else None
    return result


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_contact_sheet(rows: list[dict[str, Any]], real_gt: torch.Tensor, mixed: torch.Tensor, rendered: list[torch.Tensor], path: Path) -> None:
    thumb = 224
    header = 34
    label_width = 150
    sheet = Image.new("RGB", (label_width + 3 * thumb, header + len(rows) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((label_width + 70, 10), "Real GT", fill="black")
    draw.text((label_width + thumb + 55, 10), "VGGT input", fill="black")
    draw.text((label_width + 2 * thumb + 55, 10), "VGGT render", fill="black")
    for slot, row in enumerate(rows):
        y = header + slot * thumb
        label = f"view {row['view_index']:02d}\n{row['source_type']}"
        draw.multiline_text((6, y + 8), label, fill="black", spacing=4)
        for col, tensor in enumerate((real_gt[slot], mixed[slot], rendered[slot])):
            image = tensor_to_pil(tensor).resize((thumb, thumb), Image.Resampling.LANCZOS)
            sheet.paste(image, (label_width + col * thumb, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def render_branch(
    branch: Path,
    real8_path: Path,
    metadata_path: Path,
    device: torch.device,
    lpips_model: torch.nn.Module,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = branch / "vggt_render_eval"
    metrics_path = output_dir / "metrics.json"
    if not overwrite and metrics_path.is_file():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            renders = sorted((output_dir / "renders").glob("view_*.png"))
            if previous.get("protocol_version") == PROTOCOL_VERSION and len(previous.get("views", [])) == 8 and len(renders) == 8:
                return previous
        except (OSError, json.JSONDecodeError):
            pass

    started = time.perf_counter()
    reconstruction = torch.load(branch / "vggt_reconstruction.pt", map_location="cpu")
    real8 = torch.load(real8_path, map_location="cpu")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mixed_images = reconstruction["images"][0].float().to(device)
    real_images = real8["images"][0].float().to(device)
    if mixed_images.shape != real_images.shape or mixed_images.shape[0] != 8:
        raise RuntimeError(f"Unexpected input shapes: {mixed_images.shape} vs {real_images.shape}")
    height, width = mixed_images.shape[-2:]

    from vggt.utils.gs_optimizer import render_gaussians
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        reconstruction["pose_enc"].float(), image_size_hw=(height, width)
    )
    extrinsics = extrinsics[0].to(device)
    intrinsics = intrinsics[0].to(device)
    gaussians = {
        key: value[0].reshape(-1, value.shape[-1]).float().to(device)
        for key, value in reconstruction["gaussians"].items()
        if torch.is_tensor(value) and value.ndim >= 2
    }
    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    baseline_indices = [int(value) for value in metadata["baseline_indices"]]
    real_indices = {int(value) for value in metadata["input_indices"]}
    rendered_images: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    render_seconds = 0.0
    metric_seconds = 0.0
    for slot, view_index in enumerate(baseline_indices):
        render_started = time.perf_counter()
        with torch.inference_mode():
            rendered = render_gaussians(
                gaussians["xyz"], gaussians["rotation"], gaussians["scale"],
                gaussians["opacity"], gaussians["color"], extrinsics[slot],
                intrinsics[slot], height, width,
            )
        torch.cuda.synchronize(device)
        render_seconds += time.perf_counter() - render_started
        tensor_to_pil(rendered).save(render_dir / f"view_{view_index:02d}.png")
        rendered_images.append(rendered.detach().cpu())
        metric_started = time.perf_counter()
        render_to_input = image_metrics(mixed_images[slot], rendered, lpips_model)
        render_to_real = image_metrics(real_images[slot], rendered, lpips_model)
        input_to_real = image_metrics(real_images[slot], mixed_images[slot], lpips_model)
        metric_seconds += time.perf_counter() - metric_started
        row: dict[str, Any] = {
            "slot": slot,
            "view_index": view_index,
            "source_type": "real" if view_index in real_indices else "seva",
        }
        for prefix, values in (
            ("render_to_input", render_to_input),
            ("render_to_real_gt", render_to_real),
            ("input_to_real_gt", input_to_real),
        ):
            for metric, value in values.items():
                row[f"{prefix}.{metric}"] = value
        rows.append(row)
        del rendered
        torch.cuda.empty_cache()

    real_rows = [row for row in rows if row["source_type"] == "real"]
    seva_rows = [row for row in rows if row["source_type"] == "seva"]
    summary = {
        "all_views_render_to_input": mean_metrics(rows, "render_to_input"),
        "real_slots_render_to_real": mean_metrics(real_rows, "render_to_real_gt"),
        "seva_slots_render_to_seva": mean_metrics(seva_rows, "render_to_input"),
        "seva_slots_render_to_real_gt": mean_metrics(seva_rows, "render_to_real_gt"),
        "seva_slots_seva_to_real_gt": mean_metrics(seva_rows, "input_to_real_gt"),
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "branch": branch.name,
        "scene": branch.parent.name,
        "cfg": metadata.get("seva_cfg"),
        "real_count": len(real_rows),
        "generated_count": len(seva_rows),
        "render_mode": "merged_all_eight_view_gaussians",
        "camera_source": "VGGT predicted pose_enc converted to world-to-camera extrinsics and pixel intrinsics",
        "reference_preprocessing": "Exact tensors saved by the original VGGT run; no new image alignment",
        "image_size_hw": [height, width],
        "gaussian_count": int(gaussians["xyz"].shape[0]),
        "summary": summary,
        "timing_seconds": {
            "render_total": render_seconds,
            "render_per_view_mean": render_seconds / 8.0,
            "metric_total": metric_seconds,
            "branch_total": time.perf_counter() - started,
        },
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "views": rows,
    }
    save_contact_sheet(rows, real_images.cpu(), mixed_images.cpu(), rendered_images, output_dir / "comparison_sheet.jpg")
    atomic_json(metrics_path, result)
    del reconstruction, real8, mixed_images, real_images, gaussians, extrinsics, intrinsics
    torch.cuda.empty_cache()
    return result


def stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if not len(array):
        return {k: None for k in ("count", "mean", "min", "max", "sum", "median", "std", "ci95_low", "ci95_high")}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "count": int(len(array)), "mean": mean, "min": float(array.min()),
        "max": float(array.max()), "sum": float(array.sum()),
        "median": float(np.median(array)), "std": std,
        "ci95_low": mean - half, "ci95_high": mean + half,
    }


def flatten_scene_row(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cfg": result["cfg"], "scene": result["scene"], "combination": result["branch"],
        "real_count": result["real_count"], "generated_count": result["generated_count"],
        "gaussian_count": result["gaussian_count"],
        "render_total_seconds": result["timing_seconds"]["render_total"],
        "render_per_view_seconds": result["timing_seconds"]["render_per_view_mean"],
        "render_peak_cuda_memory_mb": result["peak_cuda_memory_mb"],
    }
    for group, values in result["summary"].items():
        row[f"{group}.count"] = values["count"]
        for metric in ("psnr", "ssim", "lpips"):
            row[f"{group}.{metric}"] = values[metric]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root: Path, results: list[dict[str, Any]]) -> None:
    out = root / "vggt_render_aggregate"
    scene_rows = [flatten_scene_row(result) for result in results]
    view_rows: list[dict[str, Any]] = []
    for result in results:
        for view in result["views"]:
            view_rows.append({
                "cfg": result["cfg"], "scene": result["scene"], "combination": result["branch"],
                "real_count": result["real_count"], "generated_count": result["generated_count"],
                **view,
            })
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        grouped[(int(row["cfg"]), int(row["real_count"]))].append(row)
    identity = {"cfg", "scene", "combination", "real_count", "generated_count"}
    metric_keys = sorted(set(scene_rows[0]) - identity) if scene_rows else []
    for (cfg, real_count), group in sorted(grouped.items()):
        for metric in metric_keys:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            summary_rows.append({
                "cfg": cfg, "real_count": real_count, "generated_count": 8 - real_count,
                "combination": f"{real_count}real_{8-real_count}gen", "metric": metric,
                **stats(values),
            })
    write_csv(out / "vggt_render_scene_metrics.csv", scene_rows)
    write_csv(out / "vggt_render_per_view_metrics.csv", view_rows)
    write_csv(out / "vggt_render_cfg_ratio_summary.csv", summary_rows)
    atomic_json(out / "vggt_render_summary.json", {
        "protocol_version": PROTOCOL_VERSION,
        "scene_rows": len(scene_rows), "per_view_rows": len(view_rows),
        "summary_rows": len(summary_rows), "expected_scene_rows": 30 * 5 * 7,
        "missing_scene_rows": 30 * 5 * 7 - len(scene_rows),
    })


def main() -> None:
    args = parse_args()
    sys.path.insert(0, "/root/vggt")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    selection = json.loads((args.root / "benchmark30_scenes.json").read_text(encoding="utf-8"))
    scenes = args.scene or list(selection["scenes"])
    jobs = [(cfg, scene, real_count) for cfg in args.cfg for scene in scenes for real_count in REAL_COUNTS]
    if args.limit > 0:
        jobs = jobs[: args.limit]
    status_path = args.root / "vggt_render_status.json"
    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    started = time.time()
    for job_index, (cfg, scene, real_count) in enumerate(jobs, start=1):
        branch_name = f"seva_{real_count}real_{8-real_count}gen"
        scene_root = args.root / f"cfg_{cfg}" / scene
        branch = scene_root / branch_name
        key = f"cfg_{cfg}/{scene}/{branch_name}"
        try:
            result = render_branch(
                branch, scene_root / "real8/vggt_reconstruction.pt",
                scene_root / f"run_metadata_{branch_name}.json", device,
                lpips_model, args.overwrite,
            )
            results.append(result)
            completed.append(key)
            print(f"[{job_index}/{len(jobs)}] COMPLETE {key}", flush=True)
        except Exception as error:
            failed.append({"job": key, "error": repr(error)})
            print(f"[{job_index}/{len(jobs)}] FAILED {key}: {error!r}", flush=True)
        atomic_json(status_path, {
            "protocol_version": PROTOCOL_VERSION, "started_at_unix": started,
            "updated_at_unix": time.time(), "job_count": len(jobs),
            "completed_count": len(completed), "failed_count": len(failed),
            "completed": completed, "failed": failed,
        })
    aggregate(args.root, results)
    atomic_json(status_path, {
        "protocol_version": PROTOCOL_VERSION, "started_at_unix": started,
        "finished_at_unix": time.time(), "job_count": len(jobs),
        "completed_count": len(completed), "failed_count": len(failed),
        "completed": completed, "failed": failed,
    })
    if failed:
        raise SystemExit(f"{len(failed)} render jobs failed")


if __name__ == "__main__":
    main()
