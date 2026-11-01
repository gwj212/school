#!/usr/bin/env python3
"""Native-size image/render/geometry evaluation for 1/2/3-condition MVGenMaster."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw


SCENES = [
    "chair_455_64092_126267",
    "chair_581_86200_170718",
    "chair_620_101264_203062",
    "bicycle_608_95757_192662",
    "plant_271_28973_57649",
]

CONDITIONS = {
    "mvgen_1real7gen": [0],
    "mvgen_2real6gen": [0, 4],
    "mvgen_3real5gen": [0, 3, 6],
}

MANIFESTS = Path("/root/seva_vggt/config/test30_mvgen")
OLD_ROOT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
NEW_ROOT = Path("/root/seva_vggt/outputs/mvgen_multicond5")
OUTPUT = NEW_ROOT / "evaluation_native"


def write_csv(path: Path, rows: list[dict]):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def filtered_xyz(reconstruction: dict) -> np.ndarray:
    gaussian = reconstruction["gaussians"]
    xyz = gaussian["xyz"][0].reshape(-1, 3).float().cpu().numpy()
    opacity = gaussian["opacity"][0].reshape(-1).float().cpu().numpy()
    scale = gaussian["scale"][0].reshape(-1, 3).float().cpu().numpy()
    finite = np.isfinite(xyz).all(1) & np.isfinite(opacity) & np.isfinite(scale).all(1)
    pre = finite & (scale > 0).all(1) & (opacity >= 0.10)
    scale_max = scale.max(1)
    limit = float(np.percentile(scale_max[pre], 99.9))
    return xyz[pre & (scale_max <= limit)]


def branch_paths(scene: str, condition: str):
    if condition == "mvgen_1real7gen":
        reconstruction = OLD_ROOT / "reconstruction" / "scenes" / scene / condition
        generation = OLD_ROOT / "mvgen" / scene
    else:
        reconstruction = NEW_ROOT / "reconstruction" / "scenes" / scene / condition
        generation = NEW_ROOT / "generation" / scene / condition
    return reconstruction, generation


def fitted(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    resized = image.copy()
    resized.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def make_sheet(scene: str, condition: str, rows: list[dict], path: Path):
    cell_width, cell_height, row_label, header = 230, 170, 105, 60
    titles = ("Real GT", "VGGT input", "render -> input size", "render -> GT size")
    sheet = Image.new("RGB", (row_label + len(titles) * cell_width, header + 8 * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), f"{scene} | {condition}", fill="black")
    for column, title in enumerate(titles):
        draw.text((row_label + column * cell_width + 7, 33), title, fill="black")
    for slot, row in enumerate(rows):
        y = header + slot * cell_height
        draw.multiline_text((5, y + 10), f"view {slot:02d}\n{row['source_type']}", fill="black")
        images = (row["gt_pil"], row["input_pil"], row["render_input_pil"], row["render_gt_pil"])
        for column, image in enumerate(images):
            x = row_label + column * cell_width
            sheet.paste(fitted(image, cell_width, cell_height - 20), (x, y))
    sheet.save(path, quality=95)


def mean(values):
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "comparisons").mkdir(exist_ok=True)
    sys.path[:0] = ["/root/seva_vggt/candidates", "/root/seva_vggt/bridge", "/root/vggt"]
    import evaluate_render_test30_native as native
    import run_pipeline as rp
    from vggt.utils.gs_optimizer import render_gaussians
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda:0")
    perceptual = lpips.LPIPS(net="alex").eval().to(device)
    all_rows, geometry_rows, timing_rows = [], [], []
    status = {"completed": [], "failed": {}, "scenes": SCENES, "conditions": CONDITIONS}
    status_path = OUTPUT / "status.json"

    for scene in SCENES:
        manifest = json.loads((MANIFESTS / f"{scene}.json").read_text())
        gt_paths = [view["image"] for view in manifest["views"]]
        real_branch = OLD_ROOT / "reconstruction" / "scenes" / scene / "real8"
        real_reconstruction = torch.load(real_branch / "vggt_reconstruction.pt", map_location="cpu")
        real_xyz = filtered_xyz(real_reconstruction)
        for condition, condition_slots in CONDITIONS.items():
            task = f"{scene}/{condition}"
            started = time.perf_counter()
            try:
                reconstruction_branch, generation_branch = branch_paths(scene, condition)
                reconstruction = torch.load(reconstruction_branch / "vggt_reconstruction.pt", map_location="cpu")
                reconstruction_meta = json.loads((reconstruction_branch / "metrics.json").read_text())
                generation_meta = json.loads((generation_branch / "generation_metadata.json").read_text())
                input_paths = reconstruction_meta["input_paths"]
                if len(input_paths) != 8:
                    raise RuntimeError(f"expected 8 VGGT inputs, got {len(input_paths)}")

                canvas = tuple(reconstruction["images"].shape[-2:])
                if canvas != (native.MODEL_CANVAS, native.MODEL_CANVAS):
                    raise RuntimeError(f"unexpected canvas {canvas}")
                extrinsic, full_intrinsic = pose_encoding_to_extri_intri(
                    reconstruction["pose_enc"].float(), (native.MODEL_CANVAS, native.MODEL_CANVAS)
                )
                extrinsic = extrinsic[0].to(device)
                full_intrinsic = full_intrinsic[0].to(device)
                gaussians = native.gaussian_dict(reconstruction, device)

                visual_rows = []
                scene_rows = []
                render_root = OUTPUT / "renders" / scene / condition
                render_root.mkdir(parents=True, exist_ok=True)
                for slot in range(8):
                    gt_pil, gt_tensor = native.open_rgb(gt_paths[slot])
                    input_pil, input_tensor = native.open_rgb(input_paths[slot])
                    gt_width, gt_height = gt_pil.size
                    input_width, input_height = input_pil.size
                    gt_geometry = native.preprocess_geometry(gt_width, gt_height)
                    input_geometry = native.preprocess_geometry(input_width, input_height)
                    render_input = native.render_native(
                        render_gaussians, gaussians, extrinsic[slot], full_intrinsic[slot], input_geometry
                    )
                    render_gt = native.render_native(
                        render_gaussians, gaussians, extrinsic[slot], full_intrinsic[slot], gt_geometry
                    )
                    native.tensor_to_pil(render_input).save(render_root / f"view_{slot:02d}_render_input.png")
                    native.tensor_to_pil(render_gt).save(render_root / f"view_{slot:02d}_render_gt.png")

                    gt_device = gt_tensor.to(device)
                    input_device = input_tensor.to(device)
                    input_to_gt = native.resize_tensor(input_device, gt_height, gt_width)
                    metrics = {}
                    for prefix, reference, prediction in (
                        ("input_to_gt", gt_device, input_to_gt),
                        ("render_to_input", input_device, render_input),
                        ("render_to_gt", gt_device, render_gt),
                    ):
                        values = native.image_metrics(reference, prediction, perceptual)
                        metrics.update({f"{prefix}_{key}": value for key, value in values.items()})
                    row = {
                        "scene": scene,
                        "category": scene.split("_")[0],
                        "condition": condition,
                        "real_count": len(condition_slots),
                        "generated_count": 8 - len(condition_slots),
                        "slot": slot,
                        "source_type": "real" if slot in condition_slots else "generated",
                        "gt_path": gt_paths[slot],
                        "input_path": input_paths[slot],
                        "gt_width": gt_width,
                        "gt_height": gt_height,
                        "input_width": input_width,
                        "input_height": input_height,
                        "render_input_exact_size": int(tuple(render_input.shape[-2:]) == (input_height, input_width)),
                        "render_gt_exact_size": int(tuple(render_gt.shape[-2:]) == (gt_height, gt_width)),
                        **metrics,
                    }
                    all_rows.append(row)
                    scene_rows.append(row)
                    visual_rows.append({
                        **row,
                        "gt_pil": gt_pil,
                        "input_pil": input_pil,
                        "render_input_pil": native.tensor_to_pil(render_input),
                        "render_gt_pil": native.tensor_to_pil(render_gt),
                    })
                    del gt_device, input_device, input_to_gt, render_input, render_gt
                    torch.cuda.empty_cache()

                make_sheet(
                    scene,
                    condition,
                    visual_rows,
                    OUTPUT / "comparisons" / f"{scene}_{condition}.jpg",
                )
                prediction_xyz = filtered_xyz(reconstruction)
                geometry = rp.compare_pointclouds(
                    real_xyz,
                    prediction_xyz,
                    sample_count=50000,
                    seed=23,
                    reference_name="real8",
                    prediction_name=condition,
                )
                geometry_row = {
                    "scene": scene,
                    "condition": condition,
                    "real_count": len(condition_slots),
                    "generated_count": 8 - len(condition_slots),
                    "normalized_chamfer_l1": geometry["symmetric_chamfer_l1_normalized"],
                    "normalized_chamfer_l2": geometry["symmetric_chamfer_l2_normalized"],
                    "hausdorff_p95_normalized": geometry["hausdorff_p95_normalized"],
                    "fscore_0p005": geometry["f_scores_at_normalized_threshold"]["0.005"]["f_score"],
                    "fscore_0p010": geometry["f_scores_at_normalized_threshold"]["0.010"]["f_score"],
                    "fscore_0p020": geometry["f_scores_at_normalized_threshold"]["0.020"]["f_score"],
                    "reference_points": len(real_xyz),
                    "prediction_points": len(prediction_xyz),
                }
                geometry_rows.append(geometry_row)
                timing_rows.append({
                    "scene": scene,
                    "condition": condition,
                    "real_count": len(condition_slots),
                    "generation_seconds": generation_meta["timing_seconds"]["diffusion_generation"],
                    "generation_warm_total_seconds": generation_meta["timing_seconds"]["warm_scene_total"]
                    if "warm_scene_total" in generation_meta["timing_seconds"] else generation_meta["timing_seconds"]["warm_branch_total"],
                    "vggt_inference_seconds": reconstruction_meta["timing_seconds"]["model_inference"],
                    "vggt_warm_total_seconds": reconstruction_meta["timing_seconds"]["warm_branch_total"],
                    "evaluation_seconds": time.perf_counter() - started,
                })
                status["completed"].append(task)
                status["failed"].pop(task, None)
                print(f"COMPLETE {task}", flush=True)
                del reconstruction, gaussians, prediction_xyz
                torch.cuda.empty_cache()
            except Exception as error:
                status["failed"][task] = {"error": repr(error), "traceback": traceback.format_exc()}
                print(f"FAILED {task}: {error!r}", flush=True)
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        del real_reconstruction, real_xyz

    summary_rows = []
    for condition, condition_slots in CONDITIONS.items():
        generated = [row for row in all_rows if row["condition"] == condition and row["source_type"] == "generated"]
        all_condition = [row for row in all_rows if row["condition"] == condition]
        geometry_condition = [row for row in geometry_rows if row["condition"] == condition]
        timing_condition = [row for row in timing_rows if row["condition"] == condition]
        row = {
            "condition": condition,
            "real_count": len(condition_slots),
            "generated_count": 8 - len(condition_slots),
            "scene_count": len(SCENES),
            "generated_view_count": len(generated),
        }
        for key in (
            "input_to_gt_psnr", "input_to_gt_ssim", "input_to_gt_lpips",
            "render_to_input_psnr", "render_to_input_ssim", "render_to_input_lpips",
            "render_to_gt_psnr", "render_to_gt_ssim", "render_to_gt_lpips",
        ):
            row[f"generated_only_{key}"] = mean(item[key] for item in generated)
            row[f"all_views_{key}"] = mean(item[key] for item in all_condition)
        for key in (
            "normalized_chamfer_l1", "normalized_chamfer_l2", "hausdorff_p95_normalized",
            "fscore_0p005", "fscore_0p010", "fscore_0p020", "prediction_points",
        ):
            row[f"geometry_{key}"] = mean(item[key] for item in geometry_condition)
        for key in (
            "generation_seconds", "generation_warm_total_seconds", "vggt_inference_seconds",
            "vggt_warm_total_seconds", "evaluation_seconds",
        ):
            row[f"timing_{key}"] = mean(item[key] for item in timing_condition)
        summary_rows.append(row)

    write_csv(OUTPUT / "metrics_per_view.csv", all_rows)
    write_csv(OUTPUT / "geometry_per_scene.csv", geometry_rows)
    write_csv(OUTPUT / "timing_per_scene.csv", timing_rows)
    write_csv(OUTPUT / "condition_summary.csv", summary_rows)
    protocol = {
        "experiment": "MVGenMaster multi-condition short verification on held-out my_test_multiview",
        "scenes": SCENES,
        "conditions": CONDITIONS,
        "fixed": {"total_vggt_inputs": 8, "seed": 23, "steps": 50, "cfg": 2.0},
        "condition_selection": "spread fixed slots: [0], [0,4], [0,3,6]",
        "native_rendering": "remove VGGT 518-square virtual padding and rasterize at each paired input/GT native size",
        "metric_pairs": {
            "input_to_gt": "real or generated VGGT input resized once to paired native GT size versus GT",
            "render_to_input": "native VGGT-G render versus the exact real/generated branch input",
            "render_to_gt": "VGGT-G render directly rasterized on native GT plane versus GT",
        },
        "geometry": "opacity>=0.10, finite positive scale, remove upper 0.1% scale outliers, no point cap; compare raw VGGT coordinates because every branch starts at slot0",
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    status["view_rows"] = len(all_rows)
    status["geometry_rows"] = len(geometry_rows)
    status["summary_rows"] = len(summary_rows)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary_rows, "failed": status["failed"]}, ensure_ascii=False, indent=2))
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
