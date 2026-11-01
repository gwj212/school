#!/usr/bin/env python3
"""Native-size rendering and metrics for the held-out 30-scene test.

The saved VGGT reconstruction was inferred from a 518x518 padded canvas.  For
each source image this evaluator removes the exact pad geometry and remaps the
camera intrinsics to the source image's native width and height before
rasterization.  Metrics are never computed against the padded model tensor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

MODEL_CANVAS = 518
PATCH = 14


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def write_csv(path: Path, rows: list[dict]):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def open_rgb(path: str | Path) -> tuple[Image.Image, torch.Tensor]:
    with Image.open(path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return rgb.copy(), tensor


def preprocess_geometry(width: int, height: int) -> dict:
    """Reproduce vggt.utils.load_fn.load_and_preprocess_images(mode='pad')."""
    if width >= height:
        resized_width = MODEL_CANVAS
        resized_height = round(height * (MODEL_CANVAS / width) / PATCH) * PATCH
    else:
        resized_height = MODEL_CANVAS
        resized_width = round(width * (MODEL_CANVAS / height) / PATCH) * PATCH
    resized_width = int(resized_width)
    resized_height = int(resized_height)
    pad_width = MODEL_CANVAS - resized_width
    pad_height = MODEL_CANVAS - resized_height
    pad_left = pad_width // 2
    pad_top = pad_height // 2
    return {
        "native_width": int(width),
        "native_height": int(height),
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": int(pad_left),
        "pad_right": int(pad_width - pad_left),
        "pad_top": int(pad_top),
        "pad_bottom": int(pad_height - pad_top),
    }


def native_intrinsics(full_intrinsics: torch.Tensor, geometry: dict) -> torch.Tensor:
    """Map K from the padded 518 square to the unpadded native image plane."""
    scale_x = geometry["native_width"] / geometry["resized_width"]
    scale_y = geometry["native_height"] / geometry["resized_height"]
    intrinsic = full_intrinsics.clone()
    intrinsic[0, 0] = full_intrinsics[0, 0] * scale_x
    intrinsic[1, 1] = full_intrinsics[1, 1] * scale_y
    intrinsic[0, 2] = (full_intrinsics[0, 2] - geometry["pad_left"]) * scale_x
    intrinsic[1, 2] = (full_intrinsics[1, 2] - geometry["pad_top"]) * scale_y
    return intrinsic


def image_metrics(reference: torch.Tensor, prediction: torch.Tensor, perceptual) -> dict:
    ref = reference.float().clamp(0, 1)
    pred = prediction.float().clamp(0, 1)
    if ref.shape != pred.shape:
        raise RuntimeError(f"metric size mismatch: {tuple(ref.shape)} vs {tuple(pred.shape)}")
    mse = torch.mean((ref - pred) ** 2).item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    ref_np = ref.permute(1, 2, 0).detach().cpu().numpy()
    pred_np = pred.permute(1, 2, 0).detach().cpu().numpy()
    ssim = structural_similarity(ref_np, pred_np, channel_axis=2, data_range=1.0)
    with torch.inference_mode():
        distance = perceptual(ref[None] * 2 - 1, pred[None] * 2 - 1)
    return {
        "psnr": finite(psnr),
        "ssim": finite(ssim),
        "lpips": finite(distance.flatten()[0].item()),
        "mse": finite(mse),
    }


def resize_tensor(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(image.shape[-2:]) == (height, width):
        return image
    return F.interpolate(image[None], size=(height, width), mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(array)


def fitted(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = (width - copy.width) // 2
    y = (height - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def make_sheet(scene: str, rows: list[dict], path: Path):
    cell_width, cell_height, row_label, header = 230, 170, 92, 58
    titles = ("Real GT", "real8 native render", "MVGen input", "MVGen native render", "MV render -> GT size")
    sheet = Image.new("RGB", (row_label + len(titles) * cell_width, header + 8 * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), scene, fill="black")
    for column, title in enumerate(titles):
        draw.text((row_label + column * cell_width + 7, 31), title, fill="black")
    for slot, row in enumerate(rows):
        y = header + slot * cell_height
        draw.multiline_text((5, y + 10), f"view {slot:02d}\n{row['source_type']}", fill="black")
        images = (
            row["real_pil"], row["real_render_pil"], row["mv_pil"],
            row["mv_render_pil"], row["mv_to_gt_render_pil"],
        )
        labels = (
            f"{row['gt_width']}x{row['gt_height']}", f"{row['gt_width']}x{row['gt_height']}",
            f"{row['mv_width']}x{row['mv_height']}", f"{row['mv_width']}x{row['mv_height']}",
            f"{row['gt_width']}x{row['gt_height']}",
        )
        for column, (image, label) in enumerate(zip(images, labels)):
            x = row_label + column * cell_width
            sheet.paste(fitted(image, cell_width, cell_height - 20), (x, y))
            draw.text((x + 7, y + cell_height - 18), label, fill="black")
    sheet.save(path, quality=95)


def gaussian_dict(reconstruction: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value[0].reshape(-1, value.shape[-1]).float().to(device)
        for key, value in reconstruction["gaussians"].items()
        if torch.is_tensor(value) and value.ndim >= 2
    }


def render_native(render_gaussians, gaussians, extrinsic, full_intrinsic, geometry):
    intrinsic = native_intrinsics(full_intrinsic, geometry)
    height = geometry["native_height"]
    width = geometry["native_width"]
    with torch.inference_mode():
        return render_gaussians(
            gaussians["xyz"], gaussians["rotation"], gaussians["scale"],
            gaussians["opacity"], gaussians["color"], extrinsic, intrinsic, height, width,
        )


def aggregate(rows: list[dict], output: Path):
    identifier = {
        "scene", "category", "slot", "source_type", "gt_path", "mv_input_path",
        "gt_width", "gt_height", "mv_width", "mv_height",
        "gt_pad_left", "gt_pad_right", "gt_pad_top", "gt_pad_bottom",
        "mv_pad_left", "mv_pad_right", "mv_pad_top", "mv_pad_bottom",
        "real8_render_size_matches_gt", "mv_render_size_matches_input", "mv_to_gt_render_size_matches_gt",
    }
    summaries = []
    for scope, subset in (
        ("all_views", rows),
        ("generated_views_only", [row for row in rows if row["source_type"] == "generated"]),
    ):
        for key in sorted(set().union(*(row.keys() for row in subset)) - identifier):
            values = [
                float(row[key]) for row in subset
                if isinstance(row.get(key), (int, float)) and row.get(key) is not None and math.isfinite(float(row[key]))
            ]
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            summaries.append({
                "scope": scope, "metric": key, "count": len(array),
                "mean": array.mean(), "min": array.min(), "max": array.max(), "sum": array.sum(),
                "median": np.median(array), "std": array.std(ddof=1) if len(array) > 1 else 0.0,
            })
    write_csv(output / "metric_summary_native.csv", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--mvgen-root", type=Path, required=True)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparisons").mkdir(exist_ok=True)

    sys.path.insert(0, "/root/vggt")
    from vggt.utils.gs_optimizer import render_gaussians
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda:0")
    perceptual = lpips.LPIPS(net="alex").eval().to(device)
    scenes = json.loads((args.manifests / "selection_protocol.json").read_text())["scenes"]
    if args.scene_limit:
        scenes = scenes[: args.scene_limit]

    all_rows, scene_rows, timing_rows = [], [], []
    status_path = args.output / "status.json"
    status = {"completed": [], "failed": {}, "scenes": scenes, "mode": "native_intrinsics_v2"}

    for scene in scenes:
        started = time.perf_counter()
        category = scene.split("_")[0]
        scene_root = args.reconstruction_root / "scenes" / scene
        scene_output = args.output / "scenes" / scene
        scene_output.mkdir(parents=True, exist_ok=True)
        try:
            manifest = json.loads((args.manifests / f"{scene}.json").read_text())
            gt_paths = [view["image"] for view in manifest["views"]]
            real_reconstruction = torch.load(scene_root / "real8" / "vggt_reconstruction.pt", map_location="cpu")
            mv_reconstruction = torch.load(scene_root / "mvgen_1real7gen" / "vggt_reconstruction.pt", map_location="cpu")
            real_metrics_meta = json.loads((scene_root / "real8" / "metrics.json").read_text())
            mv_metrics_meta = json.loads((scene_root / "mvgen_1real7gen" / "metrics.json").read_text())
            real_paths = real_metrics_meta["input_paths"]
            mv_paths = mv_metrics_meta["input_paths"]
            if real_paths != gt_paths:
                raise RuntimeError(f"{scene}: real8 inputs do not match manifest GT paths")

            real_canvas = tuple(real_reconstruction["images"].shape[-2:])
            mv_canvas = tuple(mv_reconstruction["images"].shape[-2:])
            if real_canvas != (MODEL_CANVAS, MODEL_CANVAS) or mv_canvas != (MODEL_CANVAS, MODEL_CANVAS):
                raise RuntimeError(f"{scene}: unexpected model canvases: real={real_canvas}, mv={mv_canvas}")

            real_ext, real_full_intrinsic = pose_encoding_to_extri_intri(
                real_reconstruction["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS)
            )
            mv_ext, mv_full_intrinsic = pose_encoding_to_extri_intri(
                mv_reconstruction["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS)
            )
            real_ext = real_ext[0].to(device)
            real_full_intrinsic = real_full_intrinsic[0].to(device)
            mv_ext = mv_ext[0].to(device)
            mv_full_intrinsic = mv_full_intrinsic[0].to(device)
            real_gaussians = gaussian_dict(real_reconstruction, device)
            mv_gaussians = gaussian_dict(mv_reconstruction, device)

            visual_rows, metric_rows = [], []
            for slot in range(8):
                gt_pil, gt_tensor = open_rgb(gt_paths[slot])
                mv_pil, mv_tensor = open_rgb(mv_paths[slot])
                gt_width, gt_height = gt_pil.size
                mv_width, mv_height = mv_pil.size
                gt_geometry = preprocess_geometry(gt_width, gt_height)
                mv_geometry = preprocess_geometry(mv_width, mv_height)

                real_render = render_native(
                    render_gaussians, real_gaussians, real_ext[slot], real_full_intrinsic[slot], gt_geometry
                )
                mv_render = render_native(
                    render_gaussians, mv_gaussians, mv_ext[slot], mv_full_intrinsic[slot], mv_geometry
                )
                mv_to_gt_render = render_native(
                    render_gaussians, mv_gaussians, mv_ext[slot], mv_full_intrinsic[slot], gt_geometry
                )

                real_render_dir = scene_output / "real8_native"
                mv_render_dir = scene_output / "mvgen_native"
                mv_gt_render_dir = scene_output / "mvgen_to_gt_native"
                for directory in (real_render_dir, mv_render_dir, mv_gt_render_dir):
                    directory.mkdir(exist_ok=True)
                tensor_to_pil(real_render).save(real_render_dir / f"view_{slot:02d}_render.png")
                tensor_to_pil(mv_render).save(mv_render_dir / f"view_{slot:02d}_render.png")
                tensor_to_pil(mv_to_gt_render).save(mv_gt_render_dir / f"view_{slot:02d}_render.png")

                gt_device = gt_tensor.to(device)
                mv_device = mv_tensor.to(device)
                direct_mv_at_gt = resize_tensor(mv_device, gt_height, gt_width)
                metrics = {}
                for prefix, reference, prediction in (
                    ("real8_render_to_real", gt_device, real_render),
                    ("mv_render_to_generated_input", mv_device, mv_render),
                    ("mv_render_to_real_gt", gt_device, mv_to_gt_render),
                    ("mv_generated_to_real_gt", gt_device, direct_mv_at_gt),
                ):
                    values = image_metrics(reference, prediction, perceptual)
                    metrics.update({f"{prefix}_{key}": value for key, value in values.items()})

                row = {
                    "scene": scene, "category": category, "slot": slot,
                    "source_type": "real" if slot == 0 else "generated",
                    "gt_path": str(gt_paths[slot]), "mv_input_path": str(mv_paths[slot]),
                    "gt_width": gt_width, "gt_height": gt_height,
                    "mv_width": mv_width, "mv_height": mv_height,
                    "gt_pad_left": gt_geometry["pad_left"], "gt_pad_right": gt_geometry["pad_right"],
                    "gt_pad_top": gt_geometry["pad_top"], "gt_pad_bottom": gt_geometry["pad_bottom"],
                    "mv_pad_left": mv_geometry["pad_left"], "mv_pad_right": mv_geometry["pad_right"],
                    "mv_pad_top": mv_geometry["pad_top"], "mv_pad_bottom": mv_geometry["pad_bottom"],
                    "real8_render_size_matches_gt": int(tuple(real_render.shape[-2:]) == (gt_height, gt_width)),
                    "mv_render_size_matches_input": int(tuple(mv_render.shape[-2:]) == (mv_height, mv_width)),
                    "mv_to_gt_render_size_matches_gt": int(tuple(mv_to_gt_render.shape[-2:]) == (gt_height, gt_width)),
                    **metrics,
                }
                metric_rows.append(row)
                all_rows.append(row)
                visual_rows.append({
                    **row, "real_pil": gt_pil, "mv_pil": mv_pil,
                    "real_render_pil": tensor_to_pil(real_render),
                    "mv_render_pil": tensor_to_pil(mv_render),
                    "mv_to_gt_render_pil": tensor_to_pil(mv_to_gt_render),
                })
                del real_render, mv_render, mv_to_gt_render, gt_device, mv_device, direct_mv_at_gt
                torch.cuda.empty_cache()

            write_csv(args.output / f"{scene}_metrics_per_view_native.csv", metric_rows)
            make_sheet(scene, visual_rows, args.output / "comparisons" / f"{scene}_native_comparison.jpg")
            numeric_keys = [
                key for key, value in metric_rows[0].items()
                if isinstance(value, (int, float)) and not key.endswith("width") and not key.endswith("height")
                and "pad_" not in key and "size_matches" not in key and key != "slot"
            ]
            scene_row = {"scene": scene, "category": category}
            for key in numeric_keys:
                values = [float(row[key]) for row in metric_rows if row[key] is not None and math.isfinite(float(row[key]))]
                generated = [
                    float(row[key]) for row in metric_rows
                    if row["source_type"] == "generated" and row[key] is not None and math.isfinite(float(row[key]))
                ]
                scene_row[key] = float(np.mean(values)) if values else None
                scene_row[f"generated_only.{key}"] = float(np.mean(generated)) if generated else None
            scene_rows.append(scene_row)
            timing_rows.append({"scene": scene, "native_render_and_metrics_seconds": time.perf_counter() - started})
            status["completed"].append(scene)
            status["failed"].pop(scene, None)
            print(f"COMPLETE {scene} native_eval={timing_rows[-1]['native_render_and_metrics_seconds']:.2f}s", flush=True)
            del real_reconstruction, mv_reconstruction, real_gaussians, mv_gaussians
            torch.cuda.empty_cache()
        except Exception as error:
            import traceback
            status["failed"][scene] = {"error": repr(error), "traceback": traceback.format_exc()}
            print(f"FAILED {scene}: {error!r}", flush=True)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")

    write_csv(args.output / "metrics_per_view_native.csv", all_rows)
    write_csv(args.output / "metrics_per_scene_native.csv", scene_rows)
    write_csv(args.output / "timing_per_scene_native.csv", timing_rows)
    summaries = aggregate(all_rows, args.output)
    protocol = {
        "experiment": "held-out my_test_multiview test30 native-size render evaluation v2",
        "scene_count": len(scenes),
        "model_canvas": [MODEL_CANVAS, MODEL_CANVAS],
        "correction": (
            "For every original real or MVGen input, reproduce VGGT pad-mode resize geometry, remove that virtual padding "
            "from the 518 camera plane, scale fx/fy/cx/cy to the original width and height, and rasterize directly at the native size."
        ),
        "render_sets": {
            "real8_native": "real8 branch rendered directly at each real GT image's original size",
            "mvgen_native": "MVGen branch rendered directly at each actual VGGT input image's original size",
            "mvgen_to_gt_native": "MVGen branch rendered directly at the paired real GT image's original size",
        },
        "metric_pairs": {
            "real8_render_to_real": "native real8 render versus native real test image",
            "mv_render_to_generated_input": "native MVGen-branch render versus the actual native real/generated VGGT input",
            "mv_render_to_real_gt": "MVGen-branch render on the native GT plane versus native real GT",
            "mv_generated_to_real_gt": "MVGen input resized once to the paired native GT dimensions versus native GT",
        },
        "metric_definitions": {
            "PSNR": "RGB [0,1], all native pixels/channels, -10*log10(MSE)",
            "SSIM": "skimage structural_similarity, channel_axis=2, data_range=1.0, native target dimensions",
            "LPIPS": "LPIPS AlexNet, RGB [0,1] mapped to [-1,1], native target dimensions",
            "MSE": "mean squared RGB error on [0,1] at native target dimensions",
        },
        "invalidated_old_protocol": "The earlier 518x518 metrics included model padding and are not used as native-view scores.",
    }
    (args.output / "evaluation_protocol_native.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    status["summary_rows"] = len(summaries)
    status["view_rows"] = len(all_rows)
    status["scene_rows"] = len(scene_rows)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": len(status["completed"]), "failed": status["failed"], "view_rows": len(all_rows)}, ensure_ascii=False, indent=2))
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
