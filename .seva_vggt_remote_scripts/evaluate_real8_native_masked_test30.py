#!/usr/bin/env python3
"""Re-evaluate real8 test30 after removing source-view padding Gaussians."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

MODEL_CANVAS = 518
PATCH = 14
MANIFESTS = Path("/root/seva_vggt/config/test30_mvgen")
RECON_ROOT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/reconstruction/scenes")
OLD_EVAL = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/evaluation_native_v2")
OUTPUT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/evaluation_native_masked_real8")


def preprocess_geometry(width: int, height: int) -> dict:
    if width >= height:
        rw = MODEL_CANVAS
        rh = round(height * (MODEL_CANVAS / width) / PATCH) * PATCH
    else:
        rh = MODEL_CANVAS
        rw = round(width * (MODEL_CANVAS / height) / PATCH) * PATCH
    pw, ph = MODEL_CANVAS - rw, MODEL_CANVAS - rh
    return {
        "native_width": width,
        "native_height": height,
        "resized_width": int(rw),
        "resized_height": int(rh),
        "pad_left": int(pw // 2),
        "pad_right": int(pw - pw // 2),
        "pad_top": int(ph // 2),
        "pad_bottom": int(ph - ph // 2),
    }


def native_intrinsic(k: torch.Tensor, geom: dict) -> torch.Tensor:
    result = k.clone()
    sx = geom["native_width"] / geom["resized_width"]
    sy = geom["native_height"] / geom["resized_height"]
    result[0, 0] = k[0, 0] * sx
    result[1, 1] = k[1, 1] * sy
    result[0, 2] = (k[0, 2] - geom["pad_left"]) * sx
    result[1, 2] = (k[1, 2] - geom["pad_top"]) * sy
    return result


def open_rgb(path: str | Path) -> tuple[Image.Image, torch.Tensor]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        return rgb.copy(), torch.from_numpy(array).permute(2, 0, 1).contiguous()


def to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5
    ).astype(np.uint8)
    return Image.fromarray(array)


def metrics(reference: torch.Tensor, prediction: torch.Tensor, perceptual) -> dict:
    ref = reference.float().clamp(0, 1)
    pred = prediction.float().clamp(0, 1)
    mse = torch.mean((ref - pred) ** 2).item()
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    ref_np = ref.permute(1, 2, 0).detach().cpu().numpy()
    pred_np = pred.permute(1, 2, 0).detach().cpu().numpy()
    ssim = structural_similarity(ref_np, pred_np, channel_axis=2, data_range=1.0)
    with torch.inference_mode():
        distance = perceptual(ref[None] * 2 - 1, pred[None] * 2 - 1)
    return {"psnr": psnr, "ssim": float(ssim), "lpips": float(distance.flatten()[0]), "mse": mse}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fitted(image: Image.Image, width: int, height: int) -> Image.Image:
    result = Image.new("RGB", (width, height), "white")
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    result.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return result


def make_sheet(scene: str, visual_rows: list[dict], path: Path) -> None:
    cell_w, cell_h, label_w, header = 270, 210, 85, 48
    titles = ("GT", "old raw-all render", "masked render")
    sheet = Image.new("RGB", (label_w + 3 * cell_w, header + 8 * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), scene, fill="black")
    for col, title in enumerate(titles):
        draw.text((label_w + col * cell_w + 5, 25), title, fill="black")
    for row in visual_rows:
        slot = row["slot"]
        y = header + slot * cell_h
        draw.text((5, y + 10), f"view {slot:02d}", fill="black")
        for col, key in enumerate(("gt", "old", "masked")):
            sheet.paste(fitted(row[key], cell_w, cell_h - 25), (label_w + col * cell_w, y))
        draw.text(
            (label_w + 2 * cell_w + 5, y + cell_h - 22),
            f"PSNR {row['masked_psnr']:.2f}  SSIM {row['masked_ssim']:.3f}",
            fill="black",
        )
    sheet.save(path, quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "scenes").mkdir(exist_ok=True)
    (OUTPUT / "comparisons").mkdir(exist_ok=True)
    sys.path.insert(0, "/root/vggt")
    from vggt.utils.gs_optimizer import render_gaussians
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda:0")
    perceptual = lpips.LPIPS(net="alex").eval().to(device)
    scenes = json.loads((MANIFESTS / "selection_protocol.json").read_text())["scenes"]
    old = pd.read_csv(OLD_EVAL / "metrics_per_view_native.csv", encoding="utf-8-sig")
    old_map = {(row.scene, int(row.slot)): row for row in old.itertuples(index=False)}
    status = {"completed": [], "failed": {}, "mode": "real8_source_padding_masked"}
    rows, scene_rows = [], []

    for scene in scenes:
        started = time.perf_counter()
        try:
            branch = RECON_ROOT / scene / "real8"
            reconstruction = torch.load(branch / "vggt_reconstruction.pt", map_location="cpu")
            input_paths = json.loads((branch / "metrics.json").read_text())["input_paths"]
            geoms, native = [], []
            valid = torch.zeros((8, MODEL_CANVAS, MODEL_CANVAS), dtype=torch.bool, device=device)
            for view, path in enumerate(input_paths):
                image, tensor = open_rgb(path)
                geom = preprocess_geometry(*image.size)
                geoms.append(geom)
                native.append((image, tensor))
                y0, y1 = geom["pad_top"], MODEL_CANVAS - geom["pad_bottom"]
                x0, x1 = geom["pad_left"], MODEL_CANVAS - geom["pad_right"]
                valid[view, y0:y1, x0:x1] = True

            gaussian_grid = {
                key: value[0].float().reshape(8, MODEL_CANVAS, MODEL_CANVAS, value.shape[-1]).to(device)
                for key, value in reconstruction["gaussians"].items()
                if torch.is_tensor(value) and value.ndim >= 2
            }
            gaussians = {key: value[valid] for key, value in gaussian_grid.items()}
            extrinsic, full_intrinsic = pose_encoding_to_extri_intri(
                reconstruction["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS)
            )
            extrinsic = extrinsic[0].to(device)
            full_intrinsic = full_intrinsic[0].to(device)
            visual_rows, current_rows = [], []
            scene_dir = OUTPUT / "scenes" / scene
            scene_dir.mkdir(exist_ok=True)

            for slot in range(8):
                gt_pil, gt_cpu = native[slot]
                geom = geoms[slot]
                intrinsic = native_intrinsic(full_intrinsic[slot], geom)
                with torch.inference_mode():
                    rendered = render_gaussians(
                        gaussians["xyz"], gaussians["rotation"], gaussians["scale"],
                        gaussians["opacity"], gaussians["color"], extrinsic[slot], intrinsic,
                        geom["native_height"], geom["native_width"],
                    )
                result = metrics(gt_cpu.to(device), rendered, perceptual)
                render_pil = to_pil(rendered)
                render_pil.save(scene_dir / f"view_{slot:02d}_render_masked.png")
                old_row = old_map[(scene, slot)]
                row = {
                    "scene": scene,
                    "category": scene.split("_")[0],
                    "slot": slot,
                    "gt_width": geom["native_width"],
                    "gt_height": geom["native_height"],
                    "raw_gaussian_count": int(valid.numel()),
                    "masked_gaussian_count": int(valid.sum()),
                    "removed_padding_gaussians": int((~valid).sum()),
                    "padding_fraction": float((~valid).float().mean()),
                    "old_psnr": float(old_row.real8_render_to_real_psnr),
                    "old_ssim": float(old_row.real8_render_to_real_ssim),
                    "old_lpips": float(old_row.real8_render_to_real_lpips),
                    "old_mse": float(old_row.real8_render_to_real_mse),
                    **{f"masked_{key}": value for key, value in result.items()},
                }
                row["psnr_gain"] = row["masked_psnr"] - row["old_psnr"]
                row["ssim_gain"] = row["masked_ssim"] - row["old_ssim"]
                row["lpips_reduction"] = row["old_lpips"] - row["masked_lpips"]
                rows.append(row)
                current_rows.append(row)
                old_render = Image.open(
                    OLD_EVAL / "scenes" / scene / "real8_native" / f"view_{slot:02d}_render.png"
                ).convert("RGB")
                visual_rows.append({
                    "slot": slot, "gt": gt_pil, "old": old_render, "masked": render_pil,
                    "masked_psnr": row["masked_psnr"], "masked_ssim": row["masked_ssim"],
                })
                del rendered
                torch.cuda.empty_cache()

            make_sheet(scene, visual_rows, OUTPUT / "comparisons" / f"{scene}_masked_comparison.jpg")
            scene_row = {"scene": scene, "category": scene.split("_")[0], "view_count": 8}
            for key in (
                "old_psnr", "old_ssim", "old_lpips", "old_mse",
                "masked_psnr", "masked_ssim", "masked_lpips", "masked_mse",
                "psnr_gain", "ssim_gain", "lpips_reduction", "padding_fraction",
            ):
                scene_row[f"{key}_mean"] = float(np.mean([row[key] for row in current_rows]))
            scene_row["elapsed_seconds"] = time.perf_counter() - started
            scene_rows.append(scene_row)
            status["completed"].append(scene)
            print(
                f"COMPLETE {scene} old={scene_row['old_psnr_mean']:.3f} "
                f"masked={scene_row['masked_psnr_mean']:.3f} gain={scene_row['psnr_gain_mean']:.3f}",
                flush=True,
            )
            del reconstruction, gaussian_grid, gaussians
            torch.cuda.empty_cache()
        except Exception as error:
            import traceback
            status["failed"][scene] = {"error": repr(error), "traceback": traceback.format_exc()}
            print(f"FAILED {scene}: {error!r}", flush=True)
        (OUTPUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))

    write_csv(OUTPUT / "metrics_per_view.csv", rows)
    write_csv(OUTPUT / "metrics_per_scene.csv", scene_rows)
    summary = {
        "scene_count": len(scene_rows),
        "view_count": len(rows),
        "failed": status["failed"],
        "old": {},
        "masked": {},
        "improvement": {},
    }
    for metric in ("psnr", "ssim", "lpips", "mse"):
        for prefix in ("old", "masked"):
            values = np.asarray([row[f"{prefix}_{metric}"] for row in rows], dtype=float)
            summary[prefix][metric] = {
                "mean": float(values.mean()), "median": float(np.median(values)),
                "std": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max()),
            }
    for key in ("psnr_gain", "ssim_gain", "lpips_reduction"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary["improvement"][key] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "improved_views": int((values > 0).sum()), "total_views": int(len(values)),
        }
    padding = np.asarray([row["padding_fraction_mean"] for row in scene_rows], dtype=float)
    summary["padding_fraction_scene_mean"] = float(padding.mean())
    summary["padding_fraction_scene_min"] = float(padding.min())
    summary["padding_fraction_scene_max"] = float(padding.max())
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    status["summary"] = summary
    (OUTPUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
