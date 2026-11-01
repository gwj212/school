#!/usr/bin/env python3
"""Diagnose cross-view Gaussian occlusion on a held-out real8 scene."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

MODEL_CANVAS = 518
PATCH = 14
SCENE = "laptop_373_41558_82973"
RECON = Path(
    "/root/seva_vggt/outputs/test30_mvgen_vggt/reconstruction/scenes/"
    f"{SCENE}/real8/vggt_reconstruction.pt"
)
META = RECON.parent / "metrics.json"
OUTPUT = Path(
    "/root/seva_vggt/outputs/test30_mvgen_vggt/"
    f"gaussian_occlusion_diagnostic/{SCENE}"
)


def geometry(width: int, height: int) -> dict:
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


def open_tensor(path: str) -> tuple[Image.Image, torch.Tensor]:
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
    luminance = pred.mean(dim=0)
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": float(ssim),
        "lpips": float(distance.flatten()[0]),
        "near_white_fraction": float((luminance > 0.98).float().mean()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", default="5", help="Comma-separated slots, or 'all'")
    parser.add_argument(
        "--variants",
        default=(
            "raw_all,unpad_all,unpad_scale_p99,unpad_opacity01_scale_p99,"
            "source_only_unpad,source_plus_others_01"
        ),
    )
    args = parser.parse_args()
    slots = list(range(8)) if args.slots == "all" else [int(x) for x in args.slots.split(",")]
    variants = args.variants.split(",")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/root/vggt")
    from vggt.utils.gs_optimizer import render_gaussians
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda:0")
    perceptual = lpips.LPIPS(net="alex").eval().to(device)
    reconstruction = torch.load(RECON, map_location="cpu")
    input_paths = json.loads(META.read_text())["input_paths"]
    tensors = {
        key: value[0].float().reshape(8, MODEL_CANVAS, MODEL_CANVAS, value.shape[-1]).to(device)
        for key, value in reconstruction["gaussians"].items()
        if torch.is_tensor(value) and value.ndim >= 2
    }
    shapes = {key: list(value.shape) for key, value in tensors.items()}
    if tensors["xyz"].shape[:3] != (8, MODEL_CANVAS, MODEL_CANVAS):
        raise RuntimeError(f"Unexpected Gaussian layout: {shapes}")

    extrinsic, full_intrinsic = pose_encoding_to_extri_intri(
        reconstruction["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS)
    )
    extrinsic = extrinsic[0].to(device)
    full_intrinsic = full_intrinsic[0].to(device)

    valid = torch.zeros((8, MODEL_CANVAS, MODEL_CANVAS), dtype=torch.bool, device=device)
    geoms = []
    native = []
    for view, path in enumerate(input_paths):
        image, tensor = open_tensor(path)
        width, height = image.size
        geom = geometry(width, height)
        geoms.append(geom)
        native.append((image, tensor))
        y0, y1 = geom["pad_top"], MODEL_CANVAS - geom["pad_bottom"]
        x0, x1 = geom["pad_left"], MODEL_CANVAS - geom["pad_right"]
        valid[view, y0:y1, x0:x1] = True

    source_ids = torch.arange(8, device=device)[:, None, None].expand_as(valid)
    flat_scale = tensors["scale"].amax(dim=-1)
    scale_p99 = torch.quantile(flat_scale[valid], 0.99)
    opacity = tensors["opacity"].squeeze(-1)
    raw_mask = torch.ones_like(valid)

    def make_variant(name: str, target: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if name == "raw_all":
            mask = raw_mask
        elif name == "unpad_all":
            mask = valid
        elif name == "unpad_scale_p99":
            mask = valid & (flat_scale <= scale_p99)
        elif name == "unpad_opacity01_scale_p99":
            mask = valid & (flat_scale <= scale_p99) & (opacity >= 0.10)
        elif name == "source_only_unpad":
            mask = valid & (source_ids == target)
        elif name == "others_only_unpad":
            mask = valid & (source_ids != target)
        elif name == "source_plus_others_01":
            mask = valid & (flat_scale <= scale_p99)
        elif name == "source_plus_others_025":
            mask = valid & (flat_scale <= scale_p99)
        else:
            raise ValueError(name)
        gaussians = {key: value[mask] for key, value in tensors.items()}
        if name.startswith("source_plus_others_"):
            factor = 0.10 if name.endswith("01") else 0.25
            selected_sources = source_ids[mask]
            weights = torch.where(
                selected_sources == target,
                torch.ones_like(selected_sources, dtype=gaussians["opacity"].dtype),
                torch.full_like(selected_sources, factor, dtype=gaussians["opacity"].dtype),
            )
            gaussians["opacity"] = gaussians["opacity"] * weights[:, None]
        return gaussians, mask

    rows = []
    rendered_by_slot: dict[int, dict[str, Image.Image]] = {}
    for slot in slots:
        gt_pil, gt_cpu = native[slot]
        gt = gt_cpu.to(device)
        geom = geoms[slot]
        intrinsic = native_intrinsic(full_intrinsic[slot], geom)
        rendered_by_slot[slot] = {"GT": gt_pil}
        slot_dir = OUTPUT / f"view_{slot:02d}"
        slot_dir.mkdir(exist_ok=True)
        gt_pil.save(slot_dir / "GT.png")
        for variant in variants:
            gaussian, mask = make_variant(variant, slot)
            with torch.inference_mode():
                rendered = render_gaussians(
                    gaussian["xyz"], gaussian["rotation"], gaussian["scale"],
                    gaussian["opacity"], gaussian["color"], extrinsic[slot], intrinsic,
                    geom["native_height"], geom["native_width"],
                )
            image = to_pil(rendered)
            image.save(slot_dir / f"{variant}.png")
            rendered_by_slot[slot][variant] = image
            row = {
                "scene": SCENE,
                "target_slot": slot,
                "variant": variant,
                "gaussian_count": int(mask.sum()),
                "source_gaussian_count": int((mask & (source_ids == slot)).sum()),
                "other_gaussian_count": int((mask & (source_ids != slot)).sum()),
                **metrics(gt, rendered, perceptual),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del gaussian, rendered
            torch.cuda.empty_cache()

    write_csv(OUTPUT / "metrics_per_render.csv", rows)
    summary = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary.append({
            "variant": variant,
            "view_count": len(selected),
            **{
                f"{metric}_mean": float(np.mean([row[metric] for row in selected]))
                for metric in ("psnr", "ssim", "lpips", "mse", "near_white_fraction")
            },
            "gaussian_count_mean": float(np.mean([row["gaussian_count"] for row in selected])),
        })
    write_csv(OUTPUT / "metrics_summary.csv", summary)

    for slot, images in rendered_by_slot.items():
        labels = ["GT", *variants]
        thumb_w, thumb_h, label_h = 240, 420, 48
        sheet = Image.new("RGB", (len(labels) * thumb_w, thumb_h + label_h), "white")
        draw = ImageDraw.Draw(sheet)
        for column, label in enumerate(labels):
            copy = images[label].copy()
            copy.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = column * thumb_w + (thumb_w - copy.width) // 2
            y = (thumb_h - copy.height) // 2
            sheet.paste(copy, (x, y))
            row = next((r for r in rows if r["target_slot"] == slot and r["variant"] == label), None)
            text = label if row is None else f"{label}\nPSNR {row['psnr']:.2f}  SSIM {row['ssim']:.3f}"
            draw.multiline_text((column * thumb_w + 4, thumb_h + 3), text, fill="black")
        sheet.save(OUTPUT / f"view_{slot:02d}_comparison.png")

    protocol = {
        "scene": SCENE,
        "slots": slots,
        "gaussian_shapes": shapes,
        "total_raw_gaussians": int(raw_mask.sum()),
        "total_unpadded_gaussians": int(valid.sum()),
        "padding_gaussians": int((~valid).sum()),
        "padding_fraction": float((~valid).float().mean()),
        "scale_p99": float(scale_p99),
        "opacity_min": float(opacity.min()),
        "opacity_median": float(opacity.median()),
        "opacity_max": float(opacity.max()),
        "interpretation": {
            "source_only_unpad": "Only Gaussians predicted from the target source view.",
            "others_only_unpad": "Only Gaussians predicted from the other seven views.",
            "source_plus_others_01": "Target-view opacity unchanged; other-view opacity multiplied by 0.1.",
        },
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "protocol": protocol}, indent=2), flush=True)


if __name__ == "__main__":
    main()
