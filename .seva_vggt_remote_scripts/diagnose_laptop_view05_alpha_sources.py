#!/usr/bin/env python3
"""Decompose padding-masked VGGT Gaussian rendering by alpha, depth, and source view."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


SCENE = "laptop_373_41558_82973"
TARGET_SLOT = 5
MODEL_CANVAS = 518
PATCH = 14
BRANCH = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/reconstruction/scenes") / SCENE / "real8"
OUTPUT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/gaussian_occlusion_diagnostic/alpha_source_view05")


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


def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


def image_metrics(gt: torch.Tensor, pred_hwc: torch.Tensor) -> dict:
    pred = pred_hwc.permute(2, 0, 1).clamp(0, 1)
    mse = torch.mean((gt - pred) ** 2).item()
    gt_np = gt.permute(1, 2, 0).cpu().numpy()
    pred_np = pred.permute(1, 2, 0).cpu().numpy()
    return {
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "ssim": float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0)),
    }


def save_depth(depth: torch.Tensor, alpha: torch.Tensor, path: Path) -> None:
    valid = (alpha > 0.01) & torch.isfinite(depth) & (depth > 0)
    normalized = torch.zeros_like(depth)
    if valid.any():
        values = depth[valid]
        lo = torch.quantile(values, 0.01)
        hi = torch.quantile(values, 0.99)
        normalized[valid] = ((depth[valid] - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
        normalized[valid] = 1.0 - normalized[valid]
    tensor_to_pil(normalized.unsqueeze(0)).save(path)


def fitted(image: Image.Image, width: int, height: int) -> Image.Image:
    out = Image.new("RGB", (width, height), "white")
    copy = image.convert("RGB")
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    out.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/root/vggt")
    from gsplat import rasterization
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda:0")
    reconstruction = torch.load(BRANCH / "vggt_reconstruction.pt", map_location="cpu")
    input_paths = json.loads((BRANCH / "metrics.json").read_text())["input_paths"]
    images, geoms = [], []
    valid = torch.zeros((8, MODEL_CANVAS, MODEL_CANVAS), dtype=torch.bool, device=device)
    for source, path in enumerate(input_paths):
        image = Image.open(path).convert("RGB")
        geom = preprocess_geometry(*image.size)
        images.append(image)
        geoms.append(geom)
        x0, x1 = geom["pad_left"], MODEL_CANVAS - geom["pad_right"]
        y0, y1 = geom["pad_top"], MODEL_CANVAS - geom["pad_bottom"]
        valid[source, y0:y1, x0:x1] = True

    grid = {
        key: value[0].float().reshape(8, MODEL_CANVAS, MODEL_CANVAS, value.shape[-1]).to(device)
        for key, value in reconstruction["gaussians"].items()
        if torch.is_tensor(value) and value.ndim >= 2
    }
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        reconstruction["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS)
    )
    extrinsic = extrinsic[0].to(device)
    intrinsic = intrinsic[0].to(device)
    geom = geoms[TARGET_SLOT]
    K = native_intrinsic(intrinsic[TARGET_SLOT], geom)
    gt = pil_to_tensor(images[TARGET_SLOT], device)

    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = extrinsic[TARGET_SLOT, :3, :3]
    viewmat[:3, 3] = extrinsic[TARGET_SLOT, :3, 3]
    viewmat = viewmat[None]
    Ks = K[None]

    rows = []
    rendered_items = []

    def render(name: str, select: torch.Tensor, opacity_transform=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = {key: value[select] for key, value in grid.items()}
        opacity = parts["opacity"].squeeze(-1)
        if opacity_transform is not None:
            opacity = opacity_transform(opacity)
        with torch.inference_mode():
            out, alpha, _ = rasterization(
                means=parts["xyz"], quats=parts["rotation"], scales=parts["scale"],
                opacities=opacity, colors=parts["color"], viewmats=viewmat, Ks=Ks,
                width=geom["native_width"], height=geom["native_height"],
                near_plane=0.01, far_plane=1e10, render_mode="RGB+ED",
            )
        rgb = out[0, ..., :3].clamp(0, 1)
        depth = out[0, ..., 3]
        a = alpha[0, ..., 0].clamp(0, 1)
        white = rgb + (1.0 - a[..., None])
        black_path = OUTPUT / f"{name}_rgb_black.png"
        white_path = OUTPUT / f"{name}_rgb_white.png"
        alpha_path = OUTPUT / f"{name}_alpha.png"
        depth_path = OUTPUT / f"{name}_depth.png"
        tensor_to_pil(rgb.permute(2, 0, 1)).save(black_path)
        tensor_to_pil(white.permute(2, 0, 1)).save(white_path)
        tensor_to_pil(a.unsqueeze(0)).save(alpha_path)
        save_depth(depth, a, depth_path)
        result = image_metrics(gt, rgb)
        result.update({
            "name": name,
            "gaussian_count": int(select.sum()),
            "opacity_mean": float(opacity.mean()),
            "opacity_median": float(opacity.median()),
            "opacity_p90": float(torch.quantile(opacity, 0.90)),
            "alpha_mean": float(a.mean()),
            "alpha_gt_001_fraction": float((a > 0.01).float().mean()),
            "alpha_gt_050_fraction": float((a > 0.50).float().mean()),
            "alpha_lt_001_fraction": float((a < 0.01).float().mean()),
        })
        rows.append(result)
        rendered_items.append((name, Image.open(black_path), Image.open(alpha_path), Image.open(depth_path)))
        return rgb, a, depth

    all_valid = valid
    render("unpad_all", all_valid)
    render("unpad_all_opacity_x050", all_valid, lambda x: x * 0.50)
    render("unpad_all_opacity_cap050", all_valid, lambda x: x.clamp(max=0.50))
    render("source_target_only", valid[TARGET_SLOT][None].expand_as(valid) & (torch.arange(8, device=device)[:, None, None] == TARGET_SLOT))
    render("other_sources_only", all_valid & (torch.arange(8, device=device)[:, None, None] != TARGET_SLOT))
    for source in range(8):
        select = valid & (torch.arange(8, device=device)[:, None, None] == source)
        render(f"source_{source:02d}_only", select)

    with (OUTPUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    gt_path = OUTPUT / "gt_view05.png"
    images[TARGET_SLOT].save(gt_path)
    cell_w, cell_h, label_w = 260, 240, 180
    sheet = Image.new("RGB", (label_w + cell_w * 3, cell_h * (len(rendered_items) + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), f"{SCENE} target view {TARGET_SLOT:02d}", fill="black")
    for col, title in enumerate(("RGB on black", "alpha", "expected depth")):
        draw.text((label_w + col * cell_w + 5, 5), title, fill="black")
    sheet.paste(fitted(images[TARGET_SLOT], cell_w, cell_h - 25), (label_w, 25))
    draw.text((5, 40), "GT", fill="black")
    row_map = {r["name"]: r for r in rows}
    for index, (name, rgb_image, alpha_image, depth_image) in enumerate(rendered_items, start=1):
        y = index * cell_h
        row = row_map[name]
        draw.text((5, y + 8), name, fill="black")
        draw.text((5, y + 28), f"PSNR {row['psnr']:.2f}", fill="black")
        draw.text((5, y + 48), f"alpha {row['alpha_mean']:.3f}", fill="black")
        for col, image in enumerate((rgb_image, alpha_image, depth_image)):
            sheet.paste(fitted(image, cell_w, cell_h - 25), (label_w + col * cell_w, y + 25))
    sheet.save(OUTPUT / "alpha_source_depth_sheet.jpg", quality=94)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
