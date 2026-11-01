#!/usr/bin/env python3
"""Thirty-scene SEVA->VGGT depth, camera, dedup, and ghosting diagnosis.

The script reuses saved real8 and 1real+7gen reconstructions.  It does not
rerun SEVA or VGGT and does not change model weights.  The intervention is an
offline opacity gate used only to test whether depth-inconsistent generated
pixels are causally related to reconstruction artifacts.
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
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity


PROTOCOL_VERSION = "depth_ghost_diagnosis_v1"
H = 518
W = 518
N = H * W
THRESHOLDS = (0.10, 0.20, 0.30)
CONDITIONS = (
    "original",
    "gt_depth_oracle_0p10",
    "gt_depth_oracle_0p20",
    "gt_depth_oracle_0p30",
    "source_warp_front_0p20",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("/root/seva_vggt/outputs/cfg_sweep30_t8_s12/cfg_2"),
    )
    parser.add_argument(
        "--manifest-root", type=Path,
        default=Path("/root/seva_vggt/config/benchmark30"),
    )
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("/root/seva_vggt/outputs/cfg_sweep30_t8_s12/benchmark30_scenes.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/root/seva_vggt/outputs/depth_ghost30_cfg2_t8_s12"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def stats(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {k: None for k in ("count", "mean", "min", "max", "sum", "median", "std", "ci95_low", "ci95_high")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = 1.96 * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "count": int(arr.size), "mean": mean, "min": float(arr.min()),
        "max": float(arr.max()), "sum": float(arr.sum()),
        "median": float(np.median(arr)), "std": std,
        "ci95_low": mean - half, "ci95_high": mean + half,
    }


def depth_metrics(reference: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int | None]:
    valid = mask.bool() & torch.isfinite(reference) & torch.isfinite(prediction) & (reference > 1e-6) & (prediction > 1e-6)
    count = int(valid.sum().item())
    if count == 0:
        return {k: None for k in ("count", "mae", "absrel", "rmse", "rmse_log", "delta1", "delta2", "delta3", "median_rel")}
    ref = reference[valid].float()
    pred = prediction[valid].float()
    diff = pred - ref
    rel = diff.abs() / ref.clamp_min(1e-6)
    ratio = torch.maximum(ref / pred.clamp_min(1e-6), pred / ref.clamp_min(1e-6))
    return {
        "count": count,
        "mae": finite(diff.abs().mean().item()),
        "absrel": finite(rel.mean().item()),
        "rmse": finite(torch.sqrt((diff * diff).mean()).item()),
        "rmse_log": finite(torch.sqrt(((ref.log() - pred.log()) ** 2).mean()).item()),
        "delta1": finite((ratio < 1.25).float().mean().item()),
        "delta2": finite((ratio < 1.25 ** 2).float().mean().item()),
        "delta3": finite((ratio < 1.25 ** 3).float().mean().item()),
        "median_rel": finite(rel.median().item()),
    }


def add_metrics(row: dict[str, Any], prefix: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        row[f"{prefix}.{key}"] = value


def preprocess_mask(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(path).convert("L")
    width, height = image.size
    if width >= height:
        new_width = W
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = H
        new_width = round(width * (new_height / height) / 14) * 14
    resized = image.resize((new_width, new_height), Image.Resampling.NEAREST)
    arr = np.asarray(resized, dtype=np.uint8) > 127
    out = np.zeros((H, W), dtype=bool)
    valid = np.zeros((H, W), dtype=bool)
    top = (H - new_height) // 2
    left = (W - new_width) // 2
    out[top:top + new_height, left:left + new_width] = arr
    valid[top:top + new_height, left:left + new_width] = True
    return torch.from_numpy(out), torch.from_numpy(valid)


def object_regions(mask: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    m = mask.bool()
    kernel = torch.ones(1, 1, 7, 7)
    x = m.float()[None, None]
    dilated = F.conv2d(x, kernel, padding=3)[0, 0] > 0
    eroded = F.conv2d(x, kernel, padding=3)[0, 0] >= 49
    boundary = dilated & ~eroded & valid
    ys, xs = torch.where(m)
    lower = torch.zeros_like(m)
    if ys.numel():
        cutoff = int(float(ys.min()) + 0.55 * float(ys.max() - ys.min() + 1))
        lower[cutoff:] = m[cutoff:]
    return {
        "all": valid,
        "object": m & valid,
        "boundary": boundary,
        "lower_object": lower & valid,
        "background": (~m) & valid,
    }


def transformed_gt_camera(view: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    extrinsic = torch.tensor(view["extrinsic"], dtype=torch.float32)
    intrinsic = torch.tensor(view["intrinsic"], dtype=torch.float32)
    width, height = int(view["width"]), int(view["height"])
    if width >= height:
        new_width = W
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = H
        new_width = round(width * (new_height / height) / 14) * 14
    sx, sy = new_width / width, new_height / height
    left, top = (W - new_width) // 2, (H - new_height) // 2
    out = intrinsic.clone()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = out[0, 2] * sx + left
    out[1, 2] = out[1, 2] * sy + top
    return extrinsic, out


def homogeneous(extrinsic: torch.Tensor) -> torch.Tensor:
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=extrinsic.dtype, device=extrinsic.device)
    return torch.cat([extrinsic, bottom], dim=0)


def relative_pose(extrinsics: torch.Tensor, source: int, target: int) -> torch.Tensor:
    return homogeneous(extrinsics[target]) @ torch.linalg.inv(homogeneous(extrinsics[source]))


def rotation_error_degrees(a: torch.Tensor, b: torch.Tensor) -> float:
    r = a[:3, :3] @ b[:3, :3].T
    value = ((torch.trace(r) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(value)).item())


def direction_error_degrees(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na, nb = torch.linalg.norm(a), torch.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return None
    value = torch.dot(a / na, b / nb).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(value)).item())


def scale_from_shared_source(gt_depth: torch.Tensor, mixed_depth: torch.Tensor, mask: torch.Tensor,
                             gt_conf: torch.Tensor, mixed_conf: torch.Tensor) -> tuple[float, int]:
    valid = mask & torch.isfinite(gt_depth) & torch.isfinite(mixed_depth) & (gt_depth > 1e-6) & (mixed_depth > 1e-6)
    if valid.any():
        conf_valid = valid & (gt_conf >= gt_conf[valid].median()) & (mixed_conf >= mixed_conf[valid].median())
        if int(conf_valid.sum()) >= 500:
            valid = conf_valid
    ratio = gt_depth[valid] / mixed_depth[valid]
    if ratio.numel() == 0:
        return 1.0, 0
    return float(ratio.median().item()), int(ratio.numel())


def forward_splat_depth(depth: torch.Tensor, source_mask: torch.Tensor,
                        source_extrinsic: torch.Tensor, source_intrinsic: torch.Tensor,
                        target_extrinsic: torch.Tensor, target_intrinsic: torch.Tensor) -> torch.Tensor:
    device = depth.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32), indexing="ij",
    )
    z = depth.float()
    x = (xx - source_intrinsic[0, 2]) * z / source_intrinsic[0, 0]
    y = (yy - source_intrinsic[1, 2]) * z / source_intrinsic[1, 1]
    cam = torch.stack([x, y, z], dim=-1)
    r = source_extrinsic[:3, :3]
    t = source_extrinsic[:3, 3]
    world = (cam - t) @ r
    rt = target_extrinsic[:3, :3]
    tt = target_extrinsic[:3, 3]
    target_cam = world @ rt.T + tt
    zt = target_cam[..., 2]
    u = target_intrinsic[0, 0] * target_cam[..., 0] / zt.clamp_min(1e-6) + target_intrinsic[0, 2]
    v = target_intrinsic[1, 1] * target_cam[..., 1] / zt.clamp_min(1e-6) + target_intrinsic[1, 2]
    base_valid = source_mask & torch.isfinite(zt) & (zt > 0.05)
    out = torch.full((H * W,), float("inf"), dtype=torch.float32, device=device)
    u0, v0 = torch.floor(u), torch.floor(v)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ui = (u0 + du).long()
        vi = (v0 + dv).long()
        valid = base_valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        idx = (vi[valid] * W + ui[valid]).long()
        out.scatter_reduce_(0, idx, zt[valid], reduce="amin", include_self=True)
    out = out.reshape(H, W)
    out[~torch.isfinite(out)] = float("nan")
    return out


def warp_classification(warped: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        tolerance: float = 0.20) -> dict[str, float | int | None]:
    valid = mask & torch.isfinite(warped) & torch.isfinite(target) & (warped > 1e-6) & (target > 1e-6)
    count = int(valid.sum())
    denom = int(mask.sum())
    if count == 0:
        return {"count": 0, "coverage": 0.0, "match_rate": None, "front_conflict_rate": None, "behind_occluded_rate": None}
    wv, tv = warped[valid], target[valid]
    rel_signed = (wv - tv) / tv.clamp_min(1e-6)
    return {
        "count": count,
        "coverage": count / max(denom, 1),
        "match_rate": finite((rel_signed.abs() <= tolerance).float().mean().item()),
        "front_conflict_rate": finite((rel_signed < -tolerance).float().mean().item()),
        "behind_occluded_rate": finite((rel_signed > tolerance).float().mean().item()),
    }


@torch.inference_mode()
def dedup_diagnostics(depth: torch.Tensor, confidence: torch.Tensor,
                      extrinsics: torch.Tensor, intrinsics: torch.Tensor,
                      opacity: torch.Tensor) -> dict[str, Any]:
    from vggt.models.vggt import unproject_depth_to_world_torch

    depth_b = depth[None, ..., None]
    extr_b = extrinsics[None]
    intr_b = intrinsics[None]
    xyz = unproject_depth_to_world_torch(depth_b.float(), extr_b.float(), intr_b.float())[0]
    world_maps = xyz.permute(0, 3, 1, 2)
    conf_maps = confidence[:, None]
    raw = torch.ones(8, H, W, device=depth.device)
    pair_rel_values: list[float] = []
    for source in range(8):
        points = xyz[source].reshape(H * W, 3)
        conf_source = confidence[source].reshape(H * W)
        for observation in range(8):
            if source == observation:
                continue
            r = extrinsics[observation, :3, :3]
            t = extrinsics[observation, :3, 3]
            k = intrinsics[observation]
            cam = points @ r.T + t
            z = cam[:, 2]
            u = k[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + k[0, 2]
            v = k[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + k[1, 2]
            in_fov = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.05)
            grid = torch.stack([2 * u / (W - 1) - 1, 2 * v / (H - 1) - 1], dim=-1).view(1, H * W, 1, 2)
            obs_world = F.grid_sample(world_maps[observation:observation + 1], grid, mode="bilinear", align_corners=True)
            obs_world = obs_world.view(3, H * W).T
            obs_cam = obs_world @ r.T + t
            observed_depth = obs_cam[:, 2]
            rel = (z - observed_depth).abs() / observed_depth.clamp_min(0.1)
            obs_conf = F.grid_sample(conf_maps[observation:observation + 1], grid, mode="bilinear", align_corners=True).view(H * W)
            loses = in_fov & (rel < 0.20) & (obs_conf > conf_source)
            raw[source].view(-1)[loses] = 0.0
            if in_fov.any():
                pair_rel_values.append(float(rel[in_fov].median().item()))
    raw_mean = float(raw.mean().item())
    fallback = raw_mean > 0.80
    actual = torch.ones_like(raw) if fallback else raw
    opa = opacity.reshape(8, H, W).float()
    eps = 1e-6
    bce = -(actual * (opa + eps).log() + (1 - actual) * (1 - opa + eps).log()).mean()
    rejected = raw < 0.5
    kept = raw > 0.5
    return {
        "raw_keep_ratio": raw_mean,
        "actual_keep_ratio": float(actual.mean().item()),
        "fallback_triggered": bool(fallback),
        "raw_keep_per_frame": [float(x) for x in raw.mean(dim=(1, 2)).cpu()],
        "opacity_bce_to_actual_mask": float(bce.item()),
        "opacity_mean_raw_kept": float(opa[kept].mean().item()) if kept.any() else None,
        "opacity_mean_raw_rejected": float(opa[rejected].mean().item()) if rejected.any() else None,
        "high_opacity_rejected_rate": float((opa[rejected] > 0.5).float().mean().item()) if rejected.any() else None,
        "pair_median_relative_depth_error": float(np.mean(pair_rel_values)) if pair_rel_values else None,
    }


@torch.inference_mode()
def gaussian_front_conflict(xyz: torch.Tensor, opacity: torch.Tensor,
                            source_depth: torch.Tensor, source_mask: torch.Tensor,
                            source_extrinsic: torch.Tensor, source_intrinsic: torch.Tensor,
                            tolerance: float = 0.20) -> dict[str, float | int | None]:
    points = xyz.float()
    opa = opacity.reshape(-1).float()
    r = source_extrinsic[:3, :3]
    t = source_extrinsic[:3, 3]
    cam = points @ r.T + t
    z = cam[:, 2]
    u = source_intrinsic[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + source_intrinsic[0, 2]
    v = source_intrinsic[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + source_intrinsic[1, 2]
    grid = torch.stack([2 * u / (W - 1) - 1, 2 * v / (H - 1) - 1], dim=-1).view(1, -1, 1, 2)
    sampled_depth = F.grid_sample(source_depth[None, None], grid, mode="bilinear", align_corners=True).view(-1)
    sampled_mask = F.grid_sample(source_mask.float()[None, None], grid, mode="nearest", align_corners=True).view(-1) > 0.5
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.05) & sampled_mask & (sampled_depth > 1e-6)
    high = valid & (opa > 0.5)
    if not high.any():
        return {"high_opacity_projected_count": 0, "front_conflict_rate": None, "median_relative_error": None}
    signed = (z[high] - sampled_depth[high]) / sampled_depth[high].clamp_min(1e-6)
    return {
        "high_opacity_projected_count": int(high.sum().item()),
        "front_conflict_rate": finite((signed < -tolerance).float().mean().item()),
        "median_relative_error": finite(signed.abs().median().item()),
    }


def image_metrics(reference: torch.Tensor, prediction: torch.Tensor,
                  lpips_model: torch.nn.Module) -> dict[str, float | None]:
    reference = reference.float().clamp(0, 1)
    prediction = prediction.float().clamp(0, 1)
    mse = torch.mean((reference - prediction) ** 2).item()
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    ref_np = reference.permute(1, 2, 0).detach().cpu().numpy()
    pred_np = prediction.permute(1, 2, 0).detach().cpu().numpy()
    ssim = structural_similarity(ref_np, pred_np, channel_axis=2, data_range=1.0)
    with torch.inference_mode():
        value = lpips_model(reference[None] * 2 - 1, prediction[None] * 2 - 1)
    return {"psnr": finite(psnr), "ssim": finite(ssim), "lpips": finite(value.flatten()[0].item())}


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def colorize(value: torch.Tensor, mask: torch.Tensor, vmin: float, vmax: float, error: bool = False) -> Image.Image:
    x = ((value.detach().float().cpu() - vmin) / max(vmax - vmin, 1e-8)).clamp(0, 1).numpy()
    if error:
        r = np.clip(2.0 * x, 0, 1)
        g = np.clip(2.0 * (1.0 - np.abs(x - 0.5)), 0, 1)
        b = np.clip(1.0 - 2.0 * x, 0, 1)
    else:
        r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    valid = mask.detach().cpu().numpy().astype(bool) & np.isfinite(value.detach().cpu().numpy())
    rgb[~valid] = 1.0
    return Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), mode="RGB")


def save_depth_sheet(path: Path, rows: list[dict[str, Any]], real_images: torch.Tensor,
                     mixed_images: torch.Tensor, gt_depth: torch.Tensor,
                     mixed_depth_scaled: torch.Tensor, warps: list[torch.Tensor],
                     masks: list[torch.Tensor]) -> None:
    thumb = 150
    label = 110
    header = 34
    columns = ("Real GT", "SEVA", "GT depth", "SEVA depth", "GT-SEVA rel", "source warp", "warp-SEVA rel")
    sheet = Image.new("RGB", (label + len(columns) * thumb, header + len(rows) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(columns):
        draw.text((label + c * thumb + 8, 10), name, fill="black")
    for rix, row in enumerate(rows):
        slot = int(row["slot"])
        y = header + rix * thumb
        draw.multiline_text((5, y + 8), f"view {row['view_index']:02d}\nSEVA", fill="black", spacing=3)
        obj = masks[slot]
        vals = torch.cat([gt_depth[slot][obj], mixed_depth_scaled[slot][obj]])
        vals = vals[torch.isfinite(vals)]
        if vals.numel():
            lo, hi = float(torch.quantile(vals, 0.02)), float(torch.quantile(vals, 0.98))
        else:
            lo, hi = 0.0, 1.0
        direct_rel = (gt_depth[slot] - mixed_depth_scaled[slot]).abs() / gt_depth[slot].clamp_min(1e-6)
        warp = warps[slot]
        warp_rel = (warp - mixed_depth_scaled[slot]).abs() / mixed_depth_scaled[slot].clamp_min(1e-6)
        panels = [
            tensor_to_pil(real_images[slot]), tensor_to_pil(mixed_images[slot]),
            colorize(gt_depth[slot], obj, lo, hi), colorize(mixed_depth_scaled[slot], obj, lo, hi),
            colorize(direct_rel, obj, 0, 0.5, error=True), colorize(warp, torch.isfinite(warp) & obj, lo, hi),
            colorize(warp_rel, torch.isfinite(warp) & obj, 0, 0.5, error=True),
        ]
        for c, panel in enumerate(panels):
            sheet.paste(panel.resize((thumb, thumb), Image.Resampling.LANCZOS), (label + c * thumb, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def save_ablation_sheet(path: Path, view_indices: list[int], gt_images: torch.Tensor,
                        mixed_images: torch.Tensor, renders: dict[str, list[torch.Tensor]]) -> None:
    shown = ("original", "gt_depth_oracle_0p20", "source_warp_front_0p20")
    thumb = 170
    label = 100
    header = 34
    columns = ("Real GT", "SEVA input", "Original", "GT-depth gate", "Source-warp gate")
    sheet = Image.new("RGB", (label + len(columns) * thumb, header + len(view_indices) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(columns):
        draw.text((label + c * thumb + 8, 10), name, fill="black")
    for row, view_index in enumerate(view_indices):
        slot = row + 1
        y = header + row * thumb
        draw.multiline_text((5, y + 8), f"view {view_index:02d}\nSEVA", fill="black", spacing=3)
        panels = [gt_images[slot], mixed_images[slot]] + [renders[name][row] for name in shown]
        for c, tensor in enumerate(panels):
            panel = tensor_to_pil(tensor).resize((thumb, thumb), Image.Resampling.LANCZOS)
            sheet.paste(panel, (label + c * thumb, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def scene_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def process_scene(scene: str, args: argparse.Namespace, device: torch.device,
                  lpips_model: torch.nn.Module | None) -> dict[str, Any]:
    scene_root = args.input_root / scene
    output = args.output_root / "scenes" / scene
    result_path = output / "result.json"
    if result_path.is_file() and not args.overwrite:
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("protocol_version") == PROTOCOL_VERSION:
            return previous

    metadata = json.loads((scene_root / "run_metadata_seva_1real_7gen.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.manifest_root / f"{scene}.json").read_text(encoding="utf-8"))
    baseline = [int(x) for x in metadata["baseline_indices"]]
    if len(baseline) != 8 or len(metadata["input_indices"]) != 1:
        raise RuntimeError(f"unexpected protocol for {scene}: baseline={baseline}")
    real = torch.load(scene_root / "real8/vggt_reconstruction.pt", map_location="cpu", weights_only=False)
    mixed = torch.load(scene_root / "seva_1real_7gen/vggt_reconstruction.pt", map_location="cpu", weights_only=False)

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    real_depth = real["depth"][0, ..., 0].float().to(device)
    mixed_depth = mixed["depth"][0, ..., 0].float().to(device)
    real_conf = real["depth_conf"][0].float().to(device)
    mixed_conf = mixed["depth_conf"][0].float().to(device)
    real_extr, real_intr = pose_encoding_to_extri_intri(real["pose_enc"].float(), image_size_hw=(H, W))
    mixed_extr, mixed_intr = pose_encoding_to_extri_intri(mixed["pose_enc"].float(), image_size_hw=(H, W))
    real_extr, real_intr = real_extr[0].to(device), real_intr[0].to(device)
    mixed_extr, mixed_intr = mixed_extr[0].to(device), mixed_intr[0].to(device)

    object_masks: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    mask_fallbacks: list[bool] = []
    regions: list[dict[str, torch.Tensor]] = []
    gt_extrinsics, gt_intrinsics = [], []
    for index in baseline:
        view = manifest["views"][index]
        obj, valid = preprocess_mask(view["mask"])
        fallback = not bool(obj.any())
        if fallback:
            obj = valid.clone()
        mask_fallbacks.append(fallback)
        object_masks.append(obj.to(device))
        valid_masks.append(valid.to(device))
        regions.append({k: v.to(device) for k, v in object_regions(obj, valid).items()})
        ex, intr = transformed_gt_camera(view)
        gt_extrinsics.append(ex)
        gt_intrinsics.append(intr)
    gt_extr = torch.stack(gt_extrinsics).to(device)
    gt_intr = torch.stack(gt_intrinsics).to(device)

    source_mask = regions[0]["object"]
    scale, scale_pixels = scale_from_shared_source(
        real_depth[0], mixed_depth[0], source_mask, real_conf[0], mixed_conf[0]
    )
    mixed_scaled = mixed_depth * scale
    source_alignment = depth_metrics(real_depth[0], mixed_scaled[0], source_mask)

    real_opa = real["gaussians"]["opacity"][0].float().to(device)
    mixed_opa = mixed["gaussians"]["opacity"][0].float().to(device)
    dedup_real = dedup_diagnostics(real_depth, real_conf, real_extr, real_intr, real_opa)
    dedup_mixed = dedup_diagnostics(mixed_depth, mixed_conf, mixed_extr, mixed_intr, mixed_opa)

    real_baselines = torch.tensor([torch.linalg.norm(relative_pose(real_extr, 0, s)[:3, 3]).item() for s in range(1, 8)], device=device)
    mixed_baselines = torch.tensor([torch.linalg.norm(relative_pose(mixed_extr, 0, s)[:3, 3]).item() for s in range(1, 8)], device=device)
    gt_baselines = torch.tensor([torch.linalg.norm(relative_pose(gt_extr, 0, s)[:3, 3]).item() for s in range(1, 8)], device=device)
    med_real = float(real_baselines.median().item())
    med_mixed = float(mixed_baselines.median().item())
    med_gt = float(gt_baselines.median().item())

    per_view: list[dict[str, Any]] = []
    warps: list[torch.Tensor] = [torch.full((H, W), float("nan"), device=device)]
    condition_masks: dict[str, list[torch.Tensor]] = {name: [torch.zeros((H, W), dtype=torch.bool)] for name in CONDITIONS}
    mixed_xyz = mixed["gaussians"]["xyz"][0].float().to(device)
    for slot in range(1, 8):
        view_index = baseline[slot]
        row: dict[str, Any] = {"scene": scene, "slot": slot, "view_index": view_index, "source_type": "seva"}
        row["object_mask_fallback_to_valid_image"] = mask_fallbacks[slot]
        row["depth_scale_mixed_to_real"] = scale
        row["depth_scale_alignment_pixels"] = scale_pixels
        row["real_depth_conf_mean"] = finite(real_conf[slot][regions[slot]["object"]].mean().item())
        row["mixed_depth_conf_mean"] = finite(mixed_conf[slot][regions[slot]["object"]].mean().item())
        row["mixed_to_real_conf_ratio"] = row["mixed_depth_conf_mean"] / max(row["real_depth_conf_mean"], 1e-8)
        for region_name, region_mask in regions[slot].items():
            add_metrics(row, f"gt_vs_seva_depth.{region_name}", depth_metrics(real_depth[slot], mixed_scaled[slot], region_mask))

        warp = forward_splat_depth(
            real_depth[0], source_mask, real_extr[0], real_intr[0], real_extr[slot], real_intr[slot]
        )
        warps.append(warp)
        obj = regions[slot]["object"]
        add_metrics(row, "source_warp_vs_gt_depth.object", depth_metrics(real_depth[slot], warp, obj & torch.isfinite(warp)))
        add_metrics(row, "source_warp_vs_seva_depth.object", depth_metrics(mixed_scaled[slot], warp, obj & torch.isfinite(warp)))
        add_metrics(row, "source_warp_vs_gt_class", warp_classification(warp, real_depth[slot], obj))
        add_metrics(row, "source_warp_vs_seva_class", warp_classification(warp, mixed_scaled[slot], obj))

        rel_gt_seva = (real_depth[slot] - mixed_scaled[slot]).abs() / real_depth[slot].clamp_min(1e-6)
        for threshold in THRESHOLDS:
            key = f"gt_depth_oracle_{threshold:.2f}".replace("0.", "0p")
            condition_masks[key].append(obj & torch.isfinite(rel_gt_seva) & (rel_gt_seva > threshold))
        source_front = obj & torch.isfinite(warp) & (warp < mixed_scaled[slot] * 0.80)
        condition_masks["source_warp_front_0p20"].append(source_front)
        condition_masks["original"].append(torch.zeros((H, W), dtype=torch.bool, device=device))

        for threshold in THRESHOLDS:
            bad = obj & torch.isfinite(rel_gt_seva) & (rel_gt_seva > threshold)
            row[f"gt_vs_seva_depth.object.inconsistent_rate_{threshold:.2f}"] = finite(bad.float().sum().item() / max(int(obj.sum()), 1))
            row[f"opacity_on_inconsistent.mean_{threshold:.2f}"] = finite(mixed_opa[slot].reshape(H, W)[bad].mean().item()) if bad.any() else None
            row[f"opacity_on_inconsistent.high_rate_{threshold:.2f}"] = finite((mixed_opa[slot].reshape(H, W)[bad] > 0.5).float().mean().item()) if bad.any() else None

        real_rel = relative_pose(real_extr, 0, slot)
        mixed_rel = relative_pose(mixed_extr, 0, slot)
        gt_rel = relative_pose(gt_extr, 0, slot)
        row["pose.real_vs_gt.rotation_deg"] = rotation_error_degrees(real_rel, gt_rel)
        row["pose.mixed_vs_gt.rotation_deg"] = rotation_error_degrees(mixed_rel, gt_rel)
        row["pose.mixed_vs_real.rotation_deg"] = rotation_error_degrees(mixed_rel, real_rel)
        row["pose.real_vs_gt.translation_direction_deg"] = direction_error_degrees(real_rel[:3, 3], gt_rel[:3, 3])
        row["pose.mixed_vs_gt.translation_direction_deg"] = direction_error_degrees(mixed_rel[:3, 3], gt_rel[:3, 3])
        row["pose.mixed_vs_real.translation_direction_deg"] = direction_error_degrees(mixed_rel[:3, 3], real_rel[:3, 3])
        row["pose.real_normalized_baseline"] = float(torch.linalg.norm(real_rel[:3, 3]).item() / max(med_real, 1e-8))
        row["pose.mixed_normalized_baseline"] = float(torch.linalg.norm(mixed_rel[:3, 3]).item() / max(med_mixed, 1e-8))
        row["pose.gt_normalized_baseline"] = float(torch.linalg.norm(gt_rel[:3, 3]).item() / max(med_gt, 1e-8))
        row["pose.mixed_vs_gt.normalized_baseline_abs_error"] = abs(row["pose.mixed_normalized_baseline"] - row["pose.gt_normalized_baseline"])
        row["pose.real_vs_gt.normalized_baseline_abs_error"] = abs(row["pose.real_normalized_baseline"] - row["pose.gt_normalized_baseline"])
        row["intrinsics.mixed_vs_real.fx_relative_error"] = abs(float(mixed_intr[slot, 0, 0] / real_intr[slot, 0, 0] - 1))
        row["intrinsics.mixed_vs_gt.fx_relative_error"] = abs(float(mixed_intr[slot, 0, 0] / gt_intr[slot, 0, 0] - 1))

        risk = gaussian_front_conflict(
            mixed_xyz[slot], mixed_opa[slot], mixed_depth[0], source_mask,
            mixed_extr[0], mixed_intr[0], tolerance=0.20,
        )
        add_metrics(row, "gaussian_to_source", risk)
        per_view.append(row)

    real_images = real["images"][0].float()
    mixed_images = mixed["images"][0].float()
    save_depth_sheet(
        output / "depth_comparison.jpg", per_view, real_images, mixed_images,
        real_depth.detach().cpu(), mixed_scaled.detach().cpu(),
        [x.detach().cpu() for x in warps], [x.detach().cpu() for x in object_masks],
    )

    ablation_rows: list[dict[str, Any]] = []
    if not args.no_render:
        from vggt.utils.gs_optimizer import render_gaussians

        gaussian_base = {
            key: value[0].reshape(-1, value.shape[-1]).float().to(device)
            for key, value in mixed["gaussians"].items()
            if torch.is_tensor(value) and value.ndim >= 4
        }
        saved_renders: dict[str, list[torch.Tensor]] = defaultdict(list)
        for condition in CONDITIONS:
            opacity = mixed_opa.clone().reshape(8, H, W, 1)
            removed = []
            for slot in range(1, 8):
                gate = condition_masks[condition][slot].to(device)
                removed.append(int(gate.sum().item()))
                opacity_slot = opacity[slot, ..., 0]
                opacity_slot[gate] = 0.0
            gaussian_condition = dict(gaussian_base)
            gaussian_condition["opacity"] = opacity.reshape(-1, 1)
            for slot in range(1, 8):
                rendered = render_gaussians(
                    gaussian_condition["xyz"], gaussian_condition["rotation"],
                    gaussian_condition["scale"], gaussian_condition["opacity"],
                    gaussian_condition["color"], mixed_extr[slot], mixed_intr[slot], H, W,
                )
                values = image_metrics(real_images[slot].to(device), rendered, lpips_model)
                row = {
                    "scene": scene, "slot": slot, "view_index": baseline[slot],
                    "condition": condition, "removed_pixels": removed[slot - 1],
                    "removed_fraction_of_generated_slot": removed[slot - 1] / N,
                    **values,
                }
                ablation_rows.append(row)
                if condition in ("original", "gt_depth_oracle_0p20", "source_warp_front_0p20"):
                    saved_renders[condition].append(rendered.detach().cpu())
                del rendered
            del opacity, gaussian_condition
        save_ablation_sheet(
            output / "opacity_gate_ablation.jpg", baseline[1:], real_images, mixed_images, saved_renders
        )
        del gaussian_base, saved_renders

    scene_row: dict[str, Any] = {
        "scene": scene,
        "category": scene.split("_")[0],
        "depth_scale_mixed_to_real": scale,
        "source_alignment_absrel": source_alignment["absrel"],
        "source_alignment_rmse": source_alignment["rmse"],
        "object_mask_fallback_view_count": int(sum(mask_fallbacks)),
        "dedup_real_raw_keep_ratio": dedup_real["raw_keep_ratio"],
        "dedup_real_fallback_triggered": dedup_real["fallback_triggered"],
        "dedup_real_high_opacity_rejected_rate": dedup_real["high_opacity_rejected_rate"],
        "dedup_mixed_raw_keep_ratio": dedup_mixed["raw_keep_ratio"],
        "dedup_mixed_fallback_triggered": dedup_mixed["fallback_triggered"],
        "dedup_mixed_high_opacity_rejected_rate": dedup_mixed["high_opacity_rejected_rate"],
    }
    for key in (
        "gt_vs_seva_depth.object.absrel",
        "gt_vs_seva_depth.object.rmse",
        "gt_vs_seva_depth.boundary.absrel",
        "gt_vs_seva_depth.lower_object.absrel",
        "source_warp_vs_gt_class.front_conflict_rate",
        "source_warp_vs_seva_class.front_conflict_rate",
        "pose.mixed_vs_real.rotation_deg",
        "pose.mixed_vs_real.translation_direction_deg",
        "gaussian_to_source.front_conflict_rate",
        "mixed_to_real_conf_ratio",
        "gt_vs_seva_depth.object.inconsistent_rate_0.20",
    ):
        scene_row[f"mean.{key}"] = scene_mean(per_view, key)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_view.csv", per_view)
    write_csv(output / "opacity_gate_ablation.csv", ablation_rows)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "scene": scene,
        "experiment": {
            "cfg": 2, "T": 8, "sampling_steps": 12, "seed": 23,
            "short_side": 576, "preserve_aspect_ratio": True,
            "chunk_strategy": "nearest-gt", "branch": "1real+7seva",
            "baseline": "8real", "vggt_input_size": [H, W],
            "depth_alignment": "scale-only median ratio on high-confidence pixels of the shared real source object",
            "depth_source": "saved VGGT depth output",
            "camera_primary": "saved VGGT predicted camera; exact dataset camera used for relative-pose audit",
            "intervention": "offline opacity zeroing only; no model or checkpoint changes",
        },
        "scene_summary": scene_row,
        "source_alignment": source_alignment,
        "dedup_real": dedup_real,
        "dedup_mixed": dedup_mixed,
        "per_view": per_view,
        "ablation": ablation_rows,
    }
    atomic_json(result_path, result)
    del real, mixed, real_depth, mixed_depth, mixed_scaled, real_conf, mixed_conf
    del real_opa, mixed_opa, mixed_xyz
    torch.cuda.empty_cache()
    return result


def aggregate(output_root: Path, results: list[dict[str, Any]]) -> None:
    scene_rows = [x["scene_summary"] for x in results]
    view_rows = [row for x in results for row in x["per_view"]]
    ablation_rows = [row for x in results for row in x["ablation"]]
    write_csv(output_root / "per_scene_metrics.csv", scene_rows)
    write_csv(output_root / "per_view_metrics.csv", view_rows)
    write_csv(output_root / "opacity_gate_ablation_per_view.csv", ablation_rows)

    identity = {"scene", "category"}
    summary_rows: list[dict[str, Any]] = []
    for key in sorted(set().union(*(row.keys() for row in scene_rows)) - identity):
        values = [row.get(key) for row in scene_rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        summary_rows.append({"level": "all_scenes", "group": "all", "metric": key, **stats(values)})
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        by_category[row["category"]].append(row)
    for category, rows in sorted(by_category.items()):
        for key in sorted(set().union(*(row.keys() for row in rows)) - identity):
            values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
            summary_rows.append({"level": "category", "group": category, "metric": key, **stats(values)})
    write_csv(output_root / "summary_statistics.csv", summary_rows)

    ablation_summary: list[dict[str, Any]] = []
    original_by_key = {(row["scene"], row["slot"]): row for row in ablation_rows if row["condition"] == "original"}
    for condition in CONDITIONS:
        rows = [row for row in ablation_rows if row["condition"] == condition]
        for metric in ("psnr", "ssim", "lpips", "removed_pixels", "removed_fraction_of_generated_slot"):
            values = [row[metric] for row in rows if row.get(metric) is not None]
            item = {"condition": condition, "metric": metric, **stats(values)}
            if condition != "original" and metric in ("psnr", "ssim", "lpips"):
                deltas = [row[metric] - original_by_key[(row["scene"], row["slot"])][metric] for row in rows]
                delta_stats = stats(deltas)
                for key, value in delta_stats.items():
                    item[f"delta_vs_original_{key}"] = value
                positive = [(d > 0) if metric != "lpips" else (d < 0) for d in deltas]
                item["improved_view_rate"] = float(np.mean(positive)) if positive else None
            ablation_summary.append(item)
    write_csv(output_root / "opacity_gate_ablation_summary.csv", ablation_summary)

    correlations: list[dict[str, Any]] = []
    pairs = (
        ("gt_vs_seva_depth.object.absrel", "gaussian_to_source.front_conflict_rate"),
        ("gt_vs_seva_depth.object.absrel", "source_warp_vs_seva_class.front_conflict_rate"),
        ("gt_vs_seva_depth.object.absrel", "pose.mixed_vs_real.rotation_deg"),
        ("gt_vs_seva_depth.object.absrel", "mixed_to_real_conf_ratio"),
        ("source_warp_vs_seva_class.front_conflict_rate", "gaussian_to_source.front_conflict_rate"),
    )
    for xkey, ykey in pairs:
        pairs_xy = [(float(r[xkey]), float(r[ykey])) for r in view_rows if r.get(xkey) is not None and r.get(ykey) is not None]
        if len(pairs_xy) >= 3:
            x, y = zip(*pairs_xy)
            p = pearsonr(x, y)
            s = spearmanr(x, y)
            correlations.append({
                "x": xkey, "y": ykey, "count": len(x),
                "pearson_r": finite(p.statistic), "pearson_p": finite(p.pvalue),
                "spearman_rho": finite(s.statistic), "spearman_p": finite(s.pvalue),
            })
    write_csv(output_root / "correlations.csv", correlations)
    atomic_json(output_root / "aggregate.json", {
        "protocol_version": PROTOCOL_VERSION,
        "scene_count": len(scene_rows), "per_view_count": len(view_rows),
        "expected_scene_count": 30, "expected_per_view_count": 210,
        "ablation_row_count": len(ablation_rows),
        "summary_statistics": summary_rows,
        "ablation_summary": ablation_summary,
        "correlations": correlations,
    })


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/root/vggt")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    selection = json.loads(args.scene_list.read_text(encoding="utf-8"))
    scenes = list(args.scene) if args.scene else list(selection["scenes"])
    if args.limit:
        scenes = scenes[:args.limit]
    lpips_model = None if args.no_render else lpips.LPIPS(net="alex").eval().to(device)
    status_path = args.output_root / "status.json"
    completed, failed = [], []
    started = time.time()
    for index, scene in enumerate(scenes, start=1):
        try:
            process_scene(scene, args, device, lpips_model)
            completed.append(scene)
            print(f"[{index}/{len(scenes)}] COMPLETE {scene}", flush=True)
        except Exception as error:
            failed.append({"scene": scene, "error": repr(error)})
            print(f"[{index}/{len(scenes)}] FAILED {scene}: {error!r}", flush=True)
        atomic_json(status_path, {
            "protocol_version": PROTOCOL_VERSION, "started_at_unix": started,
            "updated_at_unix": time.time(), "total": len(scenes),
            "completed": completed, "failed": failed,
        })
    results = []
    for scene in scenes:
        path = args.output_root / "scenes" / scene / "result.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    aggregate(args.output_root, results)
    atomic_json(status_path, {
        "protocol_version": PROTOCOL_VERSION, "started_at_unix": started,
        "finished_at_unix": time.time(), "total": len(scenes),
        "completed": completed, "failed": failed,
    })
    if failed:
        raise SystemExit(f"{len(failed)} scenes failed")


if __name__ == "__main__":
    main()
