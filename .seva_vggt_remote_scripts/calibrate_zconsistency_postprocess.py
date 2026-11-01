#!/usr/bin/env python3
"""Calibrate target-view z-consistency filters on representative real8 scenes."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


MODEL_CANVAS = 518
PATCH = 14
RECON_ROOT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/reconstruction/scenes")
OUTPUT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/zconsistency_calibration")
SCENES = [
    "chair_455_64092_126267",
    "laptop_373_41558_82973",
    "bicycle_62_4318_10726",
    "plant_374_42253_84495",
    "toybus_385_45834_91577",
]
MODES = {
    "front_soft_a050_t003_w025": {"alpha": 0.50, "front": 0.03, "front_weight": 0.25, "back": None, "back_weight": 1.0},
    "front_soft_a050_t003_w050": {"alpha": 0.50, "front": 0.03, "front_weight": 0.50, "back": None, "back_weight": 1.0},
    "front_soft_a050_t005_w025": {"alpha": 0.50, "front": 0.05, "front_weight": 0.25, "back": None, "back_weight": 1.0},
    "front_soft_a050_t010_w010": {"alpha": 0.50, "front": 0.10, "front_weight": 0.10, "back": None, "back_weight": 1.0},
    "front_soft_a075_t003_w025": {"alpha": 0.75, "front": 0.03, "front_weight": 0.25, "back": None, "back_weight": 1.0},
}


def geom(width: int, height: int) -> dict:
    if width >= height:
        rw, rh = MODEL_CANVAS, round(height * MODEL_CANVAS / width / PATCH) * PATCH
    else:
        rh, rw = MODEL_CANVAS, round(width * MODEL_CANVAS / height / PATCH) * PATCH
    pw, ph = MODEL_CANVAS - rw, MODEL_CANVAS - rh
    return {"w": width, "h": height, "rw": int(rw), "rh": int(rh),
            "pl": int(pw // 2), "pr": int(pw - pw // 2),
            "pt": int(ph // 2), "pb": int(ph - ph // 2)}


def native_k(k: torch.Tensor, g: dict) -> torch.Tensor:
    out = k.clone()
    sx, sy = g["w"] / g["rw"], g["h"] / g["rh"]
    out[0, 0], out[1, 1] = k[0, 0] * sx, k[1, 1] * sy
    out[0, 2], out[1, 2] = (k[0, 2] - g["pl"]) * sx, (k[1, 2] - g["pt"]) * sy
    return out


def image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict:
    mse = torch.mean((gt - pred) ** 2).item()
    return {
        "mse": mse,
        "psnr": -10 * math.log10(max(mse, 1e-12)),
        "ssim": float(structural_similarity(gt.cpu().numpy(), pred.cpu().numpy(), channel_axis=2, data_range=1.0)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/root/vggt")
    from gsplat import rasterization
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda:0")
    rows = []
    for scene in SCENES:
        branch = RECON_ROOT / scene / "real8"
        rec = torch.load(branch / "vggt_reconstruction.pt", map_location="cpu")
        paths = json.loads((branch / "metrics.json").read_text())["input_paths"]
        images = [Image.open(p).convert("RGB") for p in paths]
        geoms = [geom(*image.size) for image in images]
        valid = torch.zeros((8, MODEL_CANVAS, MODEL_CANVAS), dtype=torch.bool, device=device)
        for source, g in enumerate(geoms):
            valid[source, g["pt"]:MODEL_CANVAS-g["pb"], g["pl"]:MODEL_CANVAS-g["pr"]] = True
        grid = {
            key: value[0].float().reshape(8, MODEL_CANVAS, MODEL_CANVAS, value.shape[-1]).to(device)
            for key, value in rec["gaussians"].items() if torch.is_tensor(value) and value.ndim >= 2
        }
        flat = {key: value[valid] for key, value in grid.items()}
        source_grid = torch.arange(8, device=device)[:, None, None].expand_as(valid)
        source_ids = source_grid[valid]
        ext, intr = pose_encoding_to_extri_intri(rec["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS))
        ext, intr = ext[0].to(device), intr[0].to(device)

        for slot in range(8):
            g, gt = geoms[slot], image_tensor(images[slot], device)
            K = native_k(intr[slot], g)
            view = torch.eye(4, device=device)
            view[:3, :3], view[:3, 3] = ext[slot, :3, :3], ext[slot, :3, 3]
            target = source_ids == slot
            with torch.inference_mode():
                target_out, target_alpha, _ = rasterization(
                    means=flat["xyz"][target], quats=flat["rotation"][target],
                    scales=flat["scale"][target], opacities=flat["opacity"][target, 0],
                    colors=flat["color"][target], viewmats=view[None], Ks=K[None],
                    width=g["w"], height=g["h"], near_plane=0.01, far_plane=1e10,
                    render_mode="RGB+ED",
                )
            ref_depth = target_out[0, ..., 3]
            ref_alpha = target_alpha[0, ..., 0]
            cam = flat["xyz"] @ view[:3, :3].T + view[:3, 3]
            z = cam[:, 2]
            u = K[0, 0] * cam[:, 0] / z.clamp_min(1e-8) + K[0, 2]
            v = K[1, 1] * cam[:, 1] / z.clamp_min(1e-8) + K[1, 2]
            inside = (z > 0.01) & (u >= 0) & (u < g["w"]) & (v >= 0) & (v < g["h"])
            px = u.round().long().clamp(0, g["w"] - 1)
            py = v.round().long().clamp(0, g["h"] - 1)
            sampled_depth = ref_depth[py, px]
            sampled_alpha = ref_alpha[py, px]
            rel = (z - sampled_depth) / sampled_depth.clamp_min(1e-6)

            for mode, params in MODES.items():
                reference = inside & (~target) & (sampled_alpha >= params["alpha"]) & torch.isfinite(sampled_depth)
                front = reference & (rel < -params["front"])
                back = torch.zeros_like(front)
                if params["back"] is not None:
                    back = reference & (rel > params["back"])
                weights = torch.ones_like(flat["opacity"][:, 0])
                weights[front] = params["front_weight"]
                weights[back] = params["back_weight"]
                keep = weights > 0
                with torch.inference_mode():
                    out, alpha, _ = rasterization(
                        means=flat["xyz"][keep], quats=flat["rotation"][keep],
                        scales=flat["scale"][keep],
                        opacities=flat["opacity"][keep, 0] * weights[keep],
                        colors=flat["color"][keep], viewmats=view[None], Ks=K[None],
                        width=g["w"], height=g["h"], near_plane=0.01, far_plane=1e10,
                        render_mode="RGB",
                    )
                result = metrics(gt, out[0, ..., :3].clamp(0, 1))
                rows.append({
                    "scene": scene, "slot": slot, "mode": mode,
                    "gaussians_total": int(len(source_ids)), "gaussians_kept": int(keep.sum()),
                    "front_conflicts": int(front.sum()), "back_conflicts": int(back.sum()),
                    "alpha_mean": float(alpha.mean()), **result,
                })
                print(scene, slot, mode, f"PSNR={result['psnr']:.3f}", f"removed={int((~keep).sum())}", flush=True)
                del out, alpha
                torch.cuda.empty_cache()
        del rec, grid, flat
        torch.cuda.empty_cache()

    with (OUTPUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "views": len(subset),
            "psnr_mean": float(np.mean([r["psnr"] for r in subset])),
            "ssim_mean": float(np.mean([r["ssim"] for r in subset])),
            "mse_mean": float(np.mean([r["mse"] for r in subset])),
            "alpha_mean": float(np.mean([r["alpha_mean"] for r in subset])),
            "removed_mean": float(np.mean([r["gaussians_total"] - r["gaussians_kept"] for r in subset])),
        }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
