#!/usr/bin/env python3
"""Export all qualified MVGenMaster->VGGT-G Gaussian centers without a point cap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/root/seva_vggt/outputs/model_strategy_test5")
OUT = ROOT / "point_clouds_unlimited"
SCENES = [
    "chair_386_46082_92005",
    "car_599_92296_183295",
    "couch_349_36574_67726",
    "toybus_517_74123_144462",
    "ball_113_13373_24790",
]
OPACITY_MIN = 0.10
SCALE_PERCENTILE = 99.9


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.empty(len(xyz), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment no point cap; opacity >= {OPACITY_MIN}; max scale <= scene p{SCALE_PERCENTILE}\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene in SCENES:
        reconstruction = torch.load(ROOT / "mvgenmaster_vggt" / scene / "vggt_reconstruction.pt", map_location="cpu")
        g = reconstruction["gaussians"]
        xyz = g["xyz"][0].reshape(-1, 3).float().numpy()
        color = g["color"][0].reshape(-1, 3).float().numpy()
        opacity = g["opacity"][0].reshape(-1).float().numpy()
        scale = g["scale"][0].reshape(-1, 3).float().numpy()
        finite = (
            np.isfinite(xyz).all(axis=1) & np.isfinite(color).all(axis=1) &
            np.isfinite(opacity) & np.isfinite(scale).all(axis=1)
        )
        positive_scale = (scale > 0).all(axis=1)
        opacity_pass = opacity >= OPACITY_MIN
        pre_scale = finite & positive_scale & opacity_pass
        scale_max = scale.max(axis=1)
        scale_limit = float(np.percentile(scale_max[pre_scale], SCALE_PERCENTILE))
        keep = pre_scale & (scale_max <= scale_limit)
        xyz_out = xyz[keep].astype("<f4", copy=False)
        rgb_out = (np.clip(color[keep], 0, 1) * 255 + 0.5).astype(np.uint8)
        filename = f"{scene}_MVGenMaster_1real7gen_VGGTG_XYZRGB_opacity0p10_scaleP99p9_noCap.ply"
        path = OUT / filename
        write_binary_ply(path, xyz_out, rgb_out)
        summary = {
            "scene": scene,
            "raw_gaussians": int(len(xyz)),
            "finite_positive_opacity_pass": int(pre_scale.sum()),
            "scale_p99p9_limit": scale_limit,
            "exported_points_no_cap": int(keep.sum()),
            "removed_nonfinite_or_nonpositive": int((~(finite & positive_scale)).sum()),
            "removed_low_opacity": int((finite & positive_scale & ~opacity_pass).sum()),
            "removed_scale_outliers": int((pre_scale & (scale_max > scale_limit)).sum()),
            "file": str(path),
            "file_bytes": int(path.stat().st_size),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (OUT / "export_summary.json").write_text(json.dumps({
        "protocol": "mvgen_vggtg_no_point_cap_v1",
        "filter": {
            "finite_xyz_rgb_opacity_scale": True,
            "positive_scale": True,
            "opacity_min": OPACITY_MIN,
            "max_scale_percentile": SCALE_PERCENTILE,
            "point_cap": None,
        },
        "scenes": summaries,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
