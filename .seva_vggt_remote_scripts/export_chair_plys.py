#!/usr/bin/env python3
"""Export lossless point/Gaussian PLY variants from a saved VGGT reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


C0 = 0.28209479177387814


def flatten_gaussians(reconstruction: dict) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key in ("xyz", "color", "opacity", "scale", "rotation"):
        value = reconstruction["gaussians"][key]
        if value.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for {key}, got {tuple(value.shape)}")
        result[key] = (
            value[0]
            .detach()
            .float()
            .cpu()
            .reshape(-1, value.shape[-1])
            .numpy()
        )
    return result


def finite_mask(g: dict[str, np.ndarray]) -> np.ndarray:
    mask = np.ones(len(g["xyz"]), dtype=bool)
    for value in g.values():
        mask &= np.isfinite(value).all(axis=1)
    return mask


def write_center_ply(path: Path, g: dict[str, np.ndarray], indices: np.ndarray) -> None:
    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"comment VGGT Gaussian centers; no random downsampling\n"
            f"element vertex {len(indices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property uchar alpha\nend_header\n"
        ).encode("ascii")
        handle.write(header)
        chunk = 250_000
        for start in range(0, len(indices), chunk):
            idx = indices[start : start + chunk]
            out = np.empty(len(idx), dtype=dtype)
            out["x"], out["y"], out["z"] = g["xyz"][idx].T
            rgb = (np.clip(g["color"][idx], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            out["red"], out["green"], out["blue"] = rgb.T
            out["alpha"] = (
                np.clip(g["opacity"][idx, 0], 0.0, 1.0) * 255.0 + 0.5
            ).astype(np.uint8)
            out.tofile(handle)


def write_standard_3dgs_ply(path: Path, g: dict[str, np.ndarray], indices: np.ndarray) -> None:
    # Standard 3D Gaussian Splatting field names. VGGT stores activated RGB,
    # opacity and scale, so convert them back to SH-DC, logit and log domains.
    float_fields = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    dtype = np.dtype([(name, "<f4") for name in float_fields])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        header_lines = [
            "ply", "format binary_little_endian 1.0",
            "comment VGGT full Gaussian parameters converted to standard 3DGS domains",
            f"element vertex {len(indices)}",
        ]
        header_lines += [f"property float {name}" for name in float_fields]
        header_lines += ["end_header"]
        handle.write(("\n".join(header_lines) + "\n").encode("ascii"))
        chunk = 150_000
        eps = 1e-6
        for start in range(0, len(indices), chunk):
            idx = indices[start : start + chunk]
            out = np.empty(len(idx), dtype=dtype)
            xyz = g["xyz"][idx]
            out["x"], out["y"], out["z"] = xyz.T
            out["nx"] = 0.0
            out["ny"] = 0.0
            out["nz"] = 0.0
            fdc = (np.clip(g["color"][idx], 0.0, 1.0) - 0.5) / C0
            out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = fdc.T
            alpha = np.clip(g["opacity"][idx, 0], eps, 1.0 - eps)
            out["opacity"] = np.log(alpha / (1.0 - alpha))
            log_scale = np.log(np.clip(g["scale"][idx], eps, None))
            out["scale_0"], out["scale_1"], out["scale_2"] = log_scale.T
            rotation = g["rotation"][idx]
            out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = rotation.T
            out.tofile(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    reconstruction = torch.load(args.reconstruction, map_location="cpu")
    g = flatten_gaussians(reconstruction)
    finite = finite_mask(g)
    all_indices = np.flatnonzero(finite)
    threshold_outputs = []
    for order, threshold in ((1, 0.10), (2, 0.02)):
        indices = np.flatnonzero(finite & (g["opacity"][:, 0] >= threshold))
        threshold_tag = str(threshold).replace(".", "p")
        name = (
            f"{args.prefix}__0{order}_centers_opacity_ge_{threshold_tag}"
            f"__no_downsample__{len(indices)}pts.ply"
        )
        path = args.output_dir / name
        write_center_ply(path, g, indices)
        threshold_outputs.append((threshold, path, len(indices)))

    full_path = args.output_dir / (
        f"{args.prefix}__03_full_gaussians_all_params_standard3dgs"
        f"__{len(all_indices)}gaussians.ply"
    )
    write_standard_3dgs_ply(full_path, g, all_indices)

    outputs = []
    for threshold, path, count in threshold_outputs:
        outputs.append(
            {
                "type": "center_point_cloud",
                "opacity_threshold": threshold,
                "random_downsampling": False,
                "points": count,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    outputs.append(
        {
            "type": "standard_3dgs_full_parameters",
            "opacity_threshold": None,
            "random_downsampling": False,
            "points": len(all_indices),
            "fields": [
                "xyz", "SH-DC color", "opacity logit", "log scale", "quaternion rotation"
            ],
            "path": str(full_path),
            "bytes": full_path.stat().st_size,
            "sha256": sha256(full_path),
        }
    )
    manifest = {
        "source": str(args.reconstruction),
        "raw_gaussians": len(g["xyz"]),
        "finite_gaussians": int(finite.sum()),
        "outputs": outputs,
        "notes": {
            "01": "MeshLab-friendly Gaussian centers, opacity >= 0.10, no 500k cap.",
            "02": "MeshLab-friendly Gaussian centers, opacity >= 0.02, no 500k cap.",
            "03": "Full Gaussian attributes for a standard 3DGS viewer; MeshLab only shows centers.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.prefix}__export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
