#!/usr/bin/env python3
"""Diagnose and render two binary point-cloud PLY files without OpenGL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
    ]
)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header: {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Only binary_little_endian PLY is supported: {path}")
        count = int(next(line.split()[-1] for line in header if line.startswith("element vertex ")))
        data = np.fromfile(handle, dtype=PLY_DTYPE, count=count)
    xyz = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
    rgb = np.column_stack((data["red"], data["green"], data["blue"])).astype(np.uint8)
    alpha = data["alpha"].astype(np.float32) / 255.0
    return xyz, rgb, alpha


def sample(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= count:
        return points
    return points[rng.choice(len(points), count, replace=False)]


def cloud_metrics(
    xyz: np.ndarray, alpha: np.ndarray, rng: np.random.Generator
) -> dict[str, object]:
    probe = sample(xyz, 50_000, rng)
    tree = cKDTree(probe)
    distances = tree.query(probe, k=9, workers=-1)[0]
    nn = distances[:, 1]
    k8 = distances[:, -1]
    median_k8 = float(np.median(k8))
    mad_k8 = float(np.median(np.abs(k8 - median_k8)))
    outlier_limit = median_k8 + 4.0 * max(mad_k8, 1e-12)

    local_probe = sample(probe, 12_000, rng)
    neighbors = tree.query(local_probe, k=16, workers=-1)[1]
    neighborhoods = probe[neighbors]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / centered.shape[1]
    eigenvalues = np.linalg.eigvalsh(covariance)
    smallest = np.maximum(eigenvalues[:, 0], 0.0)
    curvature = smallest / np.maximum(eigenvalues.sum(axis=1), 1e-12)
    thickness = np.sqrt(smallest)

    bbox_min, bbox_max = xyz.min(axis=0), xyz.max(axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    return {
        "points": int(len(xyz)),
        "bbox_diagonal": diagonal,
        "opacity_mean": float(alpha.mean()),
        "opacity_p50": float(np.quantile(alpha, 0.5)),
        "opacity_p90": float(np.quantile(alpha, 0.9)),
        "nearest_neighbor": {
            "median": float(np.median(nn)),
            "p90": float(np.quantile(nn, 0.9)),
            "p99": float(np.quantile(nn, 0.99)),
        },
        "eighth_neighbor": {
            "median": median_k8,
            "p90": float(np.quantile(k8, 0.9)),
            "outlier_limit_median_plus_4mad": outlier_limit,
            "outlier_fraction": float(np.mean(k8 > outlier_limit)),
        },
        "local_surface": {
            "thickness_rms_mean": float(thickness.mean()),
            "thickness_rms_p90": float(np.quantile(thickness, 0.9)),
            "thickness_p90_normalized": float(np.quantile(thickness, 0.9) / diagonal),
            "curvature_mean": float(curvature.mean()),
            "curvature_p90": float(np.quantile(curvature, 0.9)),
        },
    }


def render_projection(
    xyz: np.ndarray,
    rgb: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    axes: tuple[int, int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    size: int = 640,
) -> Image.Image:
    transformed = (xyz - center) @ basis
    horizontal, vertical, depth_axis = axes
    x, y, depth = transformed[:, horizontal], transformed[:, vertical], transformed[:, depth_axis]
    x0, x1 = limits[0]
    y0, y1 = limits[1]
    px = np.clip(((x - x0) / max(x1 - x0, 1e-9) * (size - 1)).astype(np.int32), 0, size - 1)
    py = np.clip(((1.0 - (y - y0) / max(y1 - y0, 1e-9)) * (size - 1)).astype(np.int32), 0, size - 1)
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    px, py, depth, color = px[inside], py[inside], depth[inside], rgb[inside]
    flat = py * size + px
    order = np.argsort(depth)[::-1]
    flat_sorted = flat[order]
    _, first = np.unique(flat_sorted, return_index=True)
    chosen = order[first]
    canvas = np.full((size * size, 3), 248, dtype=np.uint8)
    canvas[flat[chosen]] = color[chosen]
    image = canvas.reshape(size, size, 3)
    # One-pixel expansion makes holes visible without changing geometry.
    base = Image.fromarray(image)
    return base.resize((size // 2, size // 2), Image.Resampling.LANCZOS)


def montage(
    real_xyz: np.ndarray,
    real_rgb: np.ndarray,
    generated_xyz: np.ndarray,
    generated_rgb: np.ndarray,
    output: Path,
    rng: np.random.Generator,
) -> None:
    fit = sample(real_xyz, 100_000, rng)
    center = np.median(fit, axis=0)
    covariance = np.cov((fit - center).T)
    _, basis = np.linalg.eigh(covariance)
    basis = basis[:, ::-1]
    real_t = (sample(real_xyz, 150_000, rng) - center) @ basis
    generated_t = (sample(generated_xyz, 150_000, rng) - center) @ basis
    combined = np.concatenate((real_t, generated_t), axis=0)
    pairs = [(0, 1, 2), (2, 1, 0), (0, 2, 1)]
    limits = []
    for horizontal, vertical, _ in pairs:
        x0, x1 = np.quantile(combined[:, horizontal], [0.005, 0.995])
        y0, y1 = np.quantile(combined[:, vertical], [0.005, 0.995])
        pad_x, pad_y = (x1 - x0) * 0.04, (y1 - y0) * 0.04
        limits.append(((x0 - pad_x, x1 + pad_x), (y0 - pad_y, y1 + pad_y)))

    panel = 320
    header = 36
    canvas = Image.new("RGB", (panel * 3, header * 2 + panel * 2), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["PCA front", "PCA side", "PCA top"]
    for column, (axes, limit, label) in enumerate(zip(pairs, limits, labels)):
        real_image = render_projection(real_xyz, real_rgb, center, basis, axes, limit)
        gen_image = render_projection(generated_xyz, generated_rgb, center, basis, axes, limit)
        x = column * panel
        draw.text((x + 8, 10), f"real20 - {label}", fill="black")
        canvas.paste(real_image, (x, header))
        draw.text((x + 8, header + panel + 10), f"SEVA 4+8 - {label}", fill="black")
        canvas.paste(gen_image, (x, header * 2 + panel))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("real", type=Path)
    parser.add_argument("generated", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    real_xyz, real_rgb, real_alpha = read_ply(args.real)
    generated_xyz, generated_rgb, generated_alpha = read_ply(args.generated)
    real_keep = real_alpha >= args.min_opacity
    generated_keep = generated_alpha >= args.min_opacity
    real_xyz, real_rgb, real_alpha = real_xyz[real_keep], real_rgb[real_keep], real_alpha[real_keep]
    generated_xyz, generated_rgb, generated_alpha = (
        generated_xyz[generated_keep], generated_rgb[generated_keep], generated_alpha[generated_keep]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "min_opacity": args.min_opacity,
        "real20": cloud_metrics(real_xyz, real_alpha, rng),
        "seva_generated": cloud_metrics(generated_xyz, generated_alpha, rng),
    }
    metrics_path = args.output_dir / "pointcloud_diagnostics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    montage(
        real_xyz, real_rgb, generated_xyz, generated_rgb,
        args.output_dir / "pointcloud_montage.jpg", rng,
    )
    print(json.dumps({"metrics": str(metrics_path), "montage": str(args.output_dir / "pointcloud_montage.jpg"), **metrics}, indent=2))


if __name__ == "__main__":
    main()
