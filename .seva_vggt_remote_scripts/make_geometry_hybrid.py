#!/usr/bin/env python3
"""Composite deterministic source-view warps over candidate NVS images."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TARGETS = [2, 5, 7, 10, 12, 15, 17]


def pad(extrinsic):
    arr = np.asarray(extrinsic, dtype=np.float32)
    out = np.eye(4, dtype=np.float32)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def scaled_k(view, w, h):
    k = np.asarray(view["intrinsic"], dtype=np.float32).copy()
    k[0] *= w / view["width"]
    k[1] *= h / view["height"]
    return k


def forward_warp(rgb, depth, source_k, source_w2c, target_k, target_w2c):
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    valid = np.isfinite(depth) & (depth > 1e-6)
    pixels = np.stack([xx[valid], yy[valid], np.ones(valid.sum())], axis=0).astype(np.float32)
    cam = np.linalg.inv(source_k) @ pixels * depth[valid][None]
    cam_h = np.concatenate([cam, np.ones((1, cam.shape[1]), dtype=np.float32)], axis=0)
    world = np.linalg.inv(source_w2c) @ cam_h
    target = target_w2c @ world
    z = target[2]
    uv = target_k @ target[:3]
    u = uv[0] / np.maximum(uv[2], 1e-8)
    v = uv[1] / np.maximum(uv[2], 1e-8)
    colors = rgb[valid]
    out = np.zeros_like(rgb)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    coverage = np.zeros((h, w), dtype=np.uint8)
    order = np.argsort(z)[::-1]  # far to near, so near samples overwrite
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ui = np.floor(u + du).astype(np.int32)
        vi = np.floor(v + dv).astype(np.int32)
        good = (z > 1e-5) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        for j in order[good[order]]:
            if z[j] <= zbuf[vi[j], ui[j]]:
                zbuf[vi[j], ui[j]] = z[j]
                out[vi[j], ui[j]] = colors[j]
                coverage[vi[j], ui[j]] = 1
    return out, coverage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    args = p.parse_args()
    for scene in args.scenes:
        manifest = json.load(open(args.manifests / f"{scene}.json", encoding="utf-8"))
        meta = json.load(open(args.candidate_root / scene / "generation_metadata.json", encoding="utf-8"))
        source = manifest["views"][0]
        h, w = meta["resolution_hw"]
        rgb = np.asarray(Image.open(source["image"]).convert("RGB").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        depth = np.load(args.candidate_root / scene / "source_depth_aligned.npy")
        source_k = scaled_k(source, w, h)
        source_w2c = pad(source["extrinsic"]); source_w2c[:3, 3] *= meta["pose_scale"]
        out_dir = args.output / scene / "images"; out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).save(out_dir / "view_00_real.png")
        coverages = []
        for idx in TARGETS:
            view = manifest["views"][idx]
            target_k = scaled_k(view, w, h)
            target_w2c = pad(view["extrinsic"]); target_w2c[:3, 3] *= meta["pose_scale"]
            warp, coverage = forward_warp(rgb, depth, source_k, source_w2c, target_k, target_w2c)
            generated = np.asarray(Image.open(args.candidate_root / scene / "images" / f"view_{idx:02d}_generated.png").convert("RGB"), dtype=np.float32) / 255.0
            distance = cv2.distanceTransform(coverage, cv2.DIST_L2, 3)
            alpha = np.clip(distance / 3.0, 0.0, 1.0)[..., None]
            hybrid = warp * alpha + generated * (1.0 - alpha)
            Image.fromarray((np.clip(hybrid, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out_dir / f"view_{idx:02d}_generated.png")
            Image.fromarray(coverage * 255).save(out_dir / f"view_{idx:02d}_warp_coverage.png")
            coverages.append(float(coverage.mean()))
        json.dump({"scene": scene, "method": "source-depth exact-camera warp over MVGenMaster; no target GT used", "mean_coverage": float(np.mean(coverages)), "per_view_coverage": coverages}, open(args.output / scene / "hybrid_metadata.json", "w", encoding="utf-8"), indent=2)
        print(scene, float(np.mean(coverages)))


if __name__ == "__main__":
    main()
