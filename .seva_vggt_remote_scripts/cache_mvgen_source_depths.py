#!/usr/bin/env python3
"""Cache DepthPro source-view conditioning maps for Stage-2B (train/validation only)."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path("/root/seva_vggt")
MVGEN = ROOT / "candidates/MVGenMaster"


def pad_w2c(value):
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (4, 4):
        return array
    result = np.eye(4, dtype=np.float32)
    result[:3, :4] = array
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.manifest_root = args.manifest_root.resolve()
    args.cache_root = args.cache_root.resolve()

    os.chdir(MVGEN)
    sys.path.insert(0, str(MVGEN))
    from depth_pro.depth_pro import create_model_and_transforms
    from depth_pro.utils import load_rgb

    protocol = json.loads(args.protocol.read_text())
    jobs = []
    for split in args.splits:
        for scene in protocol[f"{split}_scenes"]:
            jobs.append((split, scene))

    device = torch.device("cuda:0")
    model, transform = create_model_and_transforms(device=device)
    model.eval()
    written = skipped = 0
    for index, (split, scene) in enumerate(jobs, 1):
        destination = args.cache_root / scene / "mvgen_source_depth.npy"
        if destination.is_file():
            skipped += 1
            print(f"SOURCE_DEPTH {index}/{len(jobs)} skip {split}/{scene}", flush=True)
            continue
        manifest = json.loads((args.manifest_root / split / f"{scene}.json").read_text())
        views = manifest["views"][:8]
        source = views[0]
        w2cs = np.stack([pad_w2c(view["extrinsic"]) for view in views])
        radius = float(np.linalg.norm(np.linalg.inv(w2cs)[0, :3, 3]))
        rgb, _, _ = load_rgb(source["image"])
        image = transform(rgb)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = model.infer(
                image,
                f_px=torch.tensor(float(source["intrinsic"][0][0]), device=device),
            )["depth"]
        depth = prediction.float().cpu().numpy()
        mask = np.asarray(Image.open(source["mask"]).convert("L")) > 127
        if mask.shape != depth.shape:
            mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        valid = mask & np.isfinite(depth) & (depth > 0)
        if valid.sum() < 100:
            valid = np.isfinite(depth) & (depth > 0)
        depth *= radius / max(float(np.median(depth[valid])), 1e-6)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, depth.astype(np.float16))
        written += 1
        print(f"SOURCE_DEPTH {index}/{len(jobs)} write {split}/{scene}", flush=True)

    summary = {"total": len(jobs), "written": written, "skipped": skipped, "blind_accessed": False}
    (args.cache_root / "mvgen_source_depth_cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
