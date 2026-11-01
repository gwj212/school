#!/usr/bin/env python3
"""Run unchanged VGGT-G on externally generated candidate views."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


TARGETS = [2, 5, 7, 10, 12, 15, 17]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--vggt-root", type=Path, default=Path("/root/vggt"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-name", default="MVGenMaster_1real7gen")
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--max-side", type=int, default=518)
    args = p.parse_args()
    sys.path.insert(0, "/root/seva_vggt/bridge")
    sys.path.insert(0, str(args.vggt_root))
    import run_pipeline as rp
    from vggt.models.vggt import VGGT

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    model.eval(); model.point_head = None; model.track_head = None
    checkpoint = rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene in args.scenes:
        payload = json.load(open(args.manifests / f"{scene}.json", encoding="utf-8"))
        branch = args.output / scene
        images_dir = branch / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        source = Path(payload["views"][0]["image"])
        source_dst = images_dir / f"view_00_real{source.suffix.lower()}"
        shutil.copy2(source, source_dst); paths.append(str(source_dst))
        for idx in TARGETS:
            source = args.candidate_root / scene / "images" / f"view_{idx:02d}_generated.png"
            dst = images_dir / f"view_{idx:02d}_generated.png"
            shutil.copy2(source, dst); paths.append(str(dst))
        torch.cuda.reset_peak_memory_stats(device)
        metrics, points = rp.reconstruct_branch(model, paths, branch, device, dtype, args.max_side, 0.1, 500_000, 23)
        baseline_ply = args.baseline_root / scene / "real8" / "point_cloud.ply"
        real_points = rp.read_pointcloud_xyz(baseline_ply)
        geometry = rp.compare_pointclouds(real_points, points, 20_000, 23, "real8", args.candidate_name)
        record = {
            "scene": scene,
            "candidate": args.candidate_name,
            "vggt_checkpoint": checkpoint,
            "reconstruction": metrics,
            "geometry_vs_real8": geometry,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "baseline_ply": str(baseline_ply),
        }
        json.dump(record, open(branch / "metrics.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        summaries.append(record)
        print(json.dumps({"scene": scene, "elapsed": metrics["elapsed_seconds"], "geometry": geometry}, ensure_ascii=False), flush=True)
        del points, real_points
    json.dump(summaries, open(args.output / "vggt_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
