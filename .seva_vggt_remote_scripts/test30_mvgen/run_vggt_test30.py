#!/usr/bin/env python3
"""Run one shared VGGT-G model on real8 and MVGen 1+7 branches, with no PLY point cap."""

from __future__ import annotations

import argparse, json, math, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

OPACITY_MIN = 0.10
SCALE_PERCENTILE = 99.9


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(cpu_tree(v) for v in value)
    return value


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v = np.empty(len(xyz), dtype=dtype)
    v["x"], v["y"], v["z"] = xyz.T; v["red"], v["green"], v["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"comment no point cap; finite; positive scale; opacity >= {OPACITY_MIN}; max scale <= scene p{SCALE_PERCENTILE}\n"
              f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii")
    with path.open("wb") as f: f.write(header); v.tofile(f)


def export_filtered(reconstruction, path: Path):
    g = reconstruction["gaussians"]
    xyz = g["xyz"][0].reshape(-1, 3).float().cpu().numpy()
    color = g["color"][0].reshape(-1, 3).float().cpu().numpy()
    opacity = g["opacity"][0].reshape(-1).float().cpu().numpy()
    scale = g["scale"][0].reshape(-1, 3).float().cpu().numpy()
    finite = np.isfinite(xyz).all(1) & np.isfinite(color).all(1) & np.isfinite(opacity) & np.isfinite(scale).all(1)
    positive = (scale > 0).all(1); opacity_pass = opacity >= OPACITY_MIN
    pre = finite & positive & opacity_pass; scale_max = scale.max(1)
    limit = float(np.percentile(scale_max[pre], SCALE_PERCENTILE))
    keep = pre & (scale_max <= limit)
    write_ply(path, xyz[keep].astype("<f4", copy=False), (np.clip(color[keep], 0, 1) * 255 + .5).astype(np.uint8))
    return {"raw_gaussians": int(len(xyz)), "finite_positive_opacity_pass": int(pre.sum()),
            "scale_p99p9_limit": limit, "exported_points_no_cap": int(keep.sum()),
            "removed_nonfinite_or_nonpositive": int((~(finite & positive)).sum()),
            "removed_low_opacity": int((finite & positive & ~opacity_pass).sum()),
            "removed_scale_outliers": int((pre & (scale_max > limit)).sum()),
            "point_cloud": str(path), "file_bytes": path.stat().st_size}


def is_complete(branch: Path):
    return all((branch / p).exists() for p in ("vggt_reconstruction.pt", "point_cloud_filtered_no_cap.ply", "metrics.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--mvgen-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vggt-root", type=Path, default=Path("/root/vggt"))
    ap.add_argument("--max-side", type=int, default=518)
    args = ap.parse_args()
    sys.path[:0] = ["/root/seva_vggt/bridge", str(args.vggt_root)]
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    device = torch.device("cuda:0"); dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    total_started = time.perf_counter(); load_started = time.perf_counter()
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"): model.gaussian_head.enable_color_head_after_init()
    model.eval(); model.point_head = None; model.track_head = None
    checkpoint = rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.to(device); model_load_seconds = time.perf_counter() - load_started
    protocol = json.loads((args.manifests / "selection_protocol.json").read_text()); scenes = protocol["scenes"]
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    status = {"model_load_seconds": model_load_seconds, "checkpoint": checkpoint, "completed": [], "failed": {}, "scenes": scenes}
    for scene in scenes:
        manifest = json.loads((args.manifests / f"{scene}.json").read_text())
        real_paths = [v["image"] for v in manifest["views"]]
        mv_paths = [real_paths[0]] + [str(args.mvgen_root / scene / "images" / f"view_{i:02d}_generated.png") for i in range(1, 8)]
        for branch_name, paths in (("real8", real_paths), ("mvgen_1real7gen", mv_paths)):
            branch = args.output / "scenes" / scene / branch_name; branch.mkdir(parents=True, exist_ok=True)
            task = f"{scene}/{branch_name}"
            if is_complete(branch): status["completed"].append(task); continue
            started = time.perf_counter()
            try:
                pre_started = time.perf_counter()
                inp = load_and_preprocess_images(paths, mode="pad")
                inp = rp.resize_vggt_images(inp, args.max_side).to(device)
                preprocess_seconds = time.perf_counter() - pre_started
                torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
                infer_started = time.perf_counter()
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype): recon = model(inp.unsqueeze(0))
                torch.cuda.synchronize(device); inference_seconds = time.perf_counter() - infer_started
                save_started = time.perf_counter(); torch.save(cpu_tree(recon), branch / "vggt_reconstruction.pt")
                serialize_seconds = time.perf_counter() - save_started
                export_started = time.perf_counter(); ply = export_filtered(recon, branch / "point_cloud_filtered_no_cap.ply")
                export_seconds = time.perf_counter() - export_started
                metrics = {"scene": scene, "branch": branch_name, "checkpoint": checkpoint,
                    "input_paths": paths, "input_shape": list(inp.shape), "canvas_hw": list(inp.shape[-2:]),
                    "point_cloud_filter": {"finite": True, "positive_scale": True, "opacity_min": OPACITY_MIN,
                        "max_scale_percentile": SCALE_PERCENTILE, "point_cap": None}, "point_cloud": ply,
                    "timing_seconds": {"input_preprocess": preprocess_seconds, "model_inference": inference_seconds,
                        "serialize_reconstruction": serialize_seconds, "point_cloud_export": export_seconds,
                        "warm_branch_total": time.perf_counter() - started},
                    "shared_model_load_seconds": model_load_seconds,
                    "cold_branch_total_estimate": model_load_seconds + (time.perf_counter() - started),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3}
                (branch / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
                status["completed"].append(task); status["failed"].pop(task, None)
                print(f"COMPLETE {task} infer={inference_seconds:.3f}s total={metrics['timing_seconds']['warm_branch_total']:.3f}s points={ply['exported_points_no_cap']}", flush=True)
                del recon, inp; torch.cuda.empty_cache()
            except Exception as exc:
                status["failed"][task] = {"error": repr(exc), "traceback": traceback.format_exc()}
                print(f"FAILED {task}: {exc!r}", flush=True)
            status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    status["elapsed_seconds"] = time.perf_counter() - total_started
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    if status["failed"]: raise SystemExit(1)


if __name__ == "__main__": main()
