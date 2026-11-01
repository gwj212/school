#!/usr/bin/env python3
"""Frozen VGGT-G reconstruction for four consistency-adapted MVGen scenes."""

from __future__ import annotations

import json
import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

SCENES = [
    "chair_620_101264_203062",
    "laptop_373_41558_82973",
    "plant_598_92218_184789",
    "toybus_121_14153_28986",
]
MANIFESTS = Path("/root/seva_vggt/config/test30_mvgen")
GENERATED = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/mvgen_consistency_adapter_test4")
OUTPUT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt/mvgen_consistency_adapter_reconstruction_test4")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def main():
    global GENERATED, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, default=GENERATED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--branch-name", default="mvgen_consistency_1real7gen")
    args = parser.parse_args()
    GENERATED = args.generated_root
    OUTPUT = args.output
    sys.path[:0] = ["/root/seva_vggt/bridge", "/root/vggt"]
    import run_pipeline as rp
    from run_vggt_head_adapter_test5 import export_filtered
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    load_started = time.perf_counter()
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    model.eval()
    model.point_head = None
    model.track_head = None
    checkpoint = rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.to(device)
    model_load_seconds = time.perf_counter() - load_started
    status = {"completed": [], "failed": {}, "model_load_seconds": model_load_seconds, "checkpoint": checkpoint}
    status_path = OUTPUT / "status.json"
    for scene in SCENES:
        branch = OUTPUT / "scenes" / scene / args.branch_name
        branch.mkdir(parents=True, exist_ok=True)
        try:
            manifest = json.loads((MANIFESTS / f"{scene}.json").read_text())
            real0 = manifest["views"][0]["image"]
            paths = [real0] + [str(GENERATED / scene / "images" / f"view_{slot:02d}_generated.png") for slot in range(1, 8)]
            started = time.perf_counter()
            inp = load_and_preprocess_images(paths, mode="pad")
            inp = rp.resize_vggt_images(inp, 518).to(device)
            preprocess_seconds = time.perf_counter() - started
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            infer_started = time.perf_counter()
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
                reconstruction = model(inp.unsqueeze(0))
            torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - infer_started
            torch.save(cpu_tree(reconstruction), branch / "vggt_reconstruction.pt")
            point_cloud = export_filtered(reconstruction, branch / "point_cloud_filtered_no_cap.ply")
            metrics = {
                "scene": scene, "branch": args.branch_name, "input_paths": paths,
                "input_shape": list(inp.shape), "checkpoint": checkpoint,
                "timing_seconds": {"input_preprocess": preprocess_seconds, "model_inference": inference_seconds,
                                   "warm_branch_total": time.perf_counter() - started},
                "model_load_seconds": model_load_seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "point_cloud": point_cloud,
            }
            (branch / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
            status["completed"].append(scene)
            print(f"COMPLETE {scene} infer={inference_seconds:.3f}s points={point_cloud['exported_points_no_cap']}", flush=True)
            del inp, reconstruction
            torch.cuda.empty_cache()
        except Exception as exc:
            status["failed"][scene] = {"error": repr(exc), "traceback": traceback.format_exc()}
            print(f"FAILED {scene}: {exc!r}", flush=True)
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
