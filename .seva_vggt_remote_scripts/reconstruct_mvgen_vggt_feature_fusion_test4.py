#!/usr/bin/env python3
"""Run the frozen-backbone MVGen->VGGT feature-fusion pilot on four held-out categories."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

SCENES = ["chair_620_101264_203062", "laptop_373_41558_82973",
          "plant_598_92218_184789", "toybus_121_14153_28986"]


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(cpu_tree(v) for v in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = ["/root/seva_vggt/bridge", "/root/vggt"]
    from run_vggt_head_adapter_test5 import export_filtered
    import run_pipeline as rp
    from train_mvgen_vggt_feature_fusion import MVGenVGGTFeatureFusion
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"): model.gaussian_head.enable_color_head_after_init()
    model.point_head = None; model.track_head = None
    checkpoint = rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.requires_grad_(False).to(device).eval()
    fusion = MVGenVGGTFeatureFusion().to(device).eval()
    fusion.load_state_dict(torch.load(args.adapter, map_location=device)["state_dict"])
    status = {"completed": [], "failed": {}, "checkpoint": checkpoint, "adapter": str(args.adapter)}
    for scene in SCENES:
        branch = args.output / "scenes" / scene / "mvgen_feature_fusion_1real7gen"
        branch.mkdir(parents=True, exist_ok=True)
        try:
            manifest = json.loads((args.manifests / f"{scene}.json").read_text())
            paths = [manifest["views"][0]["image"]] + [str(args.generated / scene / "images" / f"view_{i:02d}_generated.png") for i in range(1, 8)]
            inp = load_and_preprocess_images(paths, mode="pad").to(device)
            fusion.current_latents = torch.load(args.latents / f"{scene}.pt", map_location=device)["latents"][None]
            handle = model.dpt_feature_head.register_forward_hook(fusion.hook)
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype): reconstruction = model(inp[None])
            torch.cuda.synchronize(device); elapsed = time.perf_counter() - started; handle.remove()
            torch.save(cpu_tree(reconstruction), branch / "vggt_reconstruction.pt")
            point_cloud = export_filtered(reconstruction, branch / "point_cloud_filtered_no_cap.ply")
            metrics = {"scene": scene, "input_paths": paths, "model_inference_seconds": elapsed,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                       "point_cloud": point_cloud, "adapter": str(args.adapter)}
            (branch / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
            status["completed"].append(scene); print(f"COMPLETE {scene} {elapsed:.3f}s", flush=True)
            del reconstruction, inp; torch.cuda.empty_cache()
        except Exception as exc:
            status["failed"][scene] = {"error": repr(exc), "traceback": traceback.format_exc()}
        (args.output / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    if status["failed"]: raise SystemExit(1)


if __name__ == "__main__": main()
