#!/usr/bin/env python3
"""Run VGGT-G on saved 2-real/3-real MVGenMaster branches."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch


DEFAULT_SCENES = [
    "chair_455_64092_126267",
    "chair_581_86200_170718",
    "chair_620_101264_203062",
    "bicycle_608_95757_192662",
    "plant_271_28973_57649",
]

CONDITIONS = ("mvgen_2real6gen", "mvgen_3real5gen")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, default=Path("/root/seva_vggt/config/test30_mvgen"))
    parser.add_argument("--generation-root", type=Path, default=Path("/root/seva_vggt/outputs/mvgen_multicond5/generation"))
    parser.add_argument("--output", type=Path, default=Path("/root/seva_vggt/outputs/mvgen_multicond5/reconstruction"))
    parser.add_argument("--vggt-root", type=Path, default=Path("/root/vggt"))
    parser.add_argument("--max-side", type=int, default=518)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    args = parser.parse_args()

    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = ["/root/seva_vggt/bridge", "/root/seva_vggt/candidates", str(args.vggt_root)]
    import run_pipeline as rp
    import run_vggt_test30 as base
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    total_started = time.perf_counter()
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

    status = {
        "experiment": "held-out five-scene MVGenMaster multi-condition VGGT-G reconstruction",
        "model_load_seconds": model_load_seconds,
        "checkpoint": checkpoint,
        "scenes": scenes,
        "conditions": list(CONDITIONS),
        "completed": [],
        "failed": {},
    }
    status_path = args.output / "status.json"

    for scene in scenes:
        manifest = json.loads((args.manifests / f"{scene}.json").read_text())
        views = manifest["views"]
        for condition in CONDITIONS:
            generation_branch = args.generation_root / scene / condition
            generation_metadata = json.loads((generation_branch / "generation_metadata.json").read_text())
            condition_slots = set(generation_metadata["condition_slots"])
            paths = []
            for slot, view in enumerate(views):
                if slot in condition_slots:
                    paths.append(view["image"])
                else:
                    paths.append(str(generation_branch / "images" / f"view_{slot:02d}_generated.png"))

            branch = args.output / "scenes" / scene / condition
            branch.mkdir(parents=True, exist_ok=True)
            task = f"{scene}/{condition}"
            if base.is_complete(branch):
                status["completed"].append(task)
                status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
                continue
            started = time.perf_counter()
            try:
                preprocess_started = time.perf_counter()
                images = load_and_preprocess_images(paths, mode="pad")
                images = rp.resize_vggt_images(images, args.max_side).to(device)
                preprocess_seconds = time.perf_counter() - preprocess_started
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                inference_started = time.perf_counter()
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
                    reconstruction = model(images.unsqueeze(0))
                torch.cuda.synchronize(device)
                inference_seconds = time.perf_counter() - inference_started

                serialize_started = time.perf_counter()
                torch.save(base.cpu_tree(reconstruction), branch / "vggt_reconstruction.pt")
                serialize_seconds = time.perf_counter() - serialize_started
                export_started = time.perf_counter()
                ply = base.export_filtered(reconstruction, branch / "point_cloud_filtered_no_cap.ply")
                export_seconds = time.perf_counter() - export_started
                metrics = {
                    "scene": scene,
                    "branch": condition,
                    "condition_slots": sorted(condition_slots),
                    "generated_slots": [slot for slot in range(8) if slot not in condition_slots],
                    "checkpoint": checkpoint,
                    "input_paths": paths,
                    "input_shape": list(images.shape),
                    "canvas_hw": list(images.shape[-2:]),
                    "point_cloud_filter": {
                        "finite": True,
                        "positive_scale": True,
                        "opacity_min": base.OPACITY_MIN,
                        "max_scale_percentile": base.SCALE_PERCENTILE,
                        "point_cap": None,
                    },
                    "point_cloud": ply,
                    "timing_seconds": {
                        "input_preprocess": preprocess_seconds,
                        "model_inference": inference_seconds,
                        "serialize_reconstruction": serialize_seconds,
                        "point_cloud_export": export_seconds,
                        "warm_branch_total": time.perf_counter() - started,
                    },
                    "shared_model_load_seconds": model_load_seconds,
                    "cold_branch_total_estimate": model_load_seconds + (time.perf_counter() - started),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                }
                (branch / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
                status["completed"].append(task)
                status["failed"].pop(task, None)
                print(
                    f"COMPLETE {task} infer={inference_seconds:.3f}s "
                    f"points={ply['exported_points_no_cap']}",
                    flush=True,
                )
                del reconstruction, images
                torch.cuda.empty_cache()
            except Exception as error:
                status["failed"][task] = {"error": repr(error), "traceback": traceback.format_exc()}
                print(f"FAILED {task}: {error!r}", flush=True)
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")

    status["elapsed_seconds"] = time.perf_counter() - total_started
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
