#!/usr/bin/env python3
"""Run all seven fixed-eight ratios while loading SEVA and VGGT-G once per scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

import run_pipeline as rp


ROOT = Path("/root/seva_vggt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seva-size", type=int, default=384)
    parser.add_argument("--vggt-max-side", type=int, default=518)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--comparison-points", type=int, default=20_000)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--seva-root", type=Path, default=ROOT / "seva")
    parser.add_argument("--vggt-root", type=Path, default=Path("/root/vggt"))
    parser.add_argument("--seva-model", type=Path, default=ROOT / "models/stable-virtual-camera")
    parser.add_argument("--seva-version", type=float, default=1.1)
    parser.add_argument("--vggt-model", type=Path, default=rp.DEFAULT_VGGT_MODEL)
    parser.add_argument("--vggt-checkpoint", type=Path, default=Path("/root/vggt/ckpts/ckpt.pth"))
    parser.add_argument("--vae-model", type=Path, default=ROOT / "models/sd-vae-ft-mse")
    parser.add_argument(
        "--clip-weights",
        type=Path,
        default=ROOT / "models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
    )
    return parser.parse_args()


def metadata_path(output: Path, real_count: int) -> Path:
    return output / f"run_metadata_seva_{real_count}real_{8-real_count}gen.json"


def branch_complete(output: Path, real_count: int) -> bool:
    name = f"seva_{real_count}real_{8-real_count}gen"
    branch = output / name
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            metadata_path(output, real_count),
            output / f"comparison_real8_vs_{name}.json",
            branch / "metrics.json",
            branch / "image_metrics.json",
            branch / "point_cloud.ply",
        )
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    payload, _, _ = rp.load_manifest(args.manifest)
    rp.validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratios = [value for value in range(1, 8) if not branch_complete(args.output_dir, value)]
    if not ratios:
        print(json.dumps({"scene": payload["scene_id"], "skipped": "all ratios complete"}))
        return

    os.environ.setdefault("HF_HOME", str(args.workspace / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
    os.environ["SEVA_WRITE_VIDEO"] = "0"
    os.environ["SEVA_VAE_MODEL"] = str(args.vae_model)
    os.environ["SEVA_CLIP_MODEL"] = str(args.clip_weights)
    sys.path.insert(0, str(args.seva_root))
    sys.path.insert(0, str(args.vggt_root))

    from seva.eval import run_one_scene, set_lowvram_mode
    from seva.model import SGMWrapper
    from seva.modules.autoencoder import AutoEncoder
    from seva.modules.conditioner import CLIPConditioner
    from seva.sampling import DiscreteDenoiser
    from seva.utils import load_model as load_seva_model

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    scene_started = time.perf_counter()

    load_started = time.perf_counter()
    set_lowvram_mode(True)
    seva_model = SGMWrapper(
        load_seva_model(
            model_version=args.seva_version,
            pretrained_model_name_or_path=str(args.seva_model),
            device="cpu",
            dtype=dtype,
        ).eval()
    )
    autoencoder = AutoEncoder(chunk_size=1).eval()
    conditioner = CLIPConditioner().eval()
    denoiser = DiscreteDenoiser(num_idx=1000, device=str(device))
    seva_load_seconds = time.perf_counter() - load_started

    generated_records: dict[int, dict[str, Any]] = {}
    for real_count in ratios:
        baseline_views, inputs, targets = rp.select_fixed_total_split(payload, real_count, 8)
        name = f"seva_{real_count}real_{8-real_count}gen"
        real_dir = args.output_dir / "real8"
        branch = args.output_dir / name
        real_dir.mkdir(parents=True, exist_ok=True)
        branch.mkdir(parents=True, exist_ok=True)
        ordered_views = inputs + targets
        c2ws, intrinsics = rp.tensor_cameras(ordered_views)
        version = {
            "H": args.seva_size,
            "W": args.seva_size,
            "T": len(ordered_views),
            "C": 4,
            "f": 8,
            "options": {
                "chunk_strategy": "nearest",
                "beta_linear_start": 5e-6,
                "log_snr_shift": 2.4,
                "guider_types": 1,
                "cfg": args.cfg,
                "camera_scale": 2.0,
                "num_steps": args.steps,
                "cfg_min": 1.2,
                "encoding_t": 1,
                "decoding_t": 1,
                "save_input": False,
                "save_output": False,
                "write_video": False,
                "return_tensors": True,
            },
        }
        image_cond = {
            "img": [view["image"] for view in inputs] + [None] * len(targets),
            "input_indices": list(range(len(inputs))),
            "prior_indices": None,
        }
        camera_cond = {
            "c2w": c2ws,
            "K": intrinsics,
            "input_indices": list(range(len(ordered_views))),
        }
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        result = next(
            run_one_scene(
                "img2img",
                version,
                model=seva_model,
                ae=autoencoder,
                conditioner=conditioner,
                denoiser=denoiser,
                image_cond=image_cond,
                camera_cond=camera_cond,
                save_path=str(branch / "seva_internal"),
                use_traj_prior=False,
                traj_prior_Ks=None,
                traj_prior_c2ws=None,
                seed=args.seed,
            )
        )
        generated = next(value for key, value in result.items() if key.startswith("samples-rgb")).detach().cpu()
        torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_started
        output_started = time.perf_counter()
        generated_paths = rp.save_generated(generated, targets, branch / "generated_only")
        generated_by_index = {int(view["index"]): path for view, path in zip(targets, generated_paths)}
        real_paths = rp.save_branch_images(baseline_views, real_dir / "images")
        mixed_views = sorted(inputs + targets, key=lambda view: int(view["index"]))
        mixed_paths = rp.save_branch_images(mixed_views, branch / "images", generated_by_index)
        image_metrics = rp.evaluate_generated(payload, targets, generated_by_index)
        (branch / "image_metrics.json").write_text(json.dumps(image_metrics, indent=2) + "\n")
        output_seconds = time.perf_counter() - output_started
        record = {
            "scene_id": payload["scene_id"],
            "manifest": str(args.manifest),
            "baseline_indices": [view["index"] for view in baseline_views],
            "input_indices": [view["index"] for view in inputs],
            "target_indices": [view["index"] for view in targets],
            "generated_paths": generated_paths,
            "real_branch": str(real_dir),
            "generated_branch": str(branch),
            "seva_steps": args.steps,
            "seva_cfg": args.cfg,
            "dtype": str(dtype),
            "gpu_capability": list(capability),
            "timing_seconds": {
                "seva_model_load_shared_per_scene": seva_load_seconds,
                "seva_generation": generation_seconds,
                "generated_outputs_and_image_metrics": output_seconds,
            },
            "seva_peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        }
        metadata_path(args.output_dir, real_count).write_text(json.dumps(record, indent=2) + "\n")
        generated_records[real_count] = {"metadata": record, "real_paths": real_paths, "mixed_paths": mixed_paths, "image": image_metrics}
        del result, generated
        torch.cuda.empty_cache()
        print(f"GENERATED {payload['scene_id']} {real_count}+{8-real_count}", flush=True)

    del seva_model, autoencoder, conditioner, denoiser
    torch.cuda.empty_cache()

    from vggt.models.vggt import VGGT

    load_started = time.perf_counter()
    model = VGGT.from_pretrained(str(args.vggt_model), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    model.eval()
    model.point_head = None
    model.track_head = None
    checkpoint_metadata = rp.load_reconstruction_checkpoint(model, args.vggt_checkpoint)
    model.to(device)
    torch.cuda.synchronize(device)
    vggt_load_seconds = time.perf_counter() - load_started

    baseline_views, _, _ = rp.select_fixed_total_split(payload, 1, 8)
    real_dir = args.output_dir / "real8"
    real_paths = rp.save_branch_images(baseline_views, real_dir / "images")
    real_metrics_path = real_dir / "metrics.json"
    if real_metrics_path.exists() and (real_dir / "point_cloud.ply").exists():
        real_metrics = json.loads(real_metrics_path.read_text())
        real_reconstruction = real_metrics["reconstruction"]
        real_points = rp.read_pointcloud_xyz(real_dir / "point_cloud.ply")
    else:
        real_reconstruction, real_points = rp.reconstruct_branch(
            model, real_paths, real_dir, device, dtype, args.vggt_max_side,
            args.opacity_threshold, args.max_points, args.seed,
        )
        real_metrics = {
            "branch": "real8",
            "description": "VGGT-G reconstruction from 8 real images.",
            "vggt_checkpoint": checkpoint_metadata,
            "reconstruction": real_reconstruction,
        }
        real_metrics_path.write_text(json.dumps(real_metrics, indent=2) + "\n")

    for real_count in ratios:
        name = f"seva_{real_count}real_{8-real_count}gen"
        branch = args.output_dir / name
        record = generated_records[real_count]
        metadata = record["metadata"]
        torch.cuda.reset_peak_memory_stats(device)
        mixed_reconstruction, mixed_points = rp.reconstruct_branch(
            model, record["mixed_paths"], branch, device, dtype, args.vggt_max_side,
            args.opacity_threshold, args.max_points, args.seed,
        )
        generated_metrics = {
            "branch": name,
            "description": f"VGGT-G reconstruction from {real_count} real inputs and {8-real_count} SEVA-generated views.",
            "vggt_checkpoint": checkpoint_metadata,
            "image_generation": record["image"],
            "reconstruction": mixed_reconstruction,
        }
        (branch / "metrics.json").write_text(json.dumps(generated_metrics, indent=2) + "\n")
        compare_started = time.perf_counter()
        comparison = rp.compare_pointclouds(
            real_points, mixed_points, args.comparison_points, args.seed, "real8", name
        )
        comparison["real_metrics"] = str(real_metrics_path)
        comparison["generated_metrics"] = str(branch / "metrics.json")
        comparison["note"] = "The real8 reconstruction is a paired pseudo-reference, not ground-truth surface geometry."
        comparison_path = args.output_dir / f"comparison_real8_vs_{name}.json"
        comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
        comparison_seconds = time.perf_counter() - compare_started

        timings = metadata["timing_seconds"]
        timings.update(
            {
                "vggt_model_load_shared_per_scene": vggt_load_seconds,
                "real_baseline_reconstruction": real_reconstruction["elapsed_seconds"],
                "mixed_vggt_reconstruction": mixed_reconstruction["elapsed_seconds"],
                "geometry_comparison": comparison_seconds,
            }
        )
        timings["warm_end_to_end"] = (
            timings["seva_generation"]
            + timings["generated_outputs_and_image_metrics"]
            + timings["mixed_vggt_reconstruction"]
        )
        timings["standalone_cold_end_to_end_estimate"] = (
            timings["warm_end_to_end"] + seva_load_seconds + vggt_load_seconds
        )
        timings["seven_ratio_amortized_end_to_end"] = (
            timings["warm_end_to_end"] + (seva_load_seconds + vggt_load_seconds) / 7.0
        )
        metadata.update(
            {
                "vggt_checkpoint": checkpoint_metadata,
                "real_baseline_reused": real_metrics_path.exists(),
                "comparison_metrics": str(comparison_path),
                "real_reconstruction": real_reconstruction,
                "generated_reconstruction": mixed_reconstruction,
                "vggt_peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_cuda_memory_mb": max(metadata["seva_peak_cuda_memory_mb"], torch.cuda.max_memory_allocated(device) / 2**20),
            }
        )
        metadata_path(args.output_dir, real_count).write_text(json.dumps(metadata, indent=2) + "\n")
        del mixed_points
        print(f"RECONSTRUCTED {payload['scene_id']} {real_count}+{8-real_count}", flush=True)

    del model, real_points
    torch.cuda.empty_cache()
    scene_summary = {
        "scene_id": payload["scene_id"],
        "ratios_run": ratios,
        "seva_model_load_seconds": seva_load_seconds,
        "vggt_model_load_seconds": vggt_load_seconds,
        "scene_wall_seconds": time.perf_counter() - scene_started,
    }
    (args.output_dir / "shared_model_scene_timing.json").write_text(json.dumps(scene_summary, indent=2) + "\n")
    print(json.dumps(scene_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
