#!/usr/bin/env python3
"""Short multi-condition MVGenMaster test on five held-out scenes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import AutoencoderKL
from easydict import EasyDict
from omegaconf import OmegaConf
from torchvision.transforms import Compose, Normalize, ToTensor


DEFAULT_SCENES = [
    "chair_455_64092_126267",
    "chair_581_86200_170718",
    "chair_620_101264_203062",
    "bicycle_608_95757_192662",
    "plant_271_28973_57649",
]

CONDITIONS = {
    "mvgen_2real6gen": [0, 4],
    "mvgen_3real5gen": [0, 3, 6],
}


def w2c(value):
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (4, 4):
        return array
    result = np.eye(4, dtype=np.float32)
    result[:3] = array
    return result


def choose_hw(path: Path, height: int, width: int):
    ratios = json.loads(path.read_text())
    return min((tuple(value) for value in ratios), key=lambda value: abs(value[0] / value[1] - height / width))


def branch_complete(output: Path, condition_slots: list[int]) -> bool:
    target_slots = sorted(set(range(8)) - set(condition_slots))
    return (output / "generation_metadata.json").is_file() and all(
        (output / "images" / f"view_{slot:02d}_{'real' if slot in condition_slots else 'generated'}.png").is_file()
        for slot in condition_slots + target_slots
    )


def infer_depth(depth_model, depth_transform, view):
    from depth_pro.utils import load_rgb

    rgb, _, _ = load_rgb(view["image"])
    tensor = depth_transform(rgb)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        depth = depth_model.infer(
            tensor,
            f_px=torch.tensor(float(view["intrinsic"][0][0]), device="cuda"),
        )["depth"].float().cpu().numpy()
    mask = np.asarray(Image.open(view["mask"]).convert("L")) > 127
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    valid = mask & np.isfinite(depth) & (depth > 0)
    if valid.sum() < 100:
        valid = np.isfinite(depth) & (depth > 0)
    return depth, float(np.median(depth[valid]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/root/seva_vggt/candidates/MVGenMaster"))
    parser.add_argument("--manifests", type=Path, default=Path("/root/seva_vggt/config/test30_mvgen"))
    parser.add_argument("--output", type=Path, default=Path("/root/seva_vggt/outputs/mvgen_multicond5/generation"))
    parser.add_argument("--base-model", type=Path, default=Path("/root/seva_vggt/candidates/MVGenMaster/check_points/sd2-community-stable-diffusion-2-1"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    args = parser.parse_args()

    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    from depth_pro.depth_pro import create_model_and_transforms
    from my_diffusers.models import UNet2DConditionModel
    from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
    from src.modules.schedulers import get_diffusion_scheduler

    total_started = time.perf_counter()
    model_started = time.perf_counter()
    model_dir = args.repo / "check_points" / "pretrained_model"
    config = EasyDict(OmegaConf.load(model_dir / "config.yaml"))
    config.pretrained_model_name_or_path = str(args.base_model.resolve())
    config.nframe = 8
    depth_model, depth_transform = create_model_and_transforms(device=torch.device("cuda"))
    depth_model.eval()
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model,
        subfolder="unet",
        rank=0,
        model_cfg=config.model_cfg,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    unet.load_state_dict(torch.load(model_dir / "ema_unet.pt", map_location="cpu"))
    vae.requires_grad_(False).to("cuda", dtype=torch.float16)
    unet.requires_grad_(False).eval().to("cuda", dtype=torch.float16)
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        args.base_model,
        vae=vae,
        unet=unet,
        scheduler=get_diffusion_scheduler(config, name="DDIM"),
        safety_checker=None,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    model_load_seconds = time.perf_counter() - model_started

    status = {
        "experiment": "held-out five-scene MVGenMaster multi-condition short test",
        "scenes": scenes,
        "conditions": CONDITIONS,
        "model_load_seconds": model_load_seconds,
        "completed": [],
        "failed": {},
    }
    status_path = args.output.parent / "generation_status.json"

    for scene in scenes:
        manifest = json.loads((args.manifests / f"{scene}.json").read_text())
        views = manifest["views"]
        for condition, condition_slots in CONDITIONS.items():
            branch = args.output / scene / condition
            task = f"{scene}/{condition}"
            if branch_complete(branch, condition_slots):
                status["completed"].append(task)
                status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
                continue
            branch_started = time.perf_counter()
            try:
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                target_slots = sorted(set(range(8)) - set(condition_slots))
                ordered_slots = condition_slots + target_slots
                ordered_views = [views[slot] for slot in ordered_slots]
                cond_num = len(condition_slots)

                all_w2cs = np.stack([w2c(view["extrinsic"]) for view in ordered_views])
                all_c2ws = np.linalg.inv(all_w2cs)
                radii = [float(np.linalg.norm(all_c2ws[index, :3, 3])) for index in range(cond_num)]

                depth_started = time.perf_counter()
                raw_depths, raw_medians, scale_candidates = [], [], []
                for index in range(cond_num):
                    depth, median = infer_depth(depth_model, depth_transform, ordered_views[index])
                    raw_depths.append(depth)
                    raw_medians.append(median)
                    scale_candidates.append(radii[index] / max(median, 1e-6))
                common_alignment_scale = float(np.median(scale_candidates))
                depth_seconds = time.perf_counter() - depth_started

                positions = all_c2ws[:, :3, 3]
                extent = float(np.max(positions.max(0) - positions.min(0)))
                pose_scale = float(config.camera_longest_side / extent) if extent > config.camera_longest_side else 1.0
                all_w2cs[:, :3, 3] *= pose_scale

                first = ImageOps.exif_transpose(Image.open(ordered_views[0]["image"])).convert("RGB")
                height, width = choose_hw(model_dir / "ratio_set.json", first.height, first.width)
                image_tensors, depth_tensors = [], []
                transform = Compose([ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])
                for index in range(cond_num):
                    source = ImageOps.exif_transpose(Image.open(ordered_views[index]["image"])).convert("RGB")
                    resized = source.resize(
                        (width, height),
                        Image.Resampling.LANCZOS if height < source.height else Image.Resampling.BICUBIC,
                    )
                    image_tensors.append(transform(resized))
                    depth = raw_depths[index] * common_alignment_scale * pose_scale
                    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
                    depth_tensors.append(torch.from_numpy(depth)[None])

                intrinsics = []
                for view in ordered_views:
                    intrinsic = np.asarray(view["intrinsic"], dtype=np.float32).copy()
                    intrinsic[0] *= width / view["width"]
                    intrinsic[1] *= height / view["height"]
                    intrinsics.append(intrinsic)

                generated_count = 8 - cond_num
                images = torch.cat([
                    torch.stack(image_tensors),
                    torch.zeros((generated_count, 3, height, width), dtype=torch.float32),
                ]).cuda()
                depths = torch.cat([
                    torch.stack(depth_tensors),
                    torch.zeros((generated_count, 1, height, width), dtype=torch.float32),
                ]).cuda()
                intrinsics_tensor = torch.from_numpy(np.stack(intrinsics)).cuda()
                extrinsics_tensor = torch.from_numpy(all_w2cs).cuda()

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                generation_started = time.perf_counter()
                local_config = copy.deepcopy(config)
                local_config.save_path = str(branch)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    result = pipeline(
                        images=images,
                        nframe=8,
                        cond_num=cond_num,
                        key_rescale=1.2,
                        height=height,
                        width=width,
                        intrinsics=intrinsics_tensor,
                        extrinsics=extrinsics_tensor,
                        num_inference_steps=args.steps,
                        guidance_scale=args.cfg,
                        output_type="np",
                        config=local_config,
                        tag=["custom"] * 8,
                        class_label=0,
                        depth=depths,
                        vae=pipeline.vae,
                        generator=torch.Generator().manual_seed(args.seed),
                    ).images
                torch.cuda.synchronize()
                generation_seconds = time.perf_counter() - generation_started

                save_started = time.perf_counter()
                image_dir = branch / "images"
                image_dir.mkdir(parents=True, exist_ok=True)
                for index, slot in enumerate(condition_slots):
                    source = ImageOps.exif_transpose(Image.open(views[slot]["image"])).convert("RGB")
                    source = source.resize((width, height), Image.Resampling.LANCZOS if height < source.height else Image.Resampling.BICUBIC)
                    source.save(image_dir / f"view_{slot:02d}_real.png")
                predictions = (np.clip(result[cond_num:], 0, 1) * 255 + 0.5).astype(np.uint8)
                for prediction, slot in zip(predictions, target_slots):
                    Image.fromarray(prediction).save(image_dir / f"view_{slot:02d}_generated.png")
                save_seconds = time.perf_counter() - save_started

                metadata = {
                    "scene_id": scene,
                    "method": "MVGenMaster exact-camera multi-condition",
                    "condition": condition,
                    "condition_slots": condition_slots,
                    "condition_original_indices": [views[slot]["original_index"] for slot in condition_slots],
                    "target_slots": target_slots,
                    "target_original_indices": [views[slot]["original_index"] for slot in target_slots],
                    "seed": args.seed,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "key_rescale": 1.2,
                    "resolution_hw": [height, width],
                    "depth_alignment": {
                        "raw_medians_m": raw_medians,
                        "camera_radii": radii,
                        "per_view_scale_candidates": scale_candidates,
                        "shared_median_alignment_scale": common_alignment_scale,
                    },
                    "pose_scale": pose_scale,
                    "timing_seconds": {
                        "depth_estimation": depth_seconds,
                        "diffusion_generation": generation_seconds,
                        "save_outputs": save_seconds,
                        "warm_branch_total": time.perf_counter() - branch_started,
                    },
                    "shared_model_load_seconds": model_load_seconds,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                }
                (branch / "generation_metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
                )
                status["completed"].append(task)
                status["failed"].pop(task, None)
                print(f"COMPLETE {task} generation={generation_seconds:.3f}s", flush=True)
                del images, depths, intrinsics_tensor, extrinsics_tensor, result
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
