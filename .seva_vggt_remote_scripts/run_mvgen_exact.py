#!/usr/bin/env python3
"""Run MVGenMaster on exact CO3D cameras without changing upstream code."""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from easydict import EasyDict
from omegaconf import OmegaConf
from torchvision.transforms import Compose, Normalize, ToTensor


def pad_w2c(extrinsic):
    arr = np.asarray(extrinsic, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    out = np.eye(4, dtype=np.float32)
    out[:3, :4] = arr
    return out


def choose_hw(ratio_file, height, width):
    ratio_set = json.load(open(ratio_file, "r", encoding="utf-8"))
    ratio_dict = {h / w: (h, w) for h, w in ratio_set}
    ratio = height / width
    return ratio_dict[min(ratio_dict, key=lambda value: abs(value - ratio))]


def infer_source_depth(repo, image_path, focal_px, object_mask_path, camera_radius):
    from depth_pro.depth_pro import create_model_and_transforms
    from depth_pro.utils import load_rgb

    model, transform = create_model_and_transforms(device=torch.device("cuda"))
    model.eval()
    rgb, _, _ = load_rgb(image_path)
    inp = transform(rgb)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        prediction = model.infer(inp, f_px=torch.tensor(float(focal_px), device="cuda"))
    depth = prediction["depth"].float().cpu().numpy()
    del model, prediction, inp
    torch.cuda.empty_cache()

    mask = np.asarray(Image.open(object_mask_path).convert("L")) > 127
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    valid = mask & np.isfinite(depth) & (depth > 0)
    if valid.sum() < 100:
        valid = np.isfinite(depth) & (depth > 0)
    raw_median = float(np.median(depth[valid]))
    scale = float(camera_radius / max(raw_median, 1e-6))
    return depth * scale, {"raw_median_m": raw_median, "camera_radius": float(camera_radius), "alignment_scale": scale}


def main():
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--target-indices", default="2,5,7,10,12,15,17")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=2.0)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from my_diffusers.models import UNet2DConditionModel
    from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
    from src.modules.schedulers import get_diffusion_scheduler

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    target_indices = [int(x) for x in args.target_indices.split(",") if x]
    manifest = json.load(open(args.manifest, "r", encoding="utf-8"))
    views = manifest["views"]
    source = views[0]
    targets = [views[i] for i in target_indices]
    output = Path(args.output).resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    model_dir = repo / "check_points" / "pretrained_model"
    config = EasyDict(OmegaConf.load(model_dir / "config.yaml"))
    config.pretrained_model_name_or_path = str(Path(args.base_model).resolve())
    config.nframe = 1 + len(targets)
    config.save_path = str(output)

    source_w2c = pad_w2c(source["extrinsic"])
    target_w2cs = [pad_w2c(v["extrinsic"]) for v in targets]
    all_w2cs = np.stack([source_w2c] + target_w2cs)
    all_c2ws = np.linalg.inv(all_w2cs)
    source_radius = float(np.linalg.norm(all_c2ws[0, :3, 3]))
    depth, depth_meta = infer_source_depth(repo, source["image"], source["intrinsic"][0][0], source["mask"], source_radius)

    cam_positions = all_c2ws[:, :3, 3]
    max_size = float(np.max(cam_positions.max(axis=0) - cam_positions.min(axis=0)))
    pose_scale = float(config.camera_longest_side / max_size) if max_size > config.camera_longest_side else 1.0
    all_w2cs[:, :3, 3] *= pose_scale
    depth *= pose_scale

    orig = Image.open(source["image"]).convert("RGB")
    orig_w, orig_h = orig.size
    h, w = choose_hw(model_dir / "ratio_set.json", orig_h, orig_w)
    resized = orig.resize((w, h), Image.Resampling.LANCZOS if h < orig_h else Image.Resampling.BICUBIC)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    image_tensor = Compose([ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])(resized)
    source_k = np.asarray(source["intrinsic"], dtype=np.float32).copy()
    source_k[0, :] *= w / source["width"]
    source_k[1, :] *= h / source["height"]
    target_ks = []
    for v in targets:
        k = np.asarray(v["intrinsic"], dtype=np.float32).copy()
        k[0, :] *= w / v["width"]
        k[1, :] *= h / v["height"]
        target_ks.append(k)

    images = torch.cat([image_tensor[None], torch.zeros((len(targets), 3, h, w))], dim=0).cuda()
    intrinsics = torch.from_numpy(np.stack([source_k] + target_ks)).cuda()
    extrinsics = torch.from_numpy(all_w2cs).cuda()
    depths = torch.cat([torch.from_numpy(depth)[None, None], torch.zeros((len(targets), 1, h, w))], dim=0).cuda()

    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae", local_files_only=True)
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        rank=0,
        model_cfg=config.model_cfg,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    weights = torch.load(model_dir / "ema_unet.pt", map_location="cpu")
    unet.load_state_dict(weights)
    unet.requires_grad_(False).eval()
    vae.to("cuda", dtype=torch.float16)
    unet.to("cuda", dtype=torch.float16)
    scheduler = get_diffusion_scheduler(config, name="DDIM")
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")

    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(args.seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        result = pipeline(
            images=images,
            nframe=1 + len(targets),
            cond_num=1,
            key_rescale=None,
            height=h,
            width=w,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            output_type="np",
            config=copy.deepcopy(config),
            tag=["custom"] * (1 + len(targets)),
            class_label=0,
            depth=depths,
            vae=pipeline.vae,
            generator=generator,
        ).images
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    resized.save(images_dir / "view_00_real.png")
    preds = (np.clip(result[1:], 0, 1) * 255).astype(np.uint8)
    for pred, idx in zip(preds, target_indices):
        Image.fromarray(pred).save(images_dir / f"view_{idx:02d}_generated.png")
    np.save(output / "source_depth_aligned.npy", depth.astype(np.float32))
    metadata = {
        "scene_id": manifest["scene_id"],
        "method": "MVGenMaster exact-camera single-source",
        "source_manifest_index": 0,
        "target_manifest_indices": target_indices,
        "target_original_indices": [v["original_index"] for v in targets],
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "resolution_hw": [h, w],
        "depth_alignment": depth_meta,
        "pose_scale": pose_scale,
        "generation_seconds": elapsed,
        "cold_process_work_seconds": time.perf_counter() - total_started,
        "peak_allocated_gib": peak_gib,
        "base_model": args.base_model,
    }
    json.dump(metadata, open(output / "generation_metadata.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
