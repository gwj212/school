#!/usr/bin/env python3
"""Resume-safe shared-model MVGenMaster inference for the held-out 30-scene test."""

from __future__ import annotations

import argparse, copy, json, os, random, sys, time, traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from easydict import EasyDict
from omegaconf import OmegaConf
from torchvision.transforms import Compose, Normalize, ToTensor


def w2c(value):
    a = np.asarray(value, dtype=np.float32)
    if a.shape == (4, 4): return a
    out = np.eye(4, dtype=np.float32); out[:3] = a; return out


def choose_hw(path: Path, height: int, width: int):
    ratios = json.loads(path.read_text())
    return min((tuple(x) for x in ratios), key=lambda x: abs(x[0] / x[1] - height / width))


def complete(output: Path) -> bool:
    return (output / "generation_metadata.json").exists() and all(
        (output / "images" / ("view_00_real.png" if i == 0 else f"view_{i:02d}_generated.png")).exists()
        for i in range(8)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.0)
    args = ap.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.repo); sys.path.insert(0, str(args.repo))
    from depth_pro.depth_pro import create_model_and_transforms
    from depth_pro.utils import load_rgb
    from my_diffusers.models import UNet2DConditionModel
    from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
    from src.modules.schedulers import get_diffusion_scheduler

    protocol = json.loads((args.manifests / "selection_protocol.json").read_text())
    scenes = protocol["scenes"]
    model_dir = args.repo / "check_points" / "pretrained_model"
    config = EasyDict(OmegaConf.load(model_dir / "config.yaml"))
    config.pretrained_model_name_or_path = str(args.base_model.resolve())
    config.nframe = 8
    depth_model, depth_transform = create_model_and_transforms(device=torch.device("cuda"))
    depth_model.eval()
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", rank=0, model_cfg=config.model_cfg,
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True, local_files_only=True)
    unet.load_state_dict(torch.load(model_dir / "ema_unet.pt", map_location="cpu"))
    vae.requires_grad_(False).to("cuda", dtype=torch.float16)
    unet.requires_grad_(False).eval().to("cuda", dtype=torch.float16)
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        args.base_model, vae=vae, unet=unet, scheduler=get_diffusion_scheduler(config, name="DDIM"),
        safety_checker=None, torch_dtype=torch.float16, local_files_only=True).to("cuda")
    load_seconds = time.perf_counter() - started
    status = {"model_load_seconds": load_seconds, "completed": [], "failed": {}, "scenes": scenes}
    status_path = args.output / "status.json"
    rows = []
    for scene in scenes:
        out = args.output / scene
        if complete(out):
            meta = json.loads((out / "generation_metadata.json").read_text())
            rows.append(meta); status["completed"].append(scene); status_path.write_text(json.dumps(status, indent=2)); continue
        scene_started = time.perf_counter()
        try:
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            manifest = json.loads((args.manifests / f"{scene}.json").read_text())
            views = manifest["views"]; source, targets = views[0], views[1:]
            all_w2cs = np.stack([w2c(v["extrinsic"]) for v in views])
            c2ws = np.linalg.inv(all_w2cs)
            radius = float(np.linalg.norm(c2ws[0, :3, 3]))
            depth_started = time.perf_counter()
            rgb, _, _ = load_rgb(source["image"])
            inp = depth_transform(rgb)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                depth_raw = depth_model.infer(inp, f_px=torch.tensor(float(source["intrinsic"][0][0]), device="cuda"))["depth"].float().cpu().numpy()
            mask = np.asarray(Image.open(source["mask"]).convert("L")) > 127
            if mask.shape != depth_raw.shape:
                mask = cv2.resize(mask.astype(np.uint8), (depth_raw.shape[1], depth_raw.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            valid = mask & np.isfinite(depth_raw) & (depth_raw > 0)
            if valid.sum() < 100: valid = np.isfinite(depth_raw) & (depth_raw > 0)
            raw_median = float(np.median(depth_raw[valid])); alignment_scale = radius / max(raw_median, 1e-6)
            depth = depth_raw * alignment_scale
            depth_seconds = time.perf_counter() - depth_started
            positions = c2ws[:, :3, 3]
            extent = float(np.max(positions.max(0) - positions.min(0)))
            pose_scale = float(config.camera_longest_side / extent) if extent > config.camera_longest_side else 1.0
            all_w2cs[:, :3, 3] *= pose_scale; depth *= pose_scale
            original = Image.open(source["image"]).convert("RGB")
            h, w = choose_hw(model_dir / "ratio_set.json", original.height, original.width)
            resized = original.resize((w, h), Image.Resampling.LANCZOS if h < original.height else Image.Resampling.BICUBIC)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            tensor = Compose([ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])(resized)
            ks = []
            for v in views:
                k = np.asarray(v["intrinsic"], dtype=np.float32).copy()
                k[0] *= w / v["width"]; k[1] *= h / v["height"]; ks.append(k)
            images = torch.cat([tensor[None], torch.zeros((7, 3, h, w))]).cuda()
            intrinsics = torch.from_numpy(np.stack(ks)).cuda()
            extrinsics = torch.from_numpy(all_w2cs).cuda()
            depths = torch.cat([torch.from_numpy(depth)[None, None], torch.zeros((7, 1, h, w))]).cuda()
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            gen_started = time.perf_counter()
            local_config = copy.deepcopy(config); local_config.save_path = str(out)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                result = pipeline(images=images, nframe=8, cond_num=1, key_rescale=None,
                    height=h, width=w, intrinsics=intrinsics, extrinsics=extrinsics,
                    num_inference_steps=args.steps, guidance_scale=args.cfg, output_type="np",
                    config=local_config, tag=["custom"] * 8, class_label=0, depth=depths,
                    vae=pipeline.vae, generator=torch.Generator().manual_seed(args.seed)).images
            torch.cuda.synchronize(); generation_seconds = time.perf_counter() - gen_started
            save_started = time.perf_counter(); image_dir = out / "images"; image_dir.mkdir(parents=True, exist_ok=True)
            resized.save(image_dir / "view_00_real.png")
            for i, pred in enumerate((np.clip(result[1:], 0, 1) * 255).astype(np.uint8), 1):
                Image.fromarray(pred).save(image_dir / f"view_{i:02d}_generated.png")
            np.save(out / "source_depth_aligned.npy", depth.astype(np.float32))
            save_seconds = time.perf_counter() - save_started
            meta = {"scene_id": scene, "method": "MVGenMaster exact-nearest test cameras, one real plus seven generated",
                    "seed": args.seed, "steps": args.steps, "cfg": args.cfg, "resolution_hw": [h, w],
                    "target_slots": list(range(1, 8)), "target_camera_reference_indices": [v["original_index"] for v in targets],
                    "depth_alignment": {"raw_median_m": raw_median, "camera_radius": radius, "alignment_scale": alignment_scale},
                    "pose_scale": pose_scale, "timing_seconds": {"depth_estimation": depth_seconds,
                        "diffusion_generation": generation_seconds, "save_outputs": save_seconds,
                        "warm_scene_total": time.perf_counter() - scene_started},
                    "shared_model_load_seconds": load_seconds,
                    "cold_first_scene_total_estimate": load_seconds + (time.perf_counter() - scene_started),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
            (out / "generation_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
            rows.append(meta); status["completed"].append(scene); status["failed"].pop(scene, None)
            print(f"COMPLETE {scene} gen={generation_seconds:.3f}s warm={meta['timing_seconds']['warm_scene_total']:.3f}s", flush=True)
            del inp, depth_raw, images, intrinsics, extrinsics, depths, result
            torch.cuda.empty_cache()
        except Exception as exc:
            status["failed"][scene] = {"error": repr(exc), "traceback": traceback.format_exc()}
            print(f"FAILED {scene}: {exc!r}", flush=True)
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    status["elapsed_seconds"] = time.perf_counter() - started
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    if status["failed"]: raise SystemExit(1)


if __name__ == "__main__": main()
