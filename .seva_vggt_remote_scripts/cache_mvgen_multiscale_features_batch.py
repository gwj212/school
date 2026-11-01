#!/usr/bin/env python3
"""Cache MVGen multi-scale features for all GeoDiff splits in one process.

The single-scene cache utility is intentionally simple but reloads DepthPro,
the VAE and the UNet for every scene.  This runner preserves the exact scene
pre-processing and output contract while keeping those frozen models resident
on the GPU.  Existing complete caches with matching sampling settings are
skipped, so interrupted runs are resumable.
"""

from __future__ import annotations

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

from cache_mvgen_multiscale_features import (
    MultiScaleCollector,
    choose_hw,
    pad_w2c,
)


class BatchRuntime:
    def __init__(self, repo: Path, base_model: Path, seed: int):
        self.repo = repo
        self.base_model = base_model
        self.seed = seed
        os.chdir(repo)
        sys.path.insert(0, str(repo))

        from depth_pro.depth_pro import create_model_and_transforms
        from my_diffusers.models import UNet2DConditionModel
        from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import (
            StableDiffusionMultiViewPipeline,
        )
        from src.modules.schedulers import get_diffusion_scheduler

        self.model_dir = repo / "check_points" / "pretrained_model"
        self.config = EasyDict(OmegaConf.load(self.model_dir / "config.yaml"))
        self.config.pretrained_model_name_or_path = str(base_model)
        self.config.nframe = 8

        self.depth_model, self.depth_transform = create_model_and_transforms(
            device=torch.device("cuda")
        )
        self.depth_model.eval()

        vae = AutoencoderKL.from_pretrained(
            base_model, subfolder="vae", local_files_only=True
        ).requires_grad_(False)
        unet = UNet2DConditionModel.from_pretrained(
            base_model,
            subfolder="unet",
            rank=0,
            model_cfg=self.config.model_cfg,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        unet.load_state_dict(torch.load(self.model_dir / "ema_unet.pt", map_location="cpu"))
        unet.requires_grad_(False).eval()
        vae.to("cuda", dtype=torch.float16)
        unet.to("cuda", dtype=torch.float16)

        self.collector = MultiScaleCollector(unet, 8)
        scheduler = get_diffusion_scheduler(self.config, name="DDIM")
        self.pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
            base_model,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

    def infer_source_depth(self, image_path, focal_px, object_mask_path, camera_radius):
        from depth_pro.utils import load_rgb

        rgb, _, _ = load_rgb(image_path)
        inp = self.depth_transform(rgb)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = self.depth_model.infer(
                inp, f_px=torch.tensor(float(focal_px), device="cuda")
            )
        depth = prediction["depth"].float().cpu().numpy()
        mask = np.asarray(Image.open(object_mask_path).convert("L")) > 127
        if mask.shape != depth.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        valid = mask & np.isfinite(depth) & (depth > 0)
        if valid.sum() < 100:
            valid = np.isfinite(depth) & (depth > 0)
        raw_median = float(np.median(depth[valid]))
        scale = float(camera_radius / max(raw_median, 1e-6))
        return depth * scale, {
            "raw_median_m": raw_median,
            "camera_radius": camera_radius,
            "alignment_scale": scale,
        }

    def cache_scene(self, manifest_path: Path, output: Path, steps: int, cfg: float):
        started = time.perf_counter()
        output.mkdir(parents=True, exist_ok=True)
        images_dir = output / "images"
        images_dir.mkdir(exist_ok=True)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        manifest = json.loads(manifest_path.read_text())
        views = manifest["views"][:8]
        if len(views) != 8:
            raise RuntimeError(f"{manifest_path}: expected 8 views, got {len(views)}")
        source = views[0]

        w2cs = np.stack([pad_w2c(v["extrinsic"]) for v in views])
        c2ws = np.linalg.inv(w2cs)
        radius = float(np.linalg.norm(c2ws[0, :3, 3]))
        depth, depth_meta = self.infer_source_depth(
            source["image"], source["intrinsic"][0][0], source["mask"], radius
        )
        positions = c2ws[:, :3, 3]
        max_size = float(np.max(positions.max(0) - positions.min(0)))
        pose_scale = (
            float(self.config.camera_longest_side / max_size)
            if max_size > self.config.camera_longest_side
            else 1.0
        )
        w2cs[:, :3, 3] *= pose_scale
        depth *= pose_scale

        image = Image.open(source["image"]).convert("RGB")
        ow, oh = image.size
        h, w = choose_hw(self.model_dir / "ratio_set.json", oh, ow)
        resized = image.resize(
            (w, h), Image.Resampling.LANCZOS if h < oh else Image.Resampling.BICUBIC
        )
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        tensor = Compose([ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])(resized)
        images = torch.cat([tensor[None], torch.zeros((7, 3, h, w))], 0).cuda()

        intrinsics = []
        for view in views:
            k = np.asarray(view["intrinsic"], dtype=np.float32).copy()
            k[0] *= w / view["width"]
            k[1] *= h / view["height"]
            intrinsics.append(k)
        intrinsics = torch.from_numpy(np.stack(intrinsics)).cuda()
        extrinsics = torch.from_numpy(w2cs).cuda()
        depths = torch.cat(
            [torch.from_numpy(depth)[None, None], torch.zeros((7, 1, h, w))], 0
        ).cuda()

        self.collector.latest.clear()
        generator = torch.Generator().manual_seed(self.seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            result = self.pipeline(
                images=images,
                nframe=8,
                cond_num=1,
                key_rescale=None,
                height=h,
                width=w,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_inference_steps=steps,
                guidance_scale=cfg,
                output_type="np",
                config=copy.deepcopy(self.config),
                tag=["custom"] * 8,
                class_label=0,
                depth=depths,
                vae=self.pipeline.vae,
                generator=generator,
            ).images
        missing = set(self.collector.names) - set(self.collector.latest)
        if missing:
            raise RuntimeError(f"missing captured features: {sorted(missing)}")

        resized.save(images_dir / "view_00_real.png")
        for slot, pred in enumerate(
            (np.clip(result[1:], 0, 1) * 255).astype(np.uint8), 1
        ):
            Image.fromarray(pred).save(images_dir / f"view_{slot:02d}_generated.png")
        feature_path = output / "mvgen_multiscale_features.pt"
        torch.save(
            {
                "scene": manifest["scene_id"],
                "features": self.collector.latest,
                "feature_shapes": {
                    k: list(v.shape) for k, v in self.collector.latest.items()
                },
                "capture": "conditional branch, final denoising call, pre-downsample residual",
                "image_hw": [h, w],
            },
            feature_path,
        )
        metadata = {
            "scene": manifest["scene_id"],
            "seed": self.seed,
            "steps": steps,
            "cfg": cfg,
            "image_hw": [h, w],
            "pose_scale": pose_scale,
            "depth_alignment": depth_meta,
            "feature_path": str(feature_path),
            "feature_shapes": {
                k: list(v.shape) for k, v in self.collector.latest.items()
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output / "generation_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        return metadata

    def close(self):
        self.collector.close()


def cache_matches(output: Path, steps: int, cfg: float) -> bool:
    feature = output / "mvgen_multiscale_features.pt"
    metadata = output / "generation_metadata.json"
    if not feature.is_file() or feature.stat().st_size == 0 or not metadata.is_file():
        return False
    try:
        info = json.loads(metadata.read_text())
        return info.get("steps") == steps and float(info.get("cfg")) == float(cfg)
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("/root/seva_vggt/candidates/MVGenMaster"))
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, required=True)
    p.add_argument("--blind-manifest-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "blind_test"),
        default=("train", "validation"),
    )
    args = p.parse_args()

    protocol = json.loads(args.protocol.read_text())
    jobs = []
    if "train" in args.splits:
        for scene in protocol["train_scenes"]:
            jobs.append(("train", scene, args.manifest_root / "train" / f"{scene}.json"))
    if "validation" in args.splits:
        for scene in protocol["validation_scenes"]:
            jobs.append(("validation", scene, args.manifest_root / "validation" / f"{scene}.json"))
    if "blind_test" in args.splits:
        for scene in protocol["blind_test_scenes"]:
            jobs.append(("blind_test", scene, args.blind_manifest_root / f"{scene}.json"))

    pending = [job for job in jobs if not cache_matches(args.cache_root / job[1], args.steps, args.cfg)]
    print(json.dumps({"total": len(jobs), "cached": len(jobs) - len(pending), "pending": len(pending)}), flush=True)
    if not pending:
        return

    runtime = BatchRuntime(args.repo, args.base_model, args.seed)
    run_started = time.perf_counter()
    try:
        for index, (split, scene, manifest) in enumerate(pending, 1):
            metadata = runtime.cache_scene(
                manifest, args.cache_root / scene, args.steps, args.cfg
            )
            elapsed = time.perf_counter() - run_started
            rate = elapsed / index
            status = {
                "completed_this_run": index,
                "pending_this_run": len(pending),
                "split": split,
                "scene": scene,
                "scene_seconds": metadata["elapsed_seconds"],
                "mean_seconds": rate,
                "eta_seconds": rate * (len(pending) - index),
            }
            print("BATCH_PROGRESS " + json.dumps(status), flush=True)
            (args.cache_root.parent / "cache_progress.json").write_text(
                json.dumps(status, indent=2) + "\n"
            )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
