#!/usr/bin/env python3
"""Generate one MVGen scene and cache its 1/8, 1/16 and 1/32 UNet features.

Features are captured from the conditional branch at the final denoising call.
For phase-1 training MVGen is frozen, so caching is exactly equivalent to
recomputing these activations.  Phase-2 LoRA training must recompute them.
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


class MultiScaleCollector:
    """Capture the pre-downsample activation of the first three UNet blocks."""

    names = ("scale_1_8", "scale_1_16", "scale_1_32")

    def __init__(self, unet, nframe: int):
        self.nframe = nframe
        self.latest: dict[str, torch.Tensor] = {}
        self.handles = [
            block.register_forward_hook(self._hook(name))
            for name, block in zip(self.names, unet.down_blocks[:3])
        ]

    def _hook(self, name):
        def save(_module, _inputs, output):
            # Diffusers down blocks return (hidden_states, residual_tuple).  The
            # largest residual is the activation before that block downsamples.
            residuals = output[1] if isinstance(output, tuple) and len(output) > 1 else (output,)
            tensors = [x for x in residuals if torch.is_tensor(x) and x.ndim == 4]
            if not tensors:
                raise RuntimeError(f"no 4D activation captured for {name}")
            value = max(tensors, key=lambda x: x.shape[-2] * x.shape[-1])
            if value.shape[0] % self.nframe:
                raise RuntimeError(f"{name}: batch {value.shape[0]} not divisible by nframe={self.nframe}")
            # CFG concatenates unconditional then conditional batches.  Keep the
            # last batch, which is the image/depth-conditioned branch.
            value = value.reshape(-1, self.nframe, *value.shape[1:])[-1]
            self.latest[name] = value.detach().to("cpu", dtype=torch.float16)

        return save

    def close(self):
        for handle in self.handles:
            handle.remove()


def pad_w2c(extrinsic):
    arr = np.asarray(extrinsic, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    out = np.eye(4, dtype=np.float32)
    out[:3, :4] = arr
    return out


def choose_hw(ratio_file, height, width):
    ratio_set = json.loads(Path(ratio_file).read_text())
    ratio_dict = {h / w: (h, w) for h, w in ratio_set}
    ratio = height / width
    return ratio_dict[min(ratio_dict, key=lambda value: abs(value - ratio))]


def infer_source_depth(image_path, focal_px, object_mask_path, camera_radius):
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
    return depth * scale, {"raw_median_m": raw_median, "camera_radius": camera_radius, "alignment_scale": scale}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("/root/seva_vggt/candidates/MVGenMaster"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--cfg", type=float, default=2.0)
    args = p.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    images_dir.mkdir(exist_ok=True)
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    from my_diffusers.models import UNet2DConditionModel
    from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
    from src.modules.schedulers import get_diffusion_scheduler

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    views = manifest["views"][:8]
    source, targets = views[0], views[1:]
    model_dir = args.repo / "check_points" / "pretrained_model"
    config = EasyDict(OmegaConf.load(model_dir / "config.yaml"))
    config.pretrained_model_name_or_path = str(args.base_model)
    config.nframe = len(views)

    w2cs = np.stack([pad_w2c(v["extrinsic"]) for v in views])
    c2ws = np.linalg.inv(w2cs)
    radius = float(np.linalg.norm(c2ws[0, :3, 3]))
    depth, depth_meta = infer_source_depth(source["image"], source["intrinsic"][0][0], source["mask"], radius)
    positions = c2ws[:, :3, 3]
    max_size = float(np.max(positions.max(0) - positions.min(0)))
    pose_scale = float(config.camera_longest_side / max_size) if max_size > config.camera_longest_side else 1.0
    w2cs[:, :3, 3] *= pose_scale
    depth *= pose_scale

    image = Image.open(source["image"]).convert("RGB")
    ow, oh = image.size
    h, w = choose_hw(model_dir / "ratio_set.json", oh, ow)
    resized = image.resize((w, h), Image.Resampling.LANCZOS if h < oh else Image.Resampling.BICUBIC)
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
    depths = torch.cat([torch.from_numpy(depth)[None, None], torch.zeros((7, 1, h, w))], 0).cuda()

    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", local_files_only=True).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", rank=0, model_cfg=config.model_cfg,
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True, local_files_only=True,
    )
    unet.load_state_dict(torch.load(model_dir / "ema_unet.pt", map_location="cpu"))
    unet.requires_grad_(False).eval()
    vae.to("cuda", dtype=torch.float16)
    unet.to("cuda", dtype=torch.float16)
    collector = MultiScaleCollector(unet, len(views))
    scheduler = get_diffusion_scheduler(config, name="DDIM")
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        args.base_model, vae=vae, unet=unet, scheduler=scheduler, safety_checker=None,
        torch_dtype=torch.float16, local_files_only=True,
    ).to("cuda")

    generator = torch.Generator().manual_seed(args.seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        result = pipeline(
            images=images, nframe=8, cond_num=1, key_rescale=None, height=h, width=w,
            intrinsics=intrinsics, extrinsics=extrinsics, num_inference_steps=args.steps,
            guidance_scale=args.cfg, output_type="np", config=copy.deepcopy(config),
            tag=["custom"] * 8, class_label=0, depth=depths, vae=pipeline.vae,
            generator=generator,
        ).images
    collector.close()
    missing = set(collector.names) - set(collector.latest)
    if missing:
        raise RuntimeError(f"missing captured features: {sorted(missing)}")

    resized.save(images_dir / "view_00_real.png")
    for slot, pred in enumerate((np.clip(result[1:], 0, 1) * 255).astype(np.uint8), 1):
        Image.fromarray(pred).save(images_dir / f"view_{slot:02d}_generated.png")
    feature_path = args.output / "mvgen_multiscale_features.pt"
    torch.save(
        {
            "scene": manifest["scene_id"],
            "features": collector.latest,
            "feature_shapes": {k: list(v.shape) for k, v in collector.latest.items()},
            "capture": "conditional branch, final denoising call, pre-downsample residual",
            "image_hw": [h, w],
        },
        feature_path,
    )
    metadata = {
        "scene": manifest["scene_id"], "seed": args.seed, "steps": args.steps,
        "cfg": args.cfg, "image_hw": [h, w], "pose_scale": pose_scale,
        "depth_alignment": depth_meta, "feature_path": str(feature_path),
        "feature_shapes": {k: list(v.shape) for k, v in collector.latest.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "generation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
