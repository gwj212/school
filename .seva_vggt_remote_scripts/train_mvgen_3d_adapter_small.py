#!/usr/bin/env python3
"""Small-sample MVGenMaster 3D-prior adapter fine-tuning.

Only ``unet.add_conv_in`` is trainable. The VAE, diffusion backbone and all
other MVGenMaster weights stay frozen. The checkpoint therefore contains only
the small adapter delta and never overwrites the upstream model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from easydict import EasyDict
from omegaconf import OmegaConf
from torchvision.transforms import Compose, Normalize, ToTensor


TARGET20 = [0, 2, 5, 7, 10, 12, 15, 17]


def pad_w2c(extrinsic):
    arr = np.asarray(extrinsic, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    out = np.eye(4, dtype=np.float32)
    out[:3, :4] = arr
    return out


def choose_hw(ratio_file: Path, height: int, width: int):
    ratio_set = json.loads(ratio_file.read_text())
    ratio_dict = {h / w: (h, w) for h, w in ratio_set}
    ratio = height / width
    return ratio_dict[min(ratio_dict, key=lambda value: abs(value - ratio))]


def load_manifest(path: Path):
    payload = json.loads(path.read_text())
    views = payload["views"]
    indices = TARGET20 if len(views) >= 18 else list(range(8))
    return payload["scene_id"], [views[i] for i in indices]


def infer_depth(depth_model, depth_transform, image_path: str, focal_px: float,
                mask_path: str | None, camera_radius: float):
    from depth_pro.utils import load_rgb

    rgb, _, _ = load_rgb(image_path)
    inp = depth_transform(rgb)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        prediction = depth_model.infer(inp, f_px=torch.tensor(float(focal_px), device="cuda"))
    depth = prediction["depth"].float().cpu().numpy()
    if mask_path and Path(mask_path).is_file():
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        if mask.shape != depth.shape:
            mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        valid = mask & np.isfinite(depth) & (depth > 0)
    else:
        valid = np.isfinite(depth) & (depth > 0)
    if int(valid.sum()) < 100:
        valid = np.isfinite(depth) & (depth > 0)
    scale = float(camera_radius / max(float(np.median(depth[valid])), 1e-6))
    return depth * scale


def source_depth_warps(images: torch.Tensor, depth: np.ndarray,
                       intrinsics: np.ndarray, extrinsics: np.ndarray):
    """Forward-warp the only real source view into every target view.

    The fixed warp is constructed from the source metric-depth estimate and
    exact cameras. It provides a geometry-aware supervision signal only where
    the real source surface is visible in the target view.
    """
    _, height, width = images[0].shape
    yy, xx = np.mgrid[0:height, 0:width]
    z = depth.reshape(-1).astype(np.float64)
    pix = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(height * width)], axis=0)
    source_k_inv = np.linalg.inv(intrinsics[0].astype(np.float64))
    source_cam = (source_k_inv @ pix) * z[None]
    source_r = extrinsics[0, :3, :3].astype(np.float64)
    source_t = extrinsics[0, :3, 3:4].astype(np.float64)
    world = source_r.T @ (source_cam - source_t)
    source_rgb = images[0].permute(1, 2, 0).numpy().reshape(-1, 3)
    source_valid = np.isfinite(z) & (z > 0) & np.isfinite(world).all(0)
    warps = torch.zeros_like(images)
    masks = torch.zeros((len(images), 1, height, width), dtype=torch.float32)
    warps[0] = images[0]
    masks[0] = 1
    for slot in range(1, len(images)):
        target_r = extrinsics[slot, :3, :3].astype(np.float64)
        target_t = extrinsics[slot, :3, 3:4].astype(np.float64)
        target_cam = target_r @ world + target_t
        target_z = target_cam[2]
        projected = intrinsics[slot].astype(np.float64) @ target_cam
        u = np.rint(projected[0] / np.maximum(projected[2], 1e-9)).astype(np.int64)
        v = np.rint(projected[1] / np.maximum(projected[2], 1e-9)).astype(np.int64)
        valid = source_valid & np.isfinite(target_z) & (target_z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        source_indices = np.nonzero(valid)[0]
        if not len(source_indices):
            continue
        target_pixels = v[valid] * width + u[valid]
        order = np.argsort(target_z[valid], kind="stable")
        ordered_pixels = target_pixels[order]
        _, first = np.unique(ordered_pixels, return_index=True)
        selected = source_indices[order[first]]
        selected_pixels = target_pixels[order[first]]
        out = np.zeros((height * width, 3), dtype=np.float32)
        mask = np.zeros(height * width, dtype=np.float32)
        out[selected_pixels] = source_rgb[selected]
        mask[selected_pixels] = 1.0
        warps[slot] = torch.from_numpy(out.reshape(height, width, 3)).permute(2, 0, 1)
        masks[slot, 0] = torch.from_numpy(mask.reshape(height, width))
    return warps, masks


def prepare_raw(path: Path, model_dir: Path, depth_model, depth_transform):
    scene, views = load_manifest(path)
    source = views[0]
    w2cs = np.stack([pad_w2c(v["extrinsic"]) for v in views])
    c2ws = np.linalg.inv(w2cs)
    source_radius = float(np.linalg.norm(c2ws[0, :3, 3]))
    depth = infer_depth(depth_model, depth_transform, source["image"],
                        source["intrinsic"][0][0], source.get("mask"), source_radius)
    positions = c2ws[:, :3, 3]
    max_size = float(np.max(positions.max(axis=0) - positions.min(axis=0)))
    pose_scale = 5.0 / max_size if max_size > 5.0 else 1.0
    w2cs[:, :3, 3] *= pose_scale
    depth *= pose_scale

    source_image = Image.open(source["image"]).convert("RGB")
    orig_w, orig_h = source_image.size
    h, w = choose_hw(model_dir / "ratio_set.json", orig_h, orig_w)
    transform = Compose([ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])
    images, intrinsics = [], []
    for view in views:
        image = Image.open(view["image"]).convert("RGB").resize(
            (w, h), Image.Resampling.LANCZOS if h < view["height"] else Image.Resampling.BICUBIC)
        images.append(transform(image))
        k = np.asarray(view["intrinsic"], dtype=np.float32).copy()
        k[0, :] *= w / float(view["width"])
        k[1, :] *= h / float(view["height"])
        intrinsics.append(k)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depths = torch.zeros((8, 1, h, w), dtype=torch.float32)
    depths[0, 0] = torch.from_numpy(depth)
    image_stack = torch.stack(images)
    intrinsic_stack = np.stack(intrinsics)
    source_warps, source_warp_masks = source_depth_warps(
        image_stack, depth, intrinsic_stack, w2cs)
    return {
        "scene": scene,
        "images": image_stack,
        "intrinsics": torch.from_numpy(intrinsic_stack),
        "extrinsics": torch.from_numpy(w2cs),
        "depths": depths,
        "source_warps": source_warps,
        "source_warp_masks": source_warp_masks,
        "hw": [h, w],
        "pose_scale": pose_scale,
    }


def cache_features(raw, vae, config, device, get_camera_embedding, get_3d_priors):
    images = raw["images"].to(device)
    h, w = images.shape[-2:]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        latents = vae.encode(images.to(torch.float16)).latent_dist.mode()
        latents = latents * vae.config.scaling_factor
    latents = latents.unsqueeze(0)
    _, f, _, lh, lw = latents.shape
    masks = torch.ones((1, f, 1, h, w), device=device, dtype=torch.float16)
    latent_masks = torch.ones((1, f, 1, lh, lw), device=device, dtype=torch.float16)
    masks[:, 0] = 0
    latent_masks[:, 0] = 0
    intrinsics = raw["intrinsics"].to(device)
    extrinsics = raw["extrinsics"].to(device)
    camera = get_camera_embedding(intrinsics, extrinsics, 1, f, h, w, config=config)
    camera = camera.to(device=device, dtype=torch.float16)
    add_inputs = torch.cat([masks, camera], dim=2)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        coords = get_3d_priors(
            config, raw["depths"], raw["intrinsics"], raw["extrinsics"], 1,
            nframe=f, device=device, colors=raw["images"], latents=latents,
            vae=vae, prior_type=config.model_cfg.get("prior_type", "3dpe"),
        )
    def cpu_half(value):
        if isinstance(value, list):
            return [item.detach().cpu().half() for item in value]
        return value.detach().cpu().half()
    return {
        "scene": raw["scene"],
        "images": raw["images"].detach().cpu().half(),
        "latents": latents.detach().cpu().half(),
        "latent_masks": latent_masks.detach().cpu().half(),
        "add_inputs": add_inputs.detach().cpu().half(),
        "coords": cpu_half(coords),
        "source_warps": raw["source_warps"].detach().cpu().half(),
        "source_warp_masks": raw["source_warp_masks"].detach().cpu().half(),
        "hw": raw["hw"],
    }


def to_device(value, device):
    if isinstance(value, list):
        return [item.to(device) for item in value]
    return value.to(device)


def image_edges(x):
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device, dtype=x.dtype)[None, None]
    ky = kx.transpose(-1, -2)
    return torch.cat([F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)], dim=1)


def diffusion_loss(sample, unet, scheduler, device, seed, vae=None,
                   pixel_weight=0.0, edge_weight=0.0, warp_weight=0.0,
                   max_timestep=None):
    latents = sample["latents"].to(device)
    masks = sample["latent_masks"].to(device)
    add_inputs = sample["add_inputs"].to(device)
    coords = to_device(sample["coords"], device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
    timestep_limit = int(max_timestep or scheduler.config.num_train_timesteps)
    timestep = torch.randint(50, timestep_limit, (1,),
                             generator=generator, device=device).long()
    noisy = scheduler.add_noise(latents, noise, timestep)
    inputs = latents * (1 - masks) + noisy * masks
    class_labels = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = unet(inputs, timestep, None, add_inputs=add_inputs,
                    class_labels=class_labels, coords=coords, return_dict=False)[0]
    pred = einops.rearrange(pred, "(b f) c h w -> b f c h w", f=8)
    noise_loss = F.mse_loss(pred.float()[:, 1:], noise.float()[:, 1:], reduction="mean")
    pixel_loss = torch.zeros((), device=device)
    edge_loss = torch.zeros((), device=device)
    warp_loss = torch.zeros((), device=device)
    if vae is not None and (pixel_weight > 0 or edge_weight > 0 or warp_weight > 0):
        alpha = scheduler.alphas_cumprod[timestep].to(device=device, dtype=latents.dtype).view(1, 1, 1, 1, 1)
        pred_x0 = (noisy - (1 - alpha).sqrt() * pred) / alpha.sqrt().clamp_min(1e-4)
        target_slot = 1 + (seed % 7)
        with torch.autocast("cuda", dtype=torch.float16):
            decoded = vae.decode(pred_x0[:, target_slot] / vae.config.scaling_factor).sample.float()
        target = sample["images"][target_slot:target_slot + 1].to(device).float()
        pixel_loss = F.l1_loss(decoded, target)
        edge_loss = F.l1_loss(image_edges(decoded), image_edges(target))
        if warp_weight > 0:
            warped = sample["source_warps"][target_slot:target_slot + 1].to(device).float()
            warp_mask = sample["source_warp_masks"][target_slot:target_slot + 1].to(device).float()
            valid_channels = warp_mask.expand_as(decoded)
            warp_loss = ((decoded - warped).abs() * valid_channels).sum() / valid_channels.sum().clamp_min(1.0)
    total = noise_loss + pixel_weight * pixel_loss + edge_weight * edge_loss + warp_weight * warp_loss
    return total, noise_loss.detach(), pixel_loss.detach(), edge_loss.detach(), warp_loss.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifests", nargs="+", type=Path, required=True)
    ap.add_argument("--val-manifests", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--pixel-weight", type=float, default=0.0)
    ap.add_argument("--edge-weight", type=float, default=0.0)
    ap.add_argument("--warp-weight", type=float, default=0.0)
    ap.add_argument("--max-timestep", type=int, default=400)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    repo = Path("/root/seva_vggt/candidates/MVGenMaster")
    os.chdir(repo); sys.path.insert(0, str(repo))
    from depth_pro.depth_pro import create_model_and_transforms
    from my_diffusers.models import UNet2DConditionModel
    from src.modules.camera import get_camera_embedding
    from src.modules.position_encoding import get_3d_priors
    from src.modules.schedulers import get_diffusion_scheduler

    device = torch.device("cuda:0")
    model_dir = repo / "check_points" / "pretrained_model"
    config = EasyDict(OmegaConf.load(model_dir / "config.yaml"))
    config.pretrained_model_name_or_path = str(repo / "check_points" / "sd2-community-stable-diffusion-2-1")
    config.nframe = 8
    config.camera_longest_side = 5.0

    started = time.perf_counter()
    depth_model, depth_transform = create_model_and_transforms(device=device)
    depth_model.eval()
    raws = [prepare_raw(path, model_dir, depth_model, depth_transform)
            for path in args.train_manifests + args.val_manifests]
    del depth_model
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path,
                                        subfolder="vae", local_files_only=True)
    vae.requires_grad_(False).to(device, dtype=torch.float16).eval()
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", rank=0,
        model_cfg=config.model_cfg, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, local_files_only=True)
    weights = torch.load(model_dir / "ema_unet.pt", map_location="cpu")
    unet.load_state_dict(weights)
    unet.requires_grad_(False)
    unet.add_conv_in.requires_grad_(True)
    unet.enable_gradient_checkpointing()
    unet.to(device).train()
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    cached = [cache_features(raw, vae, config, device, get_camera_embedding, get_3d_priors)
              for raw in raws]
    train_cache = cached[:len(args.train_manifests)]
    val_cache = cached[len(args.train_manifests):]
    del raws
    torch.cuda.empty_cache()
    scheduler = get_diffusion_scheduler(config, name="DDPM")

    unet.eval()
    with torch.no_grad():
        val_before = [float(diffusion_loss(s, unet, scheduler, device, 1000 + i, vae,
                                          args.pixel_weight, args.edge_weight, args.warp_weight,
                                          args.max_timestep)[0].item())
                      for i, s in enumerate(val_cache)]
    unet.train()
    optimizer = torch.optim.AdamW(unet.add_conv_in.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=0.01)
    trace = []
    for step in range(args.steps):
        sample = train_cache[step % len(train_cache)]
        optimizer.zero_grad(set_to_none=True)
        loss, noise_loss, pixel_loss, edge_loss, warp_loss = diffusion_loss(
            sample, unet, scheduler, device, args.seed * 10000 + step, vae,
            args.pixel_weight, args.edge_weight, args.warp_weight, args.max_timestep)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(unet.add_conv_in.parameters(), 1.0).item())
        optimizer.step()
        row = {"step": step + 1, "scene": sample["scene"], "loss": float(loss.item()),
               "noise_loss": float(noise_loss), "pixel_loss": float(pixel_loss),
               "edge_loss": float(edge_loss), "warp_loss": float(warp_loss), "grad_norm": grad_norm}
        trace.append(row)
        print(json.dumps(row), flush=True)

    unet.eval()
    with torch.no_grad():
        val_after = [float(diffusion_loss(s, unet, scheduler, device, 1000 + i, vae,
                                         args.pixel_weight, args.edge_weight, args.warp_weight,
                                         args.max_timestep)[0].item())
                     for i, s in enumerate(val_cache)]
    adapter = {key: value.detach().cpu() for key, value in unet.add_conv_in.state_dict().items()}
    checkpoint = args.output / "mvgen_add_conv_in_adapter.pt"
    torch.save({"add_conv_in": adapter, "steps": args.steps, "lr": args.lr,
                "seed": args.seed, "trainable_parameters": trainable,
                "pixel_weight": args.pixel_weight, "edge_weight": args.edge_weight,
                "warp_weight": args.warp_weight,
                "max_timestep": args.max_timestep}, checkpoint)
    summary = {
        "method": "MVGenMaster frozen backbone + trainable 3D-prior add_conv_in adapter",
        "trainable_parameters": trainable,
        "steps": args.steps, "lr": args.lr, "seed": args.seed,
        "pixel_weight": args.pixel_weight, "edge_weight": args.edge_weight,
        "warp_weight": args.warp_weight,
        "max_timestep": args.max_timestep,
        "train_scenes": [s["scene"] for s in train_cache],
        "validation_scenes": [s["scene"] for s in val_cache],
        "validation_diffusion_loss_before": val_before,
        "validation_diffusion_loss_after": val_after,
        "validation_mean_before": float(np.mean(val_before)),
        "validation_mean_after": float(np.mean(val_after)),
        "validation_relative_change": float(np.mean(val_after) / np.mean(val_before) - 1.0),
        "checkpoint": str(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    (args.output / "training_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
