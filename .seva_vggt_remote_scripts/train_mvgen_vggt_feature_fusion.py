#!/usr/bin/env python3
"""Train a compact MVGen-feature -> VGGT-G DPT fusion adapter.

All pretrained MVGen/VGGT weights remain frozen.  The adapter receives the
local, source-view and scene-mean MVGen VAE latents and injects a gated residual
into the VGGT DPT feature consumed by the Gaussian head.  Supervision is the
final differentiable Gaussian render against paired real views.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TRAIN = [
    "apple_350_36665_68011", "ball_378_44141_87986",
    "bicycle_608_95757_192662", "car_336_34863_64261",
    "couch_503_72319_140597", "cup_20_722_1367",
    "teddybear_34_1423_4310", "vase_392_47724_94792",
]
VAL = ["bottle_601_93006_187080", "bottle_618_100690_201667",
       "toaster_375_42448_85012", "toaster_407_54986_106438"]


class MVGenVGGTFeatureFusion(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, channels, 1),
        )
        nn.init.normal_(self.project[-1].weight, std=1e-4)
        nn.init.zeros_(self.project[-1].bias)
        self.gate_logit = nn.Parameter(torch.full((1, 1, channels, 1, 1), -3.0))
        self.current_latents = None
        self.last_residual_rms = None

    def residual(self, height, width, dtype):
        lat = self.current_latents
        if lat is None:
            raise RuntimeError("current_latents must be set before VGGT forward")
        b, s, c, h, w = lat.shape
        source = lat[:, :1].expand(-1, s, -1, -1, -1)
        mean = lat.mean(dim=1, keepdim=True).expand(-1, s, -1, -1, -1)
        context = torch.cat([lat, source, mean], dim=2).reshape(b * s, 12, h, w)
        delta = self.project(context.float())
        delta = F.interpolate(delta, (height, width), mode="bilinear", align_corners=False)
        delta = delta.reshape(b, s, -1, height, width)
        delta = delta * torch.sigmoid(self.gate_logit)
        self.last_residual_rms = delta.float().square().mean().sqrt()
        return delta.to(dtype=dtype)

    def hook(self, _module, _inputs, output):
        return output + self.residual(output.shape[-2], output.shape[-1], output.dtype)


def load_scene(scene, manifests, generated, latent_root, load_images):
    manifest = json.loads((manifests / f"{scene}.json").read_text())
    gt_paths = [view["image"] for view in manifest["views"]]
    paths = [gt_paths[0]] + [str(generated / scene / "images" / f"view_{i:02d}_generated.png") for i in range(1, 8)]
    return {
        "scene": scene,
        "input": load_images(paths, mode="pad"),
        "gt": load_images(gt_paths, mode="pad"),
        "latents": torch.load(latent_root / f"{scene}.pt", map_location="cpu")["latents"],
    }


def scene_gaussians(prediction):
    return {key: value[0].reshape(-1, value.shape[-1]).float()
            for key, value in prediction["gaussians"].items()
            if torch.is_tensor(value) and value.ndim >= 4}


def render_loss(pred, targets, slots, pose_fn, render_fn, ssim_fn, size):
    extr, intr = pose_fn(pred["pose_enc"].float(), (518, 518))
    extr = extr[0].float(); intr = intr[0].float() * (size / 518.0); intr[:, 2, 2] = 1
    gs = scene_gaussians(pred)
    losses, values = [], []
    for slot in slots:
        target = F.interpolate(targets[slot:slot + 1].float(), (size, size), mode="bilinear",
                               align_corners=False, antialias=True)[0]
        render = render_fn(gs["xyz"], gs["rotation"], gs["scale"], gs["opacity"], gs["color"],
                           extr[slot], intr[slot], size, size)
        l1 = F.l1_loss(render, target)
        dssim = 1 - ssim_fn(render[None], target[None])
        losses.append(0.8 * l1 + 0.2 * dssim)
        with torch.no_grad():
            mse = F.mse_loss(render, target).item()
            values.append({"slot": slot, "psnr": -10 * math.log10(max(mse, 1e-10)),
                           "ssim": float((1 - dssim).item()), "l1": float(l1.item())})
    return torch.stack(losses).mean(), values


def evaluate(model, fusion, cache, device, dtype, pose_fn, render_fn, ssim_fn, size, enabled):
    rows = []
    model.eval(); fusion.eval()
    with torch.inference_mode():
        for sample in cache:
            fusion.current_latents = sample["latents"][None].to(device)
            handle = model.dpt_feature_head.register_forward_hook(fusion.hook) if enabled else None
            with torch.autocast("cuda", dtype=dtype):
                pred = model(sample["input"][None].to(device))
            if handle is not None: handle.remove()
            _, metrics = render_loss(pred, sample["gt"].to(device), list(range(1, 8)),
                                     pose_fn, render_fn, ssim_fn, size)
            rows.extend({"scene": sample["scene"], **row} for row in metrics)
            del pred; torch.cuda.empty_cache()
    return {"rows": rows, "mean_psnr": float(np.mean([r["psnr"] for r in rows])),
            "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
            "mean_l1": float(np.mean([r["l1"] for r in rows]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--render-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.gs_optimizer import render_gaussians, ssim
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    started = time.perf_counter()
    cache = [load_scene(s, args.manifests, args.generated, args.latents, load_and_preprocess_images)
             for s in TRAIN + VAL]
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    checkpoint = rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.point_head = None; model.track_head = None; model.requires_grad_(False); model.to(device).eval()
    fusion = MVGenVGGTFeatureFusion().to(device)
    train_cache, val_cache = cache[:len(TRAIN)], cache[len(TRAIN):]
    before = evaluate(model, fusion, val_cache, device, dtype, pose_encoding_to_extri_intri,
                      render_gaussians, ssim, args.render_size, False)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=0.01)
    # Keep the frozen model deterministic, but use the Gaussian head's training
    # path so its built-in per-frame chunking/checkpointing bounds activation RAM.
    # It contains GroupNorm (no running statistics), so this does not alter the
    # frozen model parameters or evaluation semantics.
    model.gaussian_head.train(); fusion.train()
    trace = []; torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        sample = train_cache[step % len(train_cache)]
        slot = 1 + ((step * 3 + step // len(train_cache)) % 7)
        optimizer.zero_grad(set_to_none=True)
        fusion.current_latents = sample["latents"][None].to(device)
        handle = model.dpt_feature_head.register_forward_hook(fusion.hook)
        with torch.autocast("cuda", dtype=dtype):
            pred = model(sample["input"][None].to(device))
        handle.remove()
        loss_render, metrics = render_loss(pred, sample["gt"].to(device), [0, slot],
                                           pose_encoding_to_extri_intri, render_gaussians,
                                           ssim, args.render_size)
        residual_reg = fusion.last_residual_rms
        loss = loss_render + 0.002 * residual_reg
        loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0).item())
        optimizer.step()
        row = {"step": step + 1, "scene": sample["scene"], "slot": slot,
               "loss": float(loss.item()), "render_loss": float(loss_render.item()),
               "residual_rms": float(residual_reg.item()), "grad_norm": grad,
               "gate_mean": float(torch.sigmoid(fusion.gate_logit).mean().item()), "metrics": metrics}
        trace.append(row); print(json.dumps(row), flush=True)
        del pred, loss, loss_render; torch.cuda.empty_cache()
    after = evaluate(model, fusion, val_cache, device, dtype, pose_encoding_to_extri_intri,
                     render_gaussians, ssim, args.render_size, True)
    adapter = args.output / "mvgen_vggt_feature_fusion_adapter.pt"
    torch.save({"state_dict": fusion.state_dict(), "architecture": "local+source+scene_mean VAE latent -> DPT256 gated residual",
                "steps": args.steps, "lr": args.lr, "seed": args.seed}, adapter)
    summary = {
        "method": "single-graph MVGen VAE latent -> VGGT DPT -> Gaussian differentiable render fusion",
        "level": "feasibility pilot; frozen backbones; trainable feature bridge only",
        "checkpoint": checkpoint, "train_scenes": TRAIN, "validation_scenes": VAL,
        "trainable_parameters": sum(p.numel() for p in fusion.parameters()),
        "steps": args.steps, "lr": args.lr, "render_size": args.render_size,
        "validation_before": before, "validation_after": after,
        "validation_psnr_delta": after["mean_psnr"] - before["mean_psnr"],
        "validation_ssim_delta": after["mean_ssim"] - before["mean_ssim"],
        "validation_l1_delta": after["mean_l1"] - before["mean_l1"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "adapter": str(adapter),
    }
    (args.output / "training_trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
