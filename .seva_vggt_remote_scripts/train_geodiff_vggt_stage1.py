#!/usr/bin/env python3
"""Stage-1 GeoDiff-VGGT training: multiscale bridge, gates and visibility.

MVGen and VGGT are frozen.  Cached MVGen UNet features at image scales
1/8, 1/16 and 1/32 condition VGGT layers 4/11/17/23.  The same fused tokens
also predict a per-pixel visibility multiplier for Gaussian opacity.
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

LAYER_TO_SCALE = {4: "scale_1_8", 11: "scale_1_16", 17: "scale_1_32", 23: "scale_1_32"}
MODEL_CANVAS = 518
PATCH_SIZE = 14


class ScaleCrossAdapter(nn.Module):
    def __init__(self, feature_channels: int, token_dim: int = 2048, bottleneck: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(feature_channels, bottleneck, 1),
            nn.GroupNorm(16, bottleneck),
            nn.GELU(),
        )
        self.query_norm = nn.LayerNorm(token_dim)
        self.query = nn.Linear(token_dim, bottleneck)
        self.attn = nn.MultiheadAttention(bottleneck, 8, batch_first=True)
        self.out = nn.Linear(bottleneck, token_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, tokens: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        # tokens [B,S,P,D], feature [B,S,C,H,W]
        b, s, p, _ = tokens.shape
        x = feature.reshape(b * s, *feature.shape[2:]).float()
        x = self.feature(x)
        x = F.adaptive_avg_pool2d(x, (8, 8)).flatten(2).transpose(1, 2)
        q = self.query(self.query_norm(tokens.float())).reshape(b * s, p, -1)
        residual, _ = self.attn(q, x, x, need_weights=False)
        return self.out(residual).reshape(b, s, p, -1)


class GeoDiffBridge(nn.Module):
    layers = (4, 11, 17, 23)

    def __init__(self, channels: dict[str, int], token_dim: int = 2048):
        super().__init__()
        self.adapters = nn.ModuleDict(
            {str(layer): ScaleCrossAdapter(channels[scale], token_dim) for layer, scale in LAYER_TO_SCALE.items()}
        )
        self.layer_gates = nn.Parameter(torch.full((4,), -4.0))
        self.camera_gate = nn.Parameter(torch.tensor(-4.0))
        self.visibility = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, 256), nn.GELU(), nn.Linear(256, 1)
        )
        nn.init.zeros_(self.visibility[-1].weight)
        nn.init.constant_(self.visibility[-1].bias, 4.0)
        self.current_features: dict[str, torch.Tensor] | None = None
        self.visibility_logits: torch.Tensor | None = None
        self.last_residual_rms = torch.tensor(0.0)

    def hook(self, _module, _inputs, output):
        if self.current_features is None:
            raise RuntimeError("current_features must be set before VGGT forward")
        tokens, patch_start = output
        result = list(tokens)
        rms = []
        for gate_index, layer in enumerate(self.layers):
            value = result[layer]
            patch = value[:, :, patch_start:]
            scale = LAYER_TO_SCALE[layer]
            feature = self.current_features[scale]
            residual = self.adapters[str(layer)](patch, feature)
            residual = torch.sigmoid(self.layer_gates[gate_index]) * residual.to(patch.dtype)
            fused_patch = patch + residual
            if layer == 23:
                camera = value[:, :, 0] + torch.sigmoid(self.camera_gate) * residual.mean(2)
                prefix = torch.cat([camera.unsqueeze(2), value[:, :, 1:patch_start]], dim=2)
                self.visibility_logits = self.visibility(fused_patch.float()).squeeze(-1)
            else:
                prefix = value[:, :, :patch_start]
            # Avoid in-place slice assignment: downstream frozen heads still
            # need autograd to propagate into the injected bridge residual.
            result[layer] = torch.cat([prefix, fused_patch], dim=2)
            rms.append(residual.float().square().mean())
        # The output projections are zero-initialized, so the first residual is
        # exactly zero.  Add epsilon before sqrt to keep the derivative finite
        # at initialization.
        self.last_residual_rms = (torch.stack(rms).mean() + 1e-12).sqrt()
        return result, patch_start

    def apply_visibility(self, prediction: dict) -> torch.Tensor:
        if self.visibility_logits is None:
            raise RuntimeError("visibility logits were not produced")
        opacity = prediction["gaussians"]["opacity"]
        # GaussianHead stores attributes as [B,S,N,C], where N=H*W.
        b, s, n, channels = opacity.shape
        side_out = int(round(math.sqrt(n)))
        if side_out * side_out != n or channels != 1:
            raise ValueError(f"unexpected opacity shape: {tuple(opacity.shape)}")
        p = self.visibility_logits.shape[-1]
        side = int(round(math.sqrt(p)))
        if side * side != p:
            raise ValueError(f"visibility patch count is not square: {p}")
        gate = 0.05 + 0.95 * torch.sigmoid(self.visibility_logits)
        gate = F.interpolate(
            gate.reshape(b * s, 1, side, side),
            (side_out, side_out),
            mode="bilinear",
            align_corners=False,
        )
        gate = gate.reshape(b, s, n, 1).to(opacity.dtype)
        prediction["gaussians"]["opacity"] = opacity * gate
        return gate


def normalized_log_depth(depth):
    depth = depth.float().clamp_min(1e-4)
    # VGGT commonly returns [..., H, W, 1].  Remove only the scalar channel
    # and center once per full depth map (not independently for every row).
    if depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    logd = depth.log()
    center = logd.flatten(-2).median(-1).values[..., None, None]
    return logd - center


def depth_metrics(pred, target):
    pred = pred.float()
    target = target.to(pred.device).float()
    if pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if target.shape[-1] == 1:
        target = target.squeeze(-1)
    valid = torch.isfinite(pred) & torch.isfinite(target) & (pred > 1e-5) & (target > 1e-5)
    p, t = pred[valid].double(), target[valid].double()
    scale = torch.median(t) / torch.median(p).clamp_min(1e-9)
    p = p * scale
    rel = (p - t).abs() / t.clamp_min(1e-9)
    loge = torch.log(p.clamp_min(1e-9)) - torch.log(t.clamp_min(1e-9))
    return {
        "depth_absrel": float(rel.mean()),
        "depth_bad20": float((rel > 0.20).double().mean()),
        "depth_silog": float(torch.sqrt((loge.square().mean() - loge.mean().square()).clamp_min(0))),
    }


def valid_canvas_mask(views):
    valid = torch.zeros((len(views), MODEL_CANVAS, MODEL_CANVAS), dtype=torch.bool)
    for index, view in enumerate(views):
        width, height = int(view["width"]), int(view["height"])
        if width >= height:
            resized_width = MODEL_CANVAS
            resized_height = round(height * (MODEL_CANVAS / width) / PATCH_SIZE) * PATCH_SIZE
        else:
            resized_height = MODEL_CANVAS
            resized_width = round(width * (MODEL_CANVAS / height) / PATCH_SIZE) * PATCH_SIZE
        pad_width = MODEL_CANVAS - resized_width
        pad_height = MODEL_CANVAS - resized_height
        x0, y0 = pad_width // 2, pad_height // 2
        valid[index, y0:y0 + resized_height, x0:x0 + resized_width] = True
    return valid


def scene_gaussians(prediction, valid):
    # Padding pixels create real Gaussians in VGGT-G and severely occlude the
    # object when all source views are flattened.  Apply the known source-view
    # canvas mask before every differentiable render.
    valid = valid.flatten(1).to(prediction["gaussians"]["xyz"].device)
    result = {}
    for key, value in prediction["gaussians"].items():
        if torch.is_tensor(value) and value.ndim == 4:
            if value.shape[1] != valid.shape[0] or value.shape[2] != valid.shape[1]:
                raise ValueError(f"{key}: Gaussian shape {tuple(value.shape)} incompatible with mask {tuple(valid.shape)}")
            result[key] = value[0][valid].float()
    return result


def render_loss(pred, targets, valid, slots, pose_fn, render_fn, ssim_fn, size):
    extr, intr = pose_fn(pred["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS))
    extr = extr[0].float()
    intr = intr[0].float() * (size / MODEL_CANVAS)
    intr[:, 2, 2] = 1
    gs = scene_gaussians(pred, valid)
    losses, values = [], []
    for slot in slots:
        target = F.interpolate(
            targets[slot:slot + 1].float(),
            (size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        render = render_fn(
            gs["xyz"], gs["rotation"], gs["scale"], gs["opacity"], gs["color"],
            extr[slot], intr[slot], size, size,
        )
        l1 = F.l1_loss(render, target)
        dssim = 1 - ssim_fn(render[None], target[None])
        losses.append(0.8 * l1 + 0.2 * dssim)
        with torch.no_grad():
            mse = F.mse_loss(render, target).item()
            values.append({
                "slot": slot,
                "psnr": -10 * math.log10(max(mse, 1e-10)),
                "ssim": float((1 - dssim).item()),
                "l1": float(l1.item()),
            })
    return torch.stack(losses).mean(), values


def load_sample(scene: str, manifest_dir: Path, cache_root: Path, load_images):
    manifest = json.loads((manifest_dir / f"{scene}.json").read_text())
    views = manifest["views"][:8]
    gt_paths = [v["image"] for v in views]
    scene_root = cache_root / scene
    input_paths = [gt_paths[0]] + [str(scene_root / "images" / f"view_{i:02d}_generated.png") for i in range(1, 8)]
    cached = torch.load(scene_root / "mvgen_multiscale_features.pt", map_location="cpu")
    return {
        "scene": scene,
        "input": load_images(input_paths, mode="pad"),
        "gt": load_images(gt_paths, mode="pad"),
        "features": cached["features"],
        "valid": valid_canvas_mask(views),
    }


def move_features(features, device):
    return {k: v[None].to(device) for k, v in features.items()}


def run_model(model, bridge, sample, device, dtype, enabled=True):
    bridge.current_features = move_features(sample["features"], device)
    handle = model.aggregator.register_forward_hook(bridge.hook) if enabled else None
    with torch.autocast("cuda", dtype=dtype):
        pred = model(sample["input"][None].to(device))
    if handle:
        handle.remove()
        gate = bridge.apply_visibility(pred)
    else:
        gate = None
    return pred, gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--render-size", type=int, default=192)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--max-train-scenes", type=int)
    p.add_argument("--max-val-scenes", type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    protocol = json.loads(args.protocol.read_text())
    train_scenes, val_scenes = protocol["train_scenes"], protocol["validation_scenes"]
    if args.max_train_scenes is not None:
        train_scenes = train_scenes[:args.max_train_scenes]
    if args.max_val_scenes is not None:
        val_scenes = val_scenes[:args.max_val_scenes]
    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.gs_optimizer import render_gaussians, ssim

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    train = [load_sample(s, args.manifest_root / "train", args.cache_root, load_and_preprocess_images) for s in train_scenes]
    val = [load_sample(s, args.manifest_root / "validation", args.cache_root, load_and_preprocess_images) for s in val_scenes]
    channels = {k: int(v.shape[1]) for k, v in train[0]["features"].items()}
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    checkpoint = rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.point_head = None; model.track_head = None
    model.requires_grad_(False).to(device).eval(); model._release_depth = True
    # Keep the frozen backbone graph-free, but allow the injected bridge
    # residual to receive gradients through DPT/Gaussian rendering.
    model._allow_dpt_token_grad = True
    bridge = GeoDiffBridge(channels).to(device)

    # Real-eight VGGT depth is a frozen teacher, never an inference input.
    # Skip unrelated camera/Gaussian heads and persist each teacher map so an
    # interrupted 300-scene run can resume without recomputing earlier scenes.
    saved_camera = model.camera_head
    saved_dpt_feature = model.dpt_feature_head
    saved_gaussian = model.gaussian_head
    model.camera_head = None
    model.dpt_feature_head = None
    model.gaussian_head = None
    with torch.inference_mode():
        for teacher_index, sample in enumerate(train + val, 1):
            teacher_path = args.cache_root / sample["scene"] / "vggt_real8_teacher_depth.pt"
            if teacher_path.is_file():
                teacher_cache = torch.load(teacher_path, map_location="cpu")
                if isinstance(teacher_cache, dict):
                    sample["teacher_depth_raw"] = teacher_cache["raw"]
                    sample["teacher_depth"] = teacher_cache["normalized"]
                else:
                    # Compatibility with an interrupted run from the earlier
                    # normalized-only cache format: recompute once.
                    teacher_path.unlink()
                    with torch.autocast("cuda", dtype=dtype):
                        teacher = model(sample["gt"][None].to(device))
                    raw = teacher["depth"][0].cpu().half()
                    normalized = normalized_log_depth(teacher["depth"][0]).cpu().half()
                    sample["teacher_depth_raw"] = raw
                    sample["teacher_depth"] = normalized
                    torch.save({"raw": raw, "normalized": normalized}, teacher_path)
                    del teacher
            else:
                with torch.autocast("cuda", dtype=dtype):
                    teacher = model(sample["gt"][None].to(device))
                raw = teacher["depth"][0].cpu().half()
                normalized = normalized_log_depth(teacher["depth"][0]).cpu().half()
                sample["teacher_depth_raw"] = raw
                sample["teacher_depth"] = normalized
                torch.save({"raw": raw, "normalized": normalized}, teacher_path)
                del teacher
                torch.cuda.empty_cache()
            print(f"TEACHER {teacher_index}/{len(train) + len(val)} {sample['scene']}", flush=True)
    model.camera_head = saved_camera
    model.dpt_feature_head = saved_dpt_feature
    model.gaussian_head = saved_gaussian

    # Validation baseline is computed before any bridge update.  Model
    # selection and the decision to enter stage 2 therefore use validation
    # deltas only; the sealed blind30 remains untouched until the end.
    baseline_rows = []
    bridge.eval(); model.eval()
    with torch.inference_mode():
        for baseline_index, sample in enumerate(val, 1):
            pred, _ = run_model(model, bridge, sample, device, dtype, False)
            _, metrics = render_loss(
                pred,
                sample["gt"].to(device),
                sample["valid"],
                list(range(1, 8)),
                pose_encoding_to_extri_intri,
                render_gaussians,
                ssim,
                args.render_size,
            )
            pred_depth = pred["depth"][0].float()
            for item in metrics:
                item.update(depth_metrics(pred_depth[item["slot"]], sample["teacher_depth_raw"][item["slot"]]))
            baseline_rows.extend({"scene": sample["scene"], **m} for m in metrics)
            del pred
            torch.cuda.empty_cache()
            print(f"VAL_BASELINE {baseline_index}/{len(val)} {sample['scene']}", flush=True)
    baseline_psnr = float(np.mean([r["psnr"] for r in baseline_rows]))
    baseline_metrics = {
        key: float(np.mean([r[key] for r in baseline_rows]))
        for key in ("psnr", "ssim", "l1", "depth_absrel", "depth_bad20", "depth_silog")
    }
    (args.output / "validation_baseline.json").write_text(
        json.dumps({"metrics": baseline_metrics, "rows": baseline_rows}, indent=2) + "\n"
    )

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    trace = []
    best = -float("inf")
    best_metrics = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train)
        model.gaussian_head.train(); bridge.train()
        for index, sample in enumerate(train):
            slot = 1 + ((index + epoch * 3) % 7)
            optimizer.zero_grad(set_to_none=True)
            pred, gate = run_model(model, bridge, sample, device, dtype, True)
            render, metrics = render_loss(pred, sample["gt"].to(device), sample["valid"], [0, slot], pose_encoding_to_extri_intri, render_gaussians, ssim, args.render_size)
            depth = normalized_log_depth(pred["depth"][0, [0, slot]])
            teacher = sample["teacher_depth"][[0, slot]].to(device).float()
            depth_loss = F.smooth_l1_loss(depth, teacher, beta=0.1)
            visibility_reg = (1 - gate.float()).mean()
            loss = render + 0.05 * depth_loss + 0.002 * visibility_reg + 1e-4 * bridge.last_residual_rms
            loss.backward()
            nonfinite_gradients = {}
            for name, parameter in bridge.named_parameters():
                if parameter.grad is None:
                    continue
                finite = torch.isfinite(parameter.grad)
                bad = int((~finite).sum().item())
                if bad:
                    nonfinite_gradients[name] = {
                        "bad": bad,
                        "total": parameter.grad.numel(),
                    }
                    # gsplat can emit isolated NaN/Inf derivatives for
                    # degenerate projected Gaussians.  They must never enter
                    # Adam's moments; keep all finite derivatives from the
                    # same tensor and zero only invalid elements.
                    parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            grad = float(torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0))
            optimizer.step()
            row = {"epoch": epoch, "index": index, "scene": sample["scene"], "slot": slot,
                   "loss": float(loss), "render_loss": float(render), "depth_loss": float(depth_loss),
                   "visibility_mean": float(gate.mean()), "grad_norm": grad,
                   "nonfinite_gradients": nonfinite_gradients, "metrics": metrics}
            trace.append(row)
            print(json.dumps(row), flush=True)
            del pred, gate, loss
            torch.cuda.empty_cache()

        # Validation uses all seven generated views.
        bridge.eval(); model.eval(); rows = []
        with torch.inference_mode():
            for sample in val:
                pred, gate = run_model(model, bridge, sample, device, dtype, True)
                _, metrics = render_loss(pred, sample["gt"].to(device), sample["valid"], list(range(1, 8)), pose_encoding_to_extri_intri, render_gaussians, ssim, args.render_size)
                pred_depth = pred["depth"][0].float()
                for item in metrics:
                    item.update(depth_metrics(pred_depth[item["slot"]], sample["teacher_depth_raw"][item["slot"]]))
                rows.extend({"scene": sample["scene"], **m} for m in metrics)
                del pred, gate
                torch.cuda.empty_cache()
        score = float(np.mean([r["psnr"] for r in rows]))
        epoch_metrics = {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("psnr", "ssim", "l1", "depth_absrel", "depth_bad20", "depth_silog")
        }
        torch.save({"bridge": bridge.state_dict(), "channels": channels, "epoch": epoch, "protocol": protocol}, args.output / f"epoch_{epoch}.pt")
        if score > best:
            best = score
            best_metrics = epoch_metrics
            torch.save({"bridge": bridge.state_dict(), "channels": channels, "epoch": epoch, "protocol": protocol}, args.output / "best.pt")
        (args.output / f"validation_epoch_{epoch}.json").write_text(json.dumps({"metrics": epoch_metrics, "rows": rows}, indent=2) + "\n")

    summary = {
        "stage": 1, "method": "MVGen 1/8,1/16,1/32 cross-attention -> VGGT 4/11/17/23 + camera + visibility",
        "frozen": ["MVGen", "VGGT"], "trainable": ["multiscale bridge", "layer gates", "camera gate", "visibility head"],
        "train_scenes": len(train_scenes), "validation_scenes": len(val_scenes),
        "blind_test_scenes": len(protocol["blind_test_scenes"]), "blind_test_used_during_training": False,
        "validation_baseline_psnr": baseline_psnr,
        "best_validation_psnr": best,
        "validation_psnr_delta": best - baseline_psnr,
        "validation_baseline_metrics": baseline_metrics,
        "best_validation_metrics": best_metrics,
        "validation_metric_delta": {
            key: best_metrics[key] - baseline_metrics[key] for key in baseline_metrics
        },
        "checkpoint": checkpoint,
        "trainable_parameters": sum(p.numel() for p in bridge.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    (args.output / "training_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
