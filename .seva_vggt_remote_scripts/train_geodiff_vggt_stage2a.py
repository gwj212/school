#!/usr/bin/env python3
"""Stage-2A controlled unfreezing for GeoDiff-VGGT.

This stage starts from the stage-1 epoch-2 bridge, keeps MVGen frozen, and
unfreezes only VGGT's last four global blocks plus the camera/depth/Gaussian
prediction path.  The sealed blind split is never loaded by this program.
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
import torch.nn.functional as F
from PIL import Image

from train_geodiff_vggt_stage1 import (
    GeoDiffBridge,
    MODEL_CANVAS,
    PATCH_SIZE,
    depth_metrics,
    load_sample,
    normalized_log_depth,
    run_model,
    scene_gaussians,
)


METRIC_KEYS = ("psnr", "ssim", "l1", "depth_absrel", "depth_bad20", "depth_silog")


def configure_trainables(model, bridge):
    model.requires_grad_(False)
    bridge.requires_grad_(True)

    # The experiment specification asks for the final four *global* blocks,
    # not the interleaved frame blocks.
    for block in model.aggregator.global_blocks[-4:]:
        block.requires_grad_(True)
    for head_name in ("camera_head", "depth_head", "dpt_feature_head", "gaussian_head"):
        head = getattr(model, head_name, None)
        if head is not None:
            head.requires_grad_(True)

    groups = {
        "bridge": list(bridge.parameters()),
        "vggt_global_last4": [
            parameter
            for block in model.aggregator.global_blocks[-4:]
            for parameter in block.parameters()
        ],
        "vggt_heads": [
            parameter
            for name in ("camera_head", "depth_head", "dpt_feature_head", "gaussian_head")
            for parameter in getattr(model, name).parameters()
            if getattr(model, name, None) is not None
        ],
    }
    expected = {
        id(parameter)
        for module in (model, bridge)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    actual = {id(parameter) for values in groups.values() for parameter in values}
    if expected != actual:
        raise RuntimeError(
            f"optimizer grouping mismatch: missing={len(expected-actual)} extra={len(actual-expected)}"
        )
    return groups


def trainable_manifest(model, bridge):
    rows = []
    for prefix, module in (("bridge", bridge), ("vggt", model)):
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                rows.append({"name": f"{prefix}.{name}", "parameters": parameter.numel()})
    return rows


def gradient_group_stats(groups):
    result = {}
    for name, parameters in groups.items():
        norms = []
        finite = True
        for parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            finite = finite and bool(torch.isfinite(gradient).all())
            norms.append(float(gradient.float().norm()))
        result[name] = {
            "tensors_total": len(parameters),
            "tensors_with_grad": len(norms),
            "tensors_with_nonzero_grad": sum(value > 0.0 for value in norms),
            "grad_norm_l2": float(sum(value * value for value in norms) ** 0.5),
            "all_gradients_finite_after_sanitize": finite,
        }
    return result


def attach_teacher(sample, cache_root):
    payload = torch.load(
        cache_root / sample["scene"] / "vggt_real8_teacher_depth.pt", map_location="cpu"
    )
    if not isinstance(payload, dict) or not {"raw", "normalized"} <= set(payload):
        raise RuntimeError(f"missing raw/normalized teacher cache for {sample['scene']}")
    sample["teacher_depth_raw"] = payload["raw"]
    sample["teacher_depth"] = payload["normalized"]


def attach_object_masks(sample, manifest_dir):
    manifest = json.loads((manifest_dir / f"{sample['scene']}.json").read_text())
    masks = []
    for view in manifest["views"][:8]:
        image = Image.open(view["mask"]).convert("L")
        width, height = image.size
        if width >= height:
            resized_width = MODEL_CANVAS
            resized_height = round(height * MODEL_CANVAS / width / PATCH_SIZE) * PATCH_SIZE
        else:
            resized_height = MODEL_CANVAS
            resized_width = round(width * MODEL_CANVAS / height / PATCH_SIZE) * PATCH_SIZE
        image = image.resize((resized_width, resized_height), Image.Resampling.NEAREST)
        canvas = np.zeros((MODEL_CANVAS, MODEL_CANVAS), dtype=np.bool_)
        x0 = (MODEL_CANVAS - resized_width) // 2
        y0 = (MODEL_CANVAS - resized_height) // 2
        canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = np.asarray(image) > 127
        masks.append(torch.from_numpy(canvas))
    sample["object_mask"] = torch.stack(masks)


def subset_sample(sample, indices):
    index = torch.as_tensor(indices, dtype=torch.long)
    return {
        "scene": sample["scene"],
        "input": sample["input"].index_select(0, index),
        "gt": sample["gt"].index_select(0, index),
        "features": {key: value.index_select(0, index) for key, value in sample["features"].items()},
        "valid": sample["valid"].index_select(0, index),
        "object_mask": sample["object_mask"].index_select(0, index),
        "teacher_depth_raw": sample["teacher_depth_raw"].index_select(0, index),
        "teacher_depth": sample["teacher_depth"].index_select(0, index),
    }


def gaussian_regularizers(prediction, object_mask, opacity_floor, opacity_ceiling, coverage_floor):
    opacity = prediction["gaussians"]["opacity"][0, :, :, 0].float()
    mask = object_mask.flatten(1).to(opacity.device)
    if not mask.any():
        # A small number of CO3D-completion scenes contain all-zero mask PNGs.
        # Keep their RGB/depth supervision, but do not invent foreground for
        # the object-specific opacity regularizer.
        zero = opacity.sum() * 0.0
        return None, None, zero, zero, False
    selected = opacity[mask]
    mean_opacity = selected.mean()
    soft_coverage = torch.sigmoid((selected - 0.10) / 0.02).mean()
    # Linear hinge penalties remain active near the boundary; the squared
    # version was too weak when opacity was already close to the floor.
    opacity_loss = F.relu(opacity_floor - mean_opacity)
    opacity_loss = opacity_loss + F.relu(mean_opacity - opacity_ceiling)
    coverage_loss = F.relu(coverage_floor - soft_coverage)
    return mean_opacity, soft_coverage, opacity_loss, coverage_loss, True


def render_loss_with_alpha(
    prediction, targets, object_masks, valid, slots, pose_fn, ssim_fn, size
):
    """Render RGB and true target-view alpha in one differentiable rasterization."""
    from gsplat import rasterization

    extrinsics, intrinsics = pose_fn(prediction["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS))
    extrinsics = extrinsics[0].float()
    intrinsics = intrinsics[0].float()
    scale = intrinsics.new_tensor(
        [[size / MODEL_CANVAS, 0.0, 0.0], [0.0, size / MODEL_CANVAS, 0.0], [0.0, 0.0, 1.0]]
    )
    intrinsics = torch.matmul(scale.unsqueeze(0), intrinsics)
    gaussians = scene_gaussians(prediction, valid)
    bottom = extrinsics.new_tensor([[0.0, 0.0, 0.0, 1.0]])
    losses, rows, alpha_means, alpha_coverages = [], [], [], []
    for slot in slots:
        target = F.interpolate(
            targets[slot : slot + 1].float(),
            (size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        viewmat = torch.cat([extrinsics[slot], bottom], dim=0).unsqueeze(0)
        rendered, alpha, _ = rasterization(
            means=gaussians["xyz"],
            quats=gaussians["rotation"],
            scales=gaussians["scale"],
            opacities=gaussians["opacity"].squeeze(-1),
            colors=gaussians["color"],
            viewmats=viewmat,
            Ks=intrinsics[slot : slot + 1],
            width=size,
            height=size,
            sh_degree=None,
            near_plane=0.01,
            far_plane=1e10,
        )
        rgb = rendered[0, :, :, :3].permute(2, 0, 1).clamp(0, 1)
        alpha_map = alpha[0, :, :, 0].clamp(0, 1)
        mask = F.interpolate(
            object_masks[slot : slot + 1, None].float().to(alpha_map.device),
            (size, size),
            mode="nearest",
        )[0, 0]
        mask_pixels = mask.sum()
        if mask_pixels > 0:
            alpha_mean = (alpha_map * mask).sum() / mask_pixels
            alpha_coverage = (
                torch.sigmoid((alpha_map - 0.10) / 0.02) * mask
            ).sum() / mask_pixels
            alpha_means.append(alpha_mean)
            alpha_coverages.append(alpha_coverage)
            alpha_mean_value = float(alpha_mean)
            alpha_coverage_value = float(alpha_coverage)
        else:
            # Missing annotation is neutral for the loss and excluded from
            # validation aggregation; it is not treated as zero coverage.
            alpha_mean_value = None
            alpha_coverage_value = None

        l1 = F.l1_loss(rgb, target)
        dssim = 1.0 - ssim_fn(rgb[None], target[None])
        losses.append(0.8 * l1 + 0.2 * dssim)
        with torch.no_grad():
            mse = F.mse_loss(rgb, target).item()
            rows.append(
                {
                    "slot": slot,
                    "psnr": -10.0 * math.log10(max(mse, 1e-10)),
                    "ssim": float((1.0 - dssim).item()),
                    "l1": float(l1.item()),
                    "rendered_object_alpha_mean": alpha_mean_value,
                    "rendered_object_alpha_soft_coverage": alpha_coverage_value,
                }
            )
    neutral = prediction["gaussians"]["opacity"].sum() * 0.0 + 1.0
    return (
        torch.stack(losses).mean(),
        rows,
        torch.stack(alpha_means).mean() if alpha_means else neutral,
        torch.stack(alpha_coverages).mean() if alpha_coverages else neutral,
    )


def summarize(rows):
    result = {}
    for key in METRIC_KEYS + (
        "object_opacity_mean",
        "object_opacity_soft_coverage_proxy",
        "rendered_object_alpha_mean",
        "rendered_object_alpha_soft_coverage",
    ):
        values = [row[key] for row in rows if row.get(key) is not None]
        finite = [value for value in values if np.isfinite(value)]
        result[key] = float(np.mean(finite)) if finite else None
    return result


def selection_score(
    metrics, reference, retention_ratio
):
    # Reject checkpoints that trade away the stage-1 depth/SSIM gains or
    # collapse opacity.  Among eligible checkpoints, optimize a balanced,
    # dimensionless delta relative to the epoch-2 initialization.
    eligible = (
        metrics["depth_absrel"] <= reference["depth_absrel"] * 1.01
        and metrics["depth_bad20"] <= reference["depth_bad20"] * 1.01
        and metrics["depth_silog"] <= reference["depth_silog"] * 1.01
        and metrics["ssim"] >= reference["ssim"] - 0.01
        and metrics["object_opacity_mean"]
        >= reference["object_opacity_mean"] * retention_ratio
        and metrics["rendered_object_alpha_mean"]
        >= reference["rendered_object_alpha_mean"] * retention_ratio
        and metrics["rendered_object_alpha_soft_coverage"]
        >= reference["rendered_object_alpha_soft_coverage"] * retention_ratio
    )
    score = (
        (metrics["psnr"] - reference["psnr"])
        + 2.0 * (metrics["ssim"] - reference["ssim"])
        - 2.0 * (metrics["l1"] - reference["l1"])
        - 2.0 * (metrics["depth_absrel"] - reference["depth_absrel"])
        - (metrics["depth_bad20"] - reference["depth_bad20"])
        - (metrics["depth_silog"] - reference["depth_silog"])
    )
    return bool(eligible), float(score)


def evaluate(model, bridge, samples, device, dtype, pose_fn, ssim_fn, size):
    rows = []
    model.eval()
    bridge.eval()
    with torch.inference_mode():
        for sample in samples:
            prediction, gate = run_model(model, bridge, sample, device, dtype, True)
            _, image_rows, _, _ = render_loss_with_alpha(
                prediction,
                sample["gt"].to(device),
                sample["object_mask"],
                sample["valid"],
                list(range(1, len(sample["gt"]))),
                pose_fn,
                ssim_fn,
                size,
            )
            depth = prediction["depth"][0].float()
            opacity, coverage, _, _, object_mask_available = gaussian_regularizers(
                prediction, sample["object_mask"], 0.0, 1.0, 0.0
            )
            for item in image_rows:
                item.update(depth_metrics(depth[item["slot"]], sample["teacher_depth_raw"][item["slot"]]))
                rows.append(
                    {
                        "scene": sample["scene"],
                        **item,
                        "object_opacity_mean": float(opacity) if object_mask_available else None,
                        "object_opacity_soft_coverage_proxy": float(coverage) if object_mask_available else None,
                        "object_mask_available": object_mask_available,
                        "visibility_gate_mean": float(gate.float().mean()),
                    }
                )
            del prediction, gate
            torch.cuda.empty_cache()
    return {"metrics": summarize(rows), "rows": rows}


def save_checkpoint(path, model, bridge, epoch, protocol, manifest, metrics, score):
    torch.save(
        {
            "stage": "2A",
            "epoch": epoch,
            "protocol": protocol,
            "bridge": bridge.state_dict(),
            "channels": {
                key: module.feature[0].in_channels
                for key, module in (
                    ("scale_1_8", bridge.adapters["4"]),
                    ("scale_1_16", bridge.adapters["11"]),
                    ("scale_1_32", bridge.adapters["17"]),
                )
            },
            # A prefix list is robust to parameter/buffer differences when
            # reloading the controlled-unfreeze state for inference.
            "vggt_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
                if name.startswith((
                    "aggregator.global_blocks.20.",
                    "aggregator.global_blocks.21.",
                    "aggregator.global_blocks.22.",
                    "aggregator.global_blocks.23.",
                    "camera_head.",
                    "depth_head.",
                    "dpt_feature_head.",
                    "gaussian_head.",
                ))
            },
            "trainable_manifest": manifest,
            "validation_metrics": metrics,
            "selection_score": score,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bridge-lr", type=float, default=5e-5)
    parser.add_argument("--pretrained-lr", type=float, default=3e-6)
    parser.add_argument("--bridge-clip-norm", type=float, default=1.0)
    parser.add_argument("--global-clip-norm", type=float, default=1.0)
    parser.add_argument("--heads-clip-norm", type=float, default=1.0)
    parser.add_argument("--render-size", type=int, default=192)
    parser.add_argument("--train-views", type=int, default=5)
    parser.add_argument("--opacity-floor", type=float, default=0.08)
    parser.add_argument("--selection-retention-ratio", type=float, default=0.95)
    parser.add_argument("--opacity-ceiling", type=float, default=0.75)
    parser.add_argument("--coverage-floor", type=float, default=0.30)
    parser.add_argument("--opacity-weight", type=float, default=0.05)
    parser.add_argument("--coverage-weight", type=float, default=0.05)
    parser.add_argument("--rendered-alpha-mean-floor", type=float, default=0.10)
    parser.add_argument("--rendered-alpha-coverage-floor", type=float, default=0.20)
    parser.add_argument("--rendered-alpha-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-val-scenes", type=int)
    parser.add_argument("--skip-checkpoint-save", action="store_true")
    args = parser.parse_args()
    if args.train_views < 2 or args.train_views > 8:
        raise ValueError("--train-views must be in [2,8]")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    protocol = json.loads(args.protocol.read_text())
    train_scenes = protocol["train_scenes"]
    val_scenes = protocol["validation_scenes"]
    if args.max_train_scenes is not None:
        train_scenes = train_scenes[: args.max_train_scenes]
    if args.max_val_scenes is not None:
        val_scenes = val_scenes[: args.max_val_scenes]
    # There is intentionally no code path that reads blind_test_scenes.

    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.gs_optimizer import ssim
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    train = [
        load_sample(scene, args.manifest_root / "train", args.cache_root, load_and_preprocess_images)
        for scene in train_scenes
    ]
    val = [
        load_sample(scene, args.manifest_root / "validation", args.cache_root, load_and_preprocess_images)
        for scene in val_scenes
    ]
    for sample in train + val:
        attach_teacher(sample, args.cache_root)
    for sample in train:
        attach_object_masks(sample, args.manifest_root / "train")
    for sample in val:
        attach_object_masks(sample, args.manifest_root / "validation")

    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu")
    if int(stage1.get("epoch", -1)) != 2:
        raise RuntimeError("Stage-2A must start from the audited stage-1 epoch_2.pt")
    bridge = GeoDiffBridge(stage1["channels"]).to(device)
    bridge.load_state_dict(stage1["bridge"])

    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    reconstruction_checkpoint = rp.load_reconstruction_checkpoint(
        model, Path("/root/vggt/ckpts/ckpt.pth")
    )
    model.point_head = None
    model.track_head = None
    groups = configure_trainables(model, bridge)
    model.to(device)
    model._release_depth = True
    model._allow_dpt_token_grad = True
    model._depth_grad_checkpoint = True
    model.camera_head._grad_checkpoint = True

    manifest = trainable_manifest(model, bridge)
    counts = {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in groups.items()
    }
    (args.output / "trainable_manifest.json").write_text(
        json.dumps({"groups": counts, "parameters": manifest}, indent=2) + "\n"
    )

    initial = evaluate(
        model, bridge, val, device, dtype,
        pose_encoding_to_extri_intri, ssim, args.render_size,
    )
    (args.output / "validation_initial.json").write_text(json.dumps(initial, indent=2) + "\n")
    reference = initial["metrics"]
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["bridge"], "lr": args.bridge_lr, "name": "bridge"},
            {"params": groups["vggt_global_last4"], "lr": args.pretrained_lr, "name": "vggt_global_last4"},
            {"params": groups["vggt_heads"], "lr": args.pretrained_lr, "name": "vggt_heads"},
        ],
        weight_decay=0.01,
    )

    trace = []
    best_score = -float("inf")
    best_epoch = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train)
        model.train()
        bridge.train()
        for index, full_sample in enumerate(train):
            generated = list(range(1, 8))
            offset = (index + epoch * 3) % len(generated)
            rotated = generated[offset:] + generated[:offset]
            indices = [0] + rotated[: args.train_views - 1]
            sample = subset_sample(full_sample, indices)
            optimizer.zero_grad(set_to_none=True)
            prediction, gate = run_model(model, bridge, sample, device, dtype, True)
            render, _, rendered_alpha_mean, rendered_alpha_coverage = render_loss_with_alpha(
                prediction,
                sample["gt"].to(device),
                sample["object_mask"],
                sample["valid"],
                [0, 1 + (index % (len(indices) - 1))],
                pose_encoding_to_extri_intri,
                ssim,
                args.render_size,
            )
            pred_depth = normalized_log_depth(prediction["depth"][0])
            teacher = sample["teacher_depth"].to(device).float()
            depth_loss = F.smooth_l1_loss(pred_depth, teacher, beta=0.1)
            opacity, coverage, opacity_loss, coverage_loss, object_mask_available = gaussian_regularizers(
                prediction,
                sample["object_mask"],
                args.opacity_floor,
                args.opacity_ceiling,
                args.coverage_floor,
            )
            visibility_reg = (1.0 - gate.float()).mean()
            rendered_alpha_loss = F.relu(
                args.rendered_alpha_mean_floor - rendered_alpha_mean
            ) + F.relu(
                args.rendered_alpha_coverage_floor - rendered_alpha_coverage
            )
            loss = (
                render
                + 0.05 * depth_loss
                + args.opacity_weight * opacity_loss
                + args.coverage_weight * coverage_loss
                + args.rendered_alpha_weight * rendered_alpha_loss
                + 0.002 * visibility_reg
                + 1e-4 * bridge.last_residual_rms
            )
            loss.backward()
            nonfinite = {}
            for name, parameter in list(bridge.named_parameters()) + list(model.named_parameters()):
                if parameter.grad is None:
                    continue
                bad = int((~torch.isfinite(parameter.grad)).sum())
                if bad:
                    nonfinite[name] = bad
                    parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            group_gradients = gradient_group_stats(groups)
            grad_norm = float(
                sum(
                    values["grad_norm_l2"] ** 2
                    for values in group_gradients.values()
                )
                ** 0.5
            )
            group_clip_limits = {
                "bridge": args.bridge_clip_norm,
                "vggt_global_last4": args.global_clip_norm,
                "vggt_heads": args.heads_clip_norm,
            }
            group_clip_returned_norm = {
                name: float(torch.nn.utils.clip_grad_norm_(parameters, group_clip_limits[name]))
                for name, parameters in groups.items()
            }
            optimizer.step()
            row = {
                "epoch": epoch,
                "index": index,
                "scene": sample["scene"],
                "view_indices": indices,
                "loss": float(loss),
                "render_loss": float(render),
                "depth_loss": float(depth_loss),
                "opacity_loss": float(opacity_loss),
                "coverage_loss": float(coverage_loss),
                "rendered_alpha_loss": float(rendered_alpha_loss),
                "object_opacity_mean": float(opacity) if object_mask_available else None,
                "object_opacity_soft_coverage_proxy": float(coverage) if object_mask_available else None,
                "object_mask_available": object_mask_available,
                "rendered_object_alpha_mean": float(rendered_alpha_mean),
                "rendered_object_alpha_soft_coverage": float(rendered_alpha_coverage),
                "visibility_gate_mean": float(gate.float().mean()),
                "grad_norm": grad_norm,
                "nonfinite_gradients": nonfinite,
                "gradient_groups_preclip": group_gradients,
                "gradient_group_clip_limits": group_clip_limits,
                "gradient_group_clip_returned_norm": group_clip_returned_norm,
            }
            trace.append(row)
            print(json.dumps(row), flush=True)
            del prediction, gate, loss
            torch.cuda.empty_cache()

        validation = evaluate(
            model, bridge, val, device, dtype,
            pose_encoding_to_extri_intri, ssim, args.render_size,
        )
        eligible, score = selection_score(
            validation["metrics"],
            reference,
            args.selection_retention_ratio,
        )
        validation.update({"eligible": eligible, "selection_score": score})
        (args.output / f"validation_epoch_{epoch}.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )
        if not args.skip_checkpoint_save:
            save_checkpoint(
                args.output / f"epoch_{epoch}.pt", model, bridge, epoch,
                protocol, manifest, validation["metrics"], score,
            )
        if eligible and score > best_score:
            best_score = score
            best_epoch = epoch
            if not args.skip_checkpoint_save:
                save_checkpoint(
                    args.output / "best.pt", model, bridge, epoch,
                    protocol, manifest, validation["metrics"], score,
                )

    summary = {
        "stage": "2A",
        "initialization": str(args.stage1_checkpoint),
        "frozen": ["MVGen", "VGGT frame blocks", "VGGT global blocks 0-19", "VGGT patch embed"],
        "trainable": ["stage-1 bridge/gates/visibility", "VGGT global blocks 20-23", "camera/depth/DPT-feature/Gaussian heads"],
        "train_scenes": len(train),
        "validation_scenes": len(val),
        "blind_test_used": False,
        "train_views_per_step": args.train_views,
        "optimizer_parameter_counts": counts,
        "bridge_lr": args.bridge_lr,
        "pretrained_lr": args.pretrained_lr,
        "gradient_clip_norms": {
            "bridge": args.bridge_clip_norm,
            "vggt_global_last4": args.global_clip_norm,
            "vggt_heads": args.heads_clip_norm,
        },
        "constraint_thresholds": {
            "training_object_opacity_floor": args.opacity_floor,
            "checkpoint_opacity_alpha_retention_ratio": args.selection_retention_ratio,
            "source_opacity_proxy_coverage_floor": args.coverage_floor,
            "rendered_alpha_mean_floor": args.rendered_alpha_mean_floor,
            "rendered_alpha_soft_coverage_floor": args.rendered_alpha_coverage_floor,
        },
        "validation_initial": reference,
        "best_epoch": best_epoch,
        "best_selection_score": None if best_epoch is None else best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "reconstruction_checkpoint": reconstruction_checkpoint,
        "coverage_metric_note": (
            "object_opacity_soft_coverage_proxy is a source-pixel Gaussian-opacity proxy. "
            "rendered_object_alpha_mean and rendered_object_alpha_soft_coverage are computed "
            "from gsplat alpha at each held-out target camera inside the CO3D object mask."
        ),
    }
    (args.output / "training_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
