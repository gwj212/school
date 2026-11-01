#!/usr/bin/env python3
"""Train MVGen attention LoRA from frozen Stage2A VGGT depth consistency."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import zlib
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

import smoke_stage2b_mvgen_lora_live as common


def prepare_mvgen(manifest, cached_depth, config, model_dir):
    views = manifest["views"][:8]
    source = views[0]
    w2cs = np.stack([common.pad_w2c(view["extrinsic"]) for view in views])
    c2ws = np.linalg.inv(w2cs)
    depth = np.load(cached_depth).astype(np.float32)
    positions = c2ws[:, :3, 3]
    span = float(np.max(positions.max(0) - positions.min(0)))
    pose_scale = float(config.camera_longest_side / span) if span > config.camera_longest_side else 1.0
    w2cs[:, :3, 3] *= pose_scale
    depth *= pose_scale

    image = Image.open(source["image"]).convert("RGB")
    original_width, original_height = image.size
    height, width = common.choose_hw(model_dir / "ratio_set.json", original_height, original_width)
    resample = Image.Resampling.LANCZOS if height < original_height else Image.Resampling.BICUBIC
    resized = image.resize((width, height), resample)
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    source_tensor = Compose([ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])(resized)
    images = torch.cat([source_tensor[None], torch.zeros((7, 3, height, width))], 0).cuda()
    intrinsics = []
    for view in views:
        intrinsic = np.asarray(view["intrinsic"], dtype=np.float32).copy()
        intrinsic[0] *= width / view["width"]
        intrinsic[1] *= height / view["height"]
        intrinsics.append(intrinsic)
    intrinsics = torch.from_numpy(np.stack(intrinsics)).cuda()
    extrinsics = torch.from_numpy(w2cs).cuda()
    depths = torch.cat([torch.from_numpy(depth)[None, None], torch.zeros((7, 1, height, width))], 0).cuda()
    return views, images, intrinsics, extrinsics, depths, height, width


def lora_state(unet):
    return {
        name: value.detach().cpu()
        for name, value in unet.state_dict().items()
        if "lora_layer" in name
    }


def save_checkpoint(path, unet, bridge, model, optimizer, trace, completed, args):
    temporary = path.with_suffix(".tmp")
    payload = {
        "lora": lora_state(unet),
        "optimizer": optimizer.state_dict(),
        "trace": trace,
        "completed_train_steps": completed,
        "config": vars(args),
        "blind_accessed": False,
    }
    if args.joint_vggt:
        payload["bridge"] = bridge.state_dict()
        payload["vggt_trainable"] = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage2a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--vggt-views", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--joint-vggt", action="store_true")
    parser.add_argument("--bridge-lr", type=float, default=2e-5)
    parser.add_argument("--vggt-lr", type=float, default=2.5e-7)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--validation-scenes", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    for key in ("protocol", "manifest_root", "cache_root", "stage2a", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.resume is not None:
        args.resume = args.resume.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    started = time.perf_counter()

    os.chdir(common.MVGEN)
    sys.path[:0] = [str(common.MVGEN), "/root/vggt", str(common.ROOT / "bridge")]
    from diffusers import AutoencoderKL
    from easydict import EasyDict
    from omegaconf import OmegaConf
    from my_diffusers.models import UNet2DConditionModel
    from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
    from src.modules.schedulers import get_diffusion_scheduler
    from train_geodiff_vggt_stage1 import GeoDiffBridge, normalized_log_depth
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    protocol = json.loads(args.protocol.read_text())
    train_scenes = list(protocol["train_scenes"])
    if args.max_train_scenes is not None:
        train_scenes = train_scenes[:args.max_train_scenes]
    validation_scenes = list(protocol["validation_scenes"][:args.validation_scenes])
    model_dir = common.MVGEN / "check_points/pretrained_model"
    config = EasyDict(OmegaConf.load(model_dir / "config.yaml"))
    config.pretrained_model_name_or_path = str(common.BASE)
    config.nframe = 8

    vae = AutoencoderKL.from_pretrained(common.BASE, subfolder="vae", local_files_only=True).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(
        common.BASE, subfolder="unet", rank=0, model_cfg=config.model_cfg,
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True, local_files_only=True,
    )
    unet.load_state_dict(torch.load(model_dir / "ema_unet.pt", map_location="cpu"))
    lora_parameters, lora_modules = common.attach_lora(unet, args.rank)
    vae.to("cuda", dtype=torch.float16)
    unet.to("cuda", dtype=torch.float16)
    for module in unet.modules():
        layer = getattr(module, "lora_layer", None)
        if layer is not None:
            layer.to("cuda", dtype=torch.float32)
    lora_parameters = [parameter for name, parameter in unet.named_parameters() if "lora_layer" in name]
    collector = common.LiveCollector(unet)
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        common.BASE, vae=vae, unet=unet,
        scheduler=get_diffusion_scheduler(config, name="DDIM"), safety_checker=None,
        torch_dtype=torch.float16, local_files_only=True,
    ).to("cuda")
    checkpoint = torch.load(args.stage2a, map_location="cpu")
    bridge = GeoDiffBridge(checkpoint["channels"])
    bridge.load_state_dict(checkpoint["bridge"])
    bridge.requires_grad_(False).eval().cuda()
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    missing, unexpected = model.load_state_dict(checkpoint["vggt_state"], strict=False)
    model.point_head = None; model.track_head = None
    model.camera_head = None; model.dpt_feature_head = None; model.gaussian_head = None
    model.requires_grad_(False)
    model._release_depth = True
    model._allow_dpt_token_grad = True
    if args.joint_vggt:
        bridge.requires_grad_(True).train()
        for block in model.aggregator.global_blocks[-4:]:
            block.requires_grad_(True)
        model.depth_head.requires_grad_(True)
        model._depth_grad_checkpoint = True
        model.cuda().train()
        train_groups = {
            "lora": lora_parameters,
            "bridge": list(bridge.parameters()),
            "vggt_global_last4": [
                parameter
                for block in model.aggregator.global_blocks[-4:]
                for parameter in block.parameters()
            ],
            "depth_head": list(model.depth_head.parameters()),
        }
        optimizer = torch.optim.AdamW([
            {"params": train_groups["lora"], "lr": args.lr, "name": "lora"},
            {"params": train_groups["bridge"], "lr": args.bridge_lr, "name": "bridge"},
            {"params": train_groups["vggt_global_last4"], "lr": args.vggt_lr, "name": "vggt_global_last4"},
            {"params": train_groups["depth_head"], "lr": args.vggt_lr, "name": "depth_head"},
        ], weight_decay=args.weight_decay)
    else:
        model.cuda().eval()
        train_groups = {"lora": lora_parameters}
        optimizer = torch.optim.AdamW(lora_parameters, lr=args.lr, weight_decay=args.weight_decay)
    del checkpoint

    trace = []
    completed = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu")
        unet.load_state_dict(resume["lora"], strict=False)
        if args.joint_vggt:
            bridge.load_state_dict(resume["bridge"])
            model.load_state_dict(resume["vggt_trainable"], strict=False)
        optimizer.load_state_dict(resume["optimizer"])
        trace = resume.get("trace", [])
        completed = int(resume.get("completed_train_steps", 0))
        del resume

    def run_scene(scene, split, training):
        nonlocal completed
        if args.joint_vggt:
            model.train(training)
            bridge.train(training)
        manifest_path = args.manifest_root / split / f"{scene}.json"
        manifest = json.loads(manifest_path.read_text())
        depth_path = args.cache_root / scene / "mvgen_source_depth.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        views, images, intrinsics, extrinsics, depths, height, width = prepare_mvgen(
            manifest, depth_path, config, model_dir
        )
        scene_seed = args.seed + (zlib.crc32(scene.encode()) & 0x7FFFFFFF)
        generator = torch.Generator(device="cpu").manual_seed(scene_seed)

        def detach_early(_pipeline, step_index, _timestep, callback_kwargs):
            if step_index < args.steps - 1:
                callback_kwargs["latents"] = callback_kwargs["latents"].detach()
            return callback_kwargs

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        context = nullcontext() if training else torch.no_grad()
        with context, torch.autocast("cuda", dtype=torch.float16):
            result = pipeline.__call__.__wrapped__(
                pipeline,
                images=images, nframe=8, cond_num=1, key_rescale=None,
                height=height, width=width, intrinsics=intrinsics, extrinsics=extrinsics,
                num_inference_steps=args.steps, guidance_scale=2.0, output_type="latent",
                config=copy.deepcopy(config), tag=["custom"] * 8, class_label=0,
                depth=depths, vae=pipeline.vae, generator=generator,
                callback_on_step_end=detach_early if training else None,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        if training and not all(value.requires_grad for value in collector.latest.values()):
            raise RuntimeError(f"{scene}: detached live MVGen features")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            decoded = pipeline.vae.decode(
                result.images.detach() / pipeline.vae.config.scaling_factor,
                return_dict=False,
            )[0]
            decoded_vggt = common.vggt_pad_tensor((decoded / 2 + 0.5).clamp(0, 1)).cpu()
        source_vggt = load_and_preprocess_images([views[0]["image"]], mode="pad")
        vggt_input = torch.cat([source_vggt, decoded_vggt[1:args.vggt_views]], dim=0)
        teacher = torch.load(args.cache_root / scene / "vggt_real8_teacher_depth.pt", map_location="cpu")
        teacher_normalized = teacher["normalized"][:args.vggt_views].cuda().float()
        bridge.current_features = {
            key: value[:args.vggt_views][None] for key, value in collector.latest.items()
        }
        handle = model.aggregator.register_forward_hook(bridge.hook)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(vggt_input[None].cuda())
        handle.remove()
        predicted_depth = normalized_log_depth(prediction["depth"][0])
        loss = F.smooth_l1_loss(predicted_depth, teacher_normalized, beta=0.1)
        grad_norm = 0.0
        if training:
            loss.backward()
            group_stats = {}
            for name, parameters in train_groups.items():
                grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
                finite = bool(grads) and all(bool(torch.isfinite(gradient).all()) for gradient in grads)
                if not finite:
                    raise RuntimeError(f"{scene}: non-finite or missing gradients in {name}")
                group_stats[name] = math.sqrt(sum(float(gradient.float().square().sum()) for gradient in grads))
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            grad_norm = group_stats["lora"]
            optimizer.step()
            completed += 1
        row = {
            "scene": scene, "split": split, "training": training,
            "loss": float(loss.detach()), "grad_norm": grad_norm,
            "group_grad_norms": group_stats if training else {},
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "step": completed, "blind_accessed": False,
        }
        print("STAGE2B " + json.dumps(row), flush=True)
        del result, decoded, decoded_vggt, prediction, predicted_depth, loss, teacher
        collector.latest.clear()
        torch.cuda.empty_cache()
        return row

    validation_before = [run_scene(scene, "validation", False) for scene in validation_scenes]
    validation_history = [{
        "step": completed,
        "mean_loss": float(np.mean([row["loss"] for row in validation_before])),
        "rows": validation_before,
    }]
    best_validation_loss = validation_history[0]["mean_loss"]
    best_step = completed
    save_checkpoint(args.output / "best.pt", unet, bridge, model, optimizer, trace, completed, args)
    ordered = train_scenes * args.epochs
    ordered = ordered[completed:]
    for scene in ordered:
        row = run_scene(scene, "train", True)
        trace.append(row)
        if completed % args.save_every == 0:
            save_checkpoint(args.output / "latest.pt", unet, bridge, model, optimizer, trace, completed, args)
            save_checkpoint(args.output / f"step_{completed:04d}.pt", unet, bridge, model, optimizer, trace, completed, args)
            (args.output / "training_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        if completed % args.validate_every == 0:
            rows = [run_scene(name, "validation", False) for name in validation_scenes]
            mean_loss = float(np.mean([item["loss"] for item in rows]))
            validation_history.append({"step": completed, "mean_loss": mean_loss, "rows": rows})
            if mean_loss < best_validation_loss:
                best_validation_loss = mean_loss
                best_step = completed
                save_checkpoint(args.output / "best.pt", unet, bridge, model, optimizer, trace, completed, args)
            (args.output / "validation_history.json").write_text(json.dumps(validation_history, indent=2) + "\n")
    save_checkpoint(args.output / "latest.pt", unet, bridge, model, optimizer, trace, completed, args)
    if validation_history[-1]["step"] == completed:
        validation_after = validation_history[-1]["rows"]
    else:
        validation_after = [run_scene(scene, "validation", False) for scene in validation_scenes]
        validation_history.append({
            "step": completed,
            "mean_loss": float(np.mean([row["loss"] for row in validation_after])),
            "rows": validation_after,
        })
    if validation_history[-1]["mean_loss"] < best_validation_loss:
        best_validation_loss = validation_history[-1]["mean_loss"]
        best_step = completed
        save_checkpoint(args.output / "best.pt", unet, bridge, model, optimizer, trace, completed, args)
    before_mean = float(np.mean([row["loss"] for row in validation_before]))
    after_mean = float(np.mean([row["loss"] for row in validation_after]))
    summary = {
        "protocol": (
            "joint MVGen LoRA + bridge + VGGT last4 global blocks + depth head; live 1/8,1/16,1/32 features"
            if args.joint_vggt else
            "final MVGen denoising step differentiable; live 1/8,1/16,1/32 features; frozen Stage2A VGGT depth teacher"
        ),
        "train_scenes": len(train_scenes), "epochs": args.epochs,
        "completed_train_steps": completed, "validation_scenes": len(validation_scenes),
        "validation_loss_before": before_mean, "validation_loss_after": after_mean,
        "validation_loss_delta": after_mean - before_mean,
        "best_validation_loss": best_validation_loss, "best_step": best_step,
        "lora_rank": args.rank, "attention_modules": lora_modules,
        "lora_parameters": sum(parameter.numel() for parameter in lora_parameters),
        "trainable_group_parameters": {
            name: sum(parameter.numel() for parameter in parameters)
            for name, parameters in train_groups.items()
        },
        "max_peak_allocated_gib": max(row["peak_allocated_gib"] for row in trace),
        "nonfinite_steps": sum(not math.isfinite(row["loss"]) for row in trace),
        "vggt_missing_after_partial_load": len(missing),
        "vggt_unexpected_after_partial_load": len(unexpected),
        "validation_before": validation_before, "validation_after": validation_after,
        "validation_history": validation_history,
        "elapsed_seconds": time.perf_counter() - started,
        "blind_accessed": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "training_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
