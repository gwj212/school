#!/usr/bin/env python3
"""Stage-2B smoke: final-step MVGen attention LoRA -> live features -> VGGT depth.

The first N-1 denoising steps are detached.  Only the final MVGen UNet call
retains a graph, so the bridge consumes the same final-step feature locations
as stage 1 while gradients reach MVGen attention LoRA on a 48 GB GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor


ROOT = Path("/root/seva_vggt")
MVGEN = ROOT / "candidates/MVGenMaster"
BASE = MVGEN / "check_points/sd2-community-stable-diffusion-2-1"


class LiveCollector:
    names = ("scale_1_8", "scale_1_16", "scale_1_32")

    def __init__(self, unet, nframe=8):
        self.nframe = nframe
        self.latest = {}
        self.handles = [
            block.register_forward_hook(self._hook(name))
            for name, block in zip(self.names, unet.down_blocks[:3])
        ]

    def _hook(self, name):
        def save(_module, _inputs, output):
            residuals = output[1] if isinstance(output, tuple) and len(output) > 1 else (output,)
            tensors = [value for value in residuals if torch.is_tensor(value) and value.ndim == 4]
            value = max(tensors, key=lambda item: item.shape[-2] * item.shape[-1])
            value = value.reshape(-1, self.nframe, *value.shape[1:])[-1]
            # Intentionally no detach/cpu: this is the Stage-2B gradient path.
            self.latest[name] = value
        return save

    def close(self):
        for handle in self.handles:
            handle.remove()


def pad_w2c(value):
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (4, 4):
        return array
    result = np.eye(4, dtype=np.float32)
    result[:3, :4] = array
    return result


def choose_hw(path: Path, height: int, width: int):
    values = json.loads(path.read_text())
    choices = {h / w: (h, w) for h, w in values}
    ratio = height / width
    return choices[min(choices, key=lambda item: abs(item - ratio))]


def source_depth(view, camera_radius):
    from depth_pro.depth_pro import create_model_and_transforms
    from depth_pro.utils import load_rgb

    model, transform = create_model_and_transforms(device=torch.device("cuda"))
    rgb, _, _ = load_rgb(view["image"])
    image = transform(rgb)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        prediction = model.infer(
            image, f_px=torch.tensor(float(view["intrinsic"][0][0]), device="cuda")
        )["depth"]
    depth = prediction.float().cpu().numpy()
    mask = np.asarray(Image.open(view["mask"]).convert("L")) > 127
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    valid = mask & np.isfinite(depth) & (depth > 0)
    if valid.sum() < 100:
        valid = np.isfinite(depth) & (depth > 0)
    depth *= float(camera_radius / max(float(np.median(depth[valid])), 1e-6))
    del model, prediction, image
    torch.cuda.empty_cache()
    return depth


def vggt_pad_tensor(images, target_size=518):
    """Match VGGT's mode='pad' preprocessing for decoded RGB tensors."""
    _, _, height, width = images.shape
    if width >= height:
        new_width = target_size
        new_height = round(height * (target_size / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (target_size / height) / 14) * 14
    images = F.interpolate(images.float(), (new_height, new_width), mode="bicubic", align_corners=False, antialias=True)
    pad_h, pad_w = target_size - new_height, target_size - new_width
    return F.pad(
        images,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        value=1.0,
    )


class LoRAWrappedLinear(nn.Module):
    """Adds a trainable low-rank residual without requiring PEFT wrappers."""

    def __init__(self, base, lora_layer):
        super().__init__()
        self.base = base
        self.lora_layer = lora_layer

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, hidden_states, scale=1.0):
        return self.base(hidden_states) + scale * self.lora_layer(hidden_states)


def attach_lora(unet, rank):
    from my_diffusers.models.attention_processor import Attention
    from my_diffusers.models.lora import LoRALinearLayer

    unet.requires_grad_(False)
    modules = 0
    targets = [
        (name, attention)
        for name, attention in unet.named_modules()
        if isinstance(attention, Attention) and name.endswith("attn1")
    ]
    for name, attention in targets:
        def wrap(projection):
            return LoRAWrappedLinear(
                projection,
                LoRALinearLayer(projection.in_features, projection.out_features, rank=rank),
            )
        attention.to_q = wrap(attention.to_q)
        attention.to_k = wrap(attention.to_k)
        attention.to_v = wrap(attention.to_v)
        attention.to_out[0] = wrap(attention.to_out[0])
        modules += 1
    parameters = [parameter for name, parameter in unet.named_parameters() if "lora_layer" in name]
    for parameter in parameters:
        parameter.requires_grad_(True)
    if modules != 16 or not parameters:
        raise RuntimeError(f"unexpected LoRA attachment: modules={modules}, tensors={len(parameters)}")
    return parameters, modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage2a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--vggt-views", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--joint-vggt", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.cache_root = args.cache_root.resolve()
    args.stage2a = args.stage2a.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    started = time.perf_counter()

    os.chdir(MVGEN)
    sys.path[:0] = [str(MVGEN), "/root/vggt", str(ROOT / "bridge")]
    from diffusers import AutoencoderKL
    from easydict import EasyDict
    from omegaconf import OmegaConf
    from my_diffusers.models import UNet2DConditionModel
    from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
    from src.modules.schedulers import get_diffusion_scheduler
    from train_geodiff_vggt_stage1 import GeoDiffBridge, load_sample, normalized_log_depth
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    manifest = json.loads(args.manifest.read_text())
    views = manifest["views"][:8]
    source = views[0]
    model_dir = MVGEN / "check_points/pretrained_model"
    config = EasyDict(OmegaConf.load(model_dir / "config.yaml"))
    config.pretrained_model_name_or_path = str(BASE)
    config.nframe = 8

    w2cs = np.stack([pad_w2c(view["extrinsic"]) for view in views])
    c2ws = np.linalg.inv(w2cs)
    radius = float(np.linalg.norm(c2ws[0, :3, 3]))
    depth = source_depth(source, radius)
    positions = c2ws[:, :3, 3]
    span = float(np.max(positions.max(0) - positions.min(0)))
    pose_scale = float(config.camera_longest_side / span) if span > config.camera_longest_side else 1.0
    w2cs[:, :3, 3] *= pose_scale
    depth *= pose_scale

    image = Image.open(source["image"]).convert("RGB")
    original_width, original_height = image.size
    height, width = choose_hw(model_dir / "ratio_set.json", original_height, original_width)
    resized = image.resize((width, height), Image.Resampling.LANCZOS if height < original_height else Image.Resampling.BICUBIC)
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

    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", local_files_only=True).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(
        BASE, subfolder="unet", rank=0, model_cfg=config.model_cfg,
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True, local_files_only=True,
    )
    unet.load_state_dict(torch.load(model_dir / "ema_unet.pt", map_location="cpu"))
    lora_parameters, lora_modules = attach_lora(unet, args.rank)
    vae.to("cuda", dtype=torch.float16)
    unet.to("cuda", dtype=torch.float16)
    # Keep trainable low-rank matrices in fp32 for stable Adam updates.
    for module in unet.modules():
        layer = getattr(module, "lora_layer", None)
        if layer is not None:
            layer.to("cuda", dtype=torch.float32)
    lora_parameters = [parameter for name, parameter in unet.named_parameters() if "lora_layer" in name]
    collector = LiveCollector(unet)
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        BASE, vae=vae, unet=unet,
        scheduler=get_diffusion_scheduler(config, name="DDIM"), safety_checker=None,
        torch_dtype=torch.float16, local_files_only=True,
    ).to("cuda")

    def detach_early(_pipeline, step_index, _timestep, callback_kwargs):
        if step_index < args.steps - 1:
            callback_kwargs["latents"] = callback_kwargs["latents"].detach()
        return callback_kwargs

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        result = pipeline.__call__.__wrapped__(
            pipeline,
            images=images, nframe=8, cond_num=1, key_rescale=None,
            height=height, width=width, intrinsics=intrinsics, extrinsics=extrinsics,
            num_inference_steps=args.steps, guidance_scale=2.0, output_type="latent",
            config=copy.deepcopy(config), tag=["custom"] * 8, class_label=0,
            depth=depths, vae=pipeline.vae, generator=generator,
            callback_on_step_end=detach_early,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    if set(collector.latest) != set(collector.names):
        raise RuntimeError(f"missing live features: {set(collector.names)-set(collector.latest)}")
    if not all(value.requires_grad for value in collector.latest.values()):
        raise RuntimeError("final-step MVGen features are detached")

    # Decode a detached copy so VGGT sees the current LoRA-generated RGBs while
    # the trainable path remains the live multiscale feature injection.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        decoded = pipeline.vae.decode(
            result.images.detach() / pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded_vggt = vggt_pad_tensor(decoded).cpu()
    # VAE is not part of the live feature graph after the final UNet call.
    pipeline.vae.to("cpu")
    torch.cuda.empty_cache()
    scene = manifest["scene_id"]
    sample = load_sample(scene, args.manifest.parent, args.cache_root, load_and_preprocess_images)
    indices = torch.arange(args.vggt_views)
    sample = {
        **sample,
        "input": torch.cat([sample["input"][0:1], decoded_vggt[1:args.vggt_views]], dim=0),
        "gt": sample["gt"].index_select(0, indices),
        "valid": sample["valid"].index_select(0, indices),
        "features": {key: value.index_select(0, indices.to(value.device)) for key, value in collector.latest.items()},
    }
    teacher = torch.load(args.cache_root / scene / "vggt_real8_teacher_depth.pt", map_location="cpu")
    teacher_normalized = teacher["normalized"].index_select(0, indices)

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
    # This LoRA step is supervised only by the VGGT depth head.  Omitting the
    # unrelated prediction heads leaves enough memory margin for repeated use.
    model.camera_head = None
    model.dpt_feature_head = None
    model.gaussian_head = None
    model.requires_grad_(False)
    if args.joint_vggt:
        bridge.requires_grad_(True).train()
        for block in model.aggregator.global_blocks[-4:]:
            block.requires_grad_(True)
        model.depth_head.requires_grad_(True)
        model.cuda().train()
        model._depth_grad_checkpoint = True
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
            {"params": train_groups["lora"], "lr": 1e-5},
            {"params": train_groups["bridge"], "lr": 2e-5},
            {"params": train_groups["vggt_global_last4"], "lr": 2.5e-7},
            {"params": train_groups["depth_head"], "lr": 2.5e-7},
        ], weight_decay=1e-4)
    else:
        model.cuda().eval()
        train_groups = {"lora": lora_parameters}
        optimizer = torch.optim.AdamW(lora_parameters, lr=1e-5)
    model._release_depth = True
    model._allow_dpt_token_grad = True
    del checkpoint

    bridge.current_features = {
        key: value[None].cuda() for key, value in sample["features"].items()
    }
    handle = model.aggregator.register_forward_hook(bridge.hook)
    with torch.autocast("cuda", dtype=torch.float16):
        prediction = model(sample["input"][None].cuda())
    handle.remove()
    predicted_depth = normalized_log_depth(prediction["depth"][0])
    geometry_loss = F.smooth_l1_loss(predicted_depth, teacher_normalized.cuda().float(), beta=0.1)
    geometry_loss.backward()

    group_gradients = {}
    for name, parameters in train_groups.items():
        values = [parameter.grad for parameter in parameters if parameter.grad is not None]
        group_gradients[name] = {
            "tensors": len(values),
            "nonzero_tensors": sum(int(torch.count_nonzero(value).item() > 0) for value in values),
            "finite": all(bool(torch.isfinite(value).all()) for value in values),
            "norm": math.sqrt(sum(float(value.float().square().sum()) for value in values)),
        }
    grads = [parameter.grad for parameter in lora_parameters if parameter.grad is not None]
    nonzero = sum(int(torch.count_nonzero(gradient).item() > 0) for gradient in grads)
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in grads)
    grad_norm = math.sqrt(sum(float(gradient.float().square().sum()) for gradient in grads))
    if not grads or nonzero == 0 or not finite:
        raise RuntimeError(f"bad LoRA gradients: tensors={len(grads)} nonzero={nonzero} finite={finite}")

    for parameters in train_groups.values():
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    lora_state = {
        name: value.detach().cpu()
        for name, value in unet.state_dict().items()
        if "lora_layer" in name
    }
    torch.save(lora_state, args.output / "lora_smoke.pt")
    summary = {
        "scene": scene,
        "protocol": (
            "joint MVGen LoRA + bridge + VGGT last4 global blocks + depth head"
            if args.joint_vggt else
            "11 detached denoising steps + final differentiable MVGen step -> live multiscale features -> frozen Stage2A VGGT depth loss"
        ),
        "lora_rank": args.rank,
        "attention_modules": lora_modules,
        "lora_parameter_tensors": len(lora_parameters),
        "lora_parameters": sum(parameter.numel() for parameter in lora_parameters),
        "live_feature_shapes": {key: list(value.shape) for key, value in collector.latest.items()},
        "geometry_loss": float(geometry_loss),
        "gradient_tensors": len(grads),
        "nonzero_gradient_tensors": nonzero,
        "grad_norm": grad_norm,
        "all_gradients_finite": finite,
        "group_gradients": group_gradients,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "vggt_missing_after_partial_load": len(missing),
        "vggt_unexpected_after_partial_load": len(unexpected),
        "elapsed_seconds": time.perf_counter() - started,
        "blind_accessed": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
