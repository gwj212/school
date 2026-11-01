#!/usr/bin/env python3
"""Fast SEVA completion followed by VGGT Gaussian reconstruction."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("VGGT_LARGE_MODEL_ROOT", REPO_ROOT / "models"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("output/seva_vggt"))
    parser.add_argument("--seva-root", type=Path, default=REPO_ROOT / "third_party/seva")
    parser.add_argument("--vggt-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--seva-model", default=str(MODEL_ROOT / "stable-virtual-camera")
    )
    parser.add_argument("--seva-version", type=float, default=1.1)
    parser.add_argument("--vggt-model", default=str(MODEL_ROOT / "VGGT-1B"))
    parser.add_argument(
        "--vggt-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--vae-model", default=str(MODEL_ROOT / "sd-vae-ft-mse"))
    parser.add_argument(
        "--clip-weights",
        default=str(
            MODEL_ROOT
            / "CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-input-views", type=int, default=4)
    parser.add_argument("--novel-views", type=int, default=2)
    parser.add_argument("--seva-size", type=int, default=384)
    parser.add_argument("--vggt-max-side", type=int, default=294)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def collect_images(path: Path, limit: int) -> list[str]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    if not files:
        raise FileNotFoundError(f"No input images found in {path}")
    return [str(p) for p in files[:limit]]


def to_homogeneous(matrices: torch.Tensor) -> torch.Tensor:
    bottom = matrices.new_tensor([0.0, 0.0, 0.0, 1.0])
    bottom = bottom.view(1, 1, 4).expand(matrices.shape[0], 1, 4)
    return torch.cat([matrices, bottom], dim=1)


def orthogonalize(rotation: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(rotation)
    result = u @ vh
    if torch.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result


def look_at_pose(position: torch.Tensor, focus: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    z_axis = F.normalize(focus - position, dim=0)
    x_axis = F.normalize(torch.cross(z_axis, up, dim=0), dim=0)
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    pose = torch.eye(4, dtype=position.dtype, device=position.device)
    pose[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
    pose[:3, 3] = position
    return pose


def make_target_cameras(
    input_c2ws: torch.Tensor,
    count: int,
    world_points: torch.Tensor | None,
) -> torch.Tensor:
    if count <= 0:
        return input_c2ws.new_empty((0, 4, 4))
    if len(input_c2ws) > 1:
        targets = []
        for index in range(count):
            left = index % (len(input_c2ws) - 1)
            cycle = index // (len(input_c2ws) - 1)
            alpha = (cycle + 1) / (cycle + 2)
            a, b = input_c2ws[left], input_c2ws[left + 1]
            pose = torch.eye(4, dtype=a.dtype, device=a.device)
            pose[:3, :3] = orthogonalize((1.0 - alpha) * a[:3, :3] + alpha * b[:3, :3])
            pose[:3, 3] = (1.0 - alpha) * a[:3, 3] + alpha * b[:3, 3]
            targets.append(pose)
        return torch.stack(targets)

    reference = input_c2ws[0]
    position = reference[:3, 3]
    if world_points is not None and world_points.numel() > 0:
        points = world_points.reshape(-1, 3)
        finite = torch.isfinite(points).all(dim=-1)
        focus = points[finite].median(dim=0).values if finite.any() else None
    else:
        focus = None
    if focus is None:
        focus = position + reference[:3, 2]
    distance = torch.linalg.norm(focus - position).clamp_min(0.1)
    right = reference[:3, 0]
    up = -reference[:3, 1]
    offsets = torch.linspace(-0.12, 0.12, count + 2, device=position.device)[1:-1]
    return torch.stack(
        [look_at_pose(position + right * distance * offset, focus, up) for offset in offsets]
    )


def load_reconstruction_checkpoint(model: torch.nn.Module, checkpoint: Path | None) -> None:
    if checkpoint is None:
        print("[warning] no Gaussian reconstruction checkpoint supplied; custom heads use initialization weights")
        return
    state = torch.load(checkpoint, map_location="cpu")
    if "gaussian_head_state_dict" in state:
        model.gaussian_head.load_state_dict(state["gaussian_head_state_dict"])
    if "dpt_feature_head_state_dict" in state:
        model.dpt_feature_head.load_state_dict(state["dpt_feature_head_state_dict"])


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def save_rgb_views(images: torch.Tensor, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    images = ((images.detach().cpu() + 1.0) * 127.5).clamp(0, 255).byte()
    for index, image in enumerate(images):
        array = image.permute(1, 2, 0).numpy()
        Image.fromarray(array).save(directory / f"{index:03d}.png")


def resize_vggt_images(images: torch.Tensor, max_side: int) -> torch.Tensor:
    height, width = images.shape[-2:]
    if max_side <= 0 or max(height, width) <= max_side:
        return images
    scale = max_side / max(height, width)
    target_height = max(14, int(round(height * scale / 14)) * 14)
    target_width = max(14, int(round(width * scale / 14)) * 14)
    return F.interpolate(
        images,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def stage_vggt_geometry(
    model: torch.nn.Module, device: torch.device, dtype: torch.dtype
) -> None:
    model.aggregator.to(device=device, dtype=dtype)
    model.depth_head.to(device=device, dtype=dtype)
    model.camera_head.to(device="cpu", dtype=torch.float32)


def stage_vggt_reconstruction(
    model: torch.nn.Module, device: torch.device, dtype: torch.dtype
) -> None:
    model.aggregator.to(device=device, dtype=dtype)
    model.depth_head.to(device=device, dtype=dtype)
    model.dpt_feature_head.to(device=device, dtype=dtype)
    model.gaussian_head.to(device=device, dtype=torch.float32)
    model.camera_head.to(device="cpu", dtype=torch.float32)


def unload_vggt_cuda(model: torch.nn.Module) -> None:
    for name in ("aggregator", "depth_head", "dpt_feature_head", "gaussian_head"):
        module = getattr(model, name, None)
        if module is not None:
            module.cpu()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.seva_size % 64:
        raise ValueError("--seva-size must be divisible by 64")
    if args.max_input_views + args.novel_views > 21:
        raise ValueError("SEVA supports at most 21 total frames per fast pass")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ["SEVA_WRITE_VIDEO"] = "0"
    os.environ["SEVA_VAE_MODEL"] = args.vae_model
    os.environ["SEVA_CLIP_MODEL"] = args.clip_weights
    sys.path.insert(0, str(args.seva_root))
    sys.path.insert(0, str(args.vggt_root))

    from seva.eval import run_one_scene, set_lowvram_mode
    from seva.model import SGMWrapper
    from seva.modules.autoencoder import AutoEncoder
    from seva.modules.conditioner import CLIPConditioner
    from seva.sampling import DiscreteDenoiser
    from seva.utils import load_model as load_seva_model
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    input_paths = collect_images(args.input, args.max_input_views)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    vggt = VGGT.from_pretrained(args.vggt_model, enable_gaussian=True).eval()
    load_reconstruction_checkpoint(vggt, args.vggt_checkpoint)
    vggt.point_head = None
    vggt.track_head = None
    geometry_source_images = load_and_preprocess_images(input_paths, mode="pad")
    input_images = resize_vggt_images(geometry_source_images, args.vggt_max_side)

    gaussian_head, feature_head = vggt.gaussian_head, vggt.dpt_feature_head
    vggt.gaussian_head = None
    vggt.dpt_feature_head = None
    stage_vggt_geometry(vggt, device, dtype)
    geometry_images = geometry_source_images.to(device)
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        geometry = vggt(geometry_images.unsqueeze(0))
    vggt.gaussian_head, vggt.dpt_feature_head = gaussian_head, feature_head

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        geometry["pose_enc"], image_size_hw=geometry_source_images.shape[-2:]
    )
    input_c2ws = torch.linalg.inv(to_homogeneous(extrinsics[0].float())).cpu()
    input_intrinsics = intrinsics[0].float().cpu()
    world_points = geometry.get("world_points")
    world_points = world_points[0].float().cpu() if world_points is not None else None
    target_c2ws = make_target_cameras(input_c2ws, args.novel_views, world_points)
    all_c2ws = torch.cat([input_c2ws, target_c2ws])
    target_intrinsics = input_intrinsics.median(dim=0).values[None].repeat(args.novel_views, 1, 1)
    all_intrinsics = torch.cat([input_intrinsics, target_intrinsics])

    del geometry, geometry_images, geometry_source_images
    unload_vggt_cuda(vggt)

    set_lowvram_mode(True)
    seva_model = SGMWrapper(
        load_seva_model(
            model_version=args.seva_version,
            pretrained_model_name_or_path=args.seva_model,
            device="cpu",
            dtype=dtype,
        ).eval()
    )
    autoencoder = AutoEncoder(chunk_size=1).eval()
    conditioner = CLIPConditioner().eval()
    denoiser = DiscreteDenoiser(num_idx=1000, device=str(device))

    total_frames = len(input_paths) + args.novel_views
    version = {
        "H": args.seva_size,
        "W": args.seva_size,
        "T": total_frames,
        "C": 4,
        "f": 8,
        "options": {
            "chunk_strategy": "nearest",
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": 1,
            "cfg": args.cfg,
            "camera_scale": 2.0,
            "num_steps": args.steps,
            "cfg_min": 1.2,
            "encoding_t": 1,
            "decoding_t": 1,
            "save_input": False,
            "save_output": False,
            "write_video": False,
            "return_tensors": True,
        },
    }
    image_cond = {
        "img": input_paths + [None] * args.novel_views,
        "input_indices": list(range(len(input_paths))),
        "prior_indices": None,
    }
    camera_cond = {
        "c2w": all_c2ws.clone(),
        "K": all_intrinsics.clone(),
        "input_indices": list(range(total_frames)),
    }
    result = next(
        run_one_scene(
            "img2img",
            version,
            model=seva_model,
            ae=autoencoder,
            conditioner=conditioner,
            denoiser=denoiser,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=str(args.output_dir / "seva"),
            use_traj_prior=False,
            traj_prior_Ks=None,
            traj_prior_c2ws=None,
            seed=args.seed,
        )
    )
    generated = next(
        value for key, value in result.items() if key.startswith("samples-rgb")
    ).detach().cpu()
    save_rgb_views(generated, args.output_dir / "seva_views")

    del seva_model, autoencoder, conditioner, denoiser, result
    torch.cuda.empty_cache()
    stage_vggt_reconstruction(vggt, device, dtype)
    generated_vggt = F.interpolate(
        (generated.to(device) + 1.0) / 2.0,
        size=input_images.shape[-2:],
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    reconstruction_input = torch.cat(
        [input_images.to(device), generated_vggt], dim=0
    )
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        reconstruction = vggt(reconstruction_input.unsqueeze(0))
    torch.save(cpu_tree(reconstruction), args.output_dir / "vggt_reconstruction.pt")
    metadata = {
        "input_paths": input_paths,
        "input_c2ws": input_c2ws,
        "target_c2ws": target_c2ws,
        "intrinsics": all_intrinsics,
        "steps": args.steps,
        "vggt_image_size": tuple(input_images.shape[-2:]),
        "video_written": False,
        "handoff": "in-memory RGB tensor",
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(metadata, args.output_dir / "pipeline_metadata.pt")
    print(f"completed in {metadata['elapsed_seconds']:.1f}s: {args.output_dir}")


if __name__ == "__main__":
    main()
