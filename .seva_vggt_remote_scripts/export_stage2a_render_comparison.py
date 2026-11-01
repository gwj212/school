#!/usr/bin/env python3
"""Export GT/MVGen/Stage1/Stage2A Gaussian-render comparison panels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8))


def label_panel(images: list[tuple[str, Image.Image]]) -> Image.Image:
    width = sum(image.width for _, image in images)
    height = max(image.height for _, image in images) + 28
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    x = 0
    for label, image in images:
        panel.paste(image, (x, 28))
        draw.text((x + 6, 7), label, fill="black")
        x += image.width
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 4, 7])
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    for name in ("protocol", "manifest_root", "cache_root", "stage1", "stage2a", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from train_geodiff_vggt_stage1 import GeoDiffBridge, MODEL_CANVAS, load_sample, run_model, scene_gaussians
    from vggt.models.vggt import VGGT
    from vggt.utils.gs_optimizer import render_gaussians, ssim
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    protocol = json.loads(args.protocol.read_text())
    scenes = protocol["validation_scenes"][:args.scenes]
    samples = {
        scene: load_sample(scene, args.manifest_root / "validation", args.cache_root, load_and_preprocess_images)
        for scene in scenes
    }
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    stage1 = torch.load(args.stage1, map_location="cpu")
    stage2a = torch.load(args.stage2a, map_location="cpu")
    bridge = GeoDiffBridge(stage1["channels"]).to(device).eval()
    bridge.load_state_dict(stage1["bridge"])
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.point_head = None
    model.track_head = None
    model.requires_grad_(False).to(device).eval()
    model._release_depth = True
    model._allow_dpt_token_grad = False

    results: dict[tuple[str, int, str], torch.Tensor] = {}
    metrics = []

    def render_method(method: str):
        for scene in scenes:
            sample = samples[scene]
            with torch.inference_mode():
                prediction, _ = run_model(model, bridge, sample, device, dtype, True)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    prediction["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS)
                )
                extrinsic = extrinsic[0].float()
                intrinsic = intrinsic[0].float() * (args.size / MODEL_CANVAS)
                intrinsic[:, 2, 2] = 1
                gaussians = scene_gaussians(prediction, sample["valid"])
                for slot in args.slots:
                    rendered = render_gaussians(
                        gaussians["xyz"], gaussians["rotation"], gaussians["scale"],
                        gaussians["opacity"], gaussians["color"],
                        extrinsic[slot], intrinsic[slot], args.size, args.size,
                    ).clamp(0, 1).cpu()
                    target = F.interpolate(
                        sample["gt"][slot:slot + 1].float(), (args.size, args.size),
                        mode="bilinear", align_corners=False, antialias=True,
                    )[0]
                    mse = float(F.mse_loss(rendered, target))
                    metrics.append({
                        "scene": scene, "slot": slot, "method": method,
                        "psnr": -10 * math.log10(max(mse, 1e-10)),
                        "ssim": float(ssim(rendered[None], target[None])),
                        "l1": float(F.l1_loss(rendered, target)),
                    })
                    results[(scene, slot, method)] = rendered
            del prediction, gaussians
            torch.cuda.empty_cache()

    render_method("stage1")
    bridge.load_state_dict(stage2a["bridge"])
    missing, unexpected = model.load_state_dict(stage2a["vggt_state"], strict=False)
    render_method("stage2a")

    overview_rows = []
    for scene in scenes:
        scene_dir = args.output / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        for slot in args.slots:
            gt = F.interpolate(samples[scene]["gt"][slot:slot + 1].float(), (args.size, args.size), mode="bilinear", align_corners=False, antialias=True)[0]
            mvgen = F.interpolate(samples[scene]["input"][slot:slot + 1].float(), (args.size, args.size), mode="bilinear", align_corners=False, antialias=True)[0]
            items = [
                ("GT real", to_pil(gt)),
                ("MVGen RGB", to_pil(mvgen)),
                ("Stage1 render", to_pil(results[(scene, slot, "stage1")])),
                ("Stage2A render", to_pil(results[(scene, slot, "stage2a")])),
            ]
            panel = label_panel(items)
            panel.save(scene_dir / f"view_{slot:02d}_panel.png")
            for label, image in items:
                safe = label.lower().replace(" ", "_")
                image.save(scene_dir / f"view_{slot:02d}_{safe}.png")
            thumb = panel.resize((panel.width // 2, panel.height // 2), Image.Resampling.LANCZOS)
            overview_rows.append((f"{scene} view {slot:02d}", thumb))

    row_width = max(image.width for _, image in overview_rows)
    row_height = max(image.height for _, image in overview_rows) + 22
    overview = Image.new("RGB", (row_width, row_height * len(overview_rows)), "white")
    draw = ImageDraw.Draw(overview)
    for index, (label, image) in enumerate(overview_rows):
        y = index * row_height
        draw.text((5, y + 3), label, fill="black")
        overview.paste(image, (0, y + 22))
    overview.save(args.output / "overview.png")

    aggregate = {}
    for method in ("stage1", "stage2a"):
        rows = [row for row in metrics if row["method"] == method]
        aggregate[method] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("psnr", "ssim", "l1")
        }
    summary = {
        "scenes": scenes, "slots": args.slots, "render_size": args.size,
        "aggregate": aggregate, "per_view": metrics,
        "vggt_missing_after_stage2a_load": len(missing),
        "vggt_unexpected_after_stage2a_load": len(unexpected),
        "blind_accessed": False,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
