#!/usr/bin/env python3
"""Audit whether foreground/background Gaussian contribution explains Stage-2A failures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(x, y):
    x = rankdata(x)
    y = rankdata(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rasterize(gs, selector, viewmat, intrinsic, size):
    from gsplat import rasterization

    rendered, alpha, _ = rasterization(
        means=gs["xyz"][selector],
        quats=gs["rotation"][selector],
        scales=gs["scale"][selector],
        opacities=gs["opacity"][selector].squeeze(-1),
        colors=gs["color"][selector],
        viewmats=viewmat,
        Ks=intrinsic,
        width=size,
        height=size,
        sh_degree=None,
        near_plane=0.01,
        far_plane=1e10,
    )
    rgb = rendered[0, :, :, :3].permute(2, 0, 1).clamp(0, 1)
    alpha = alpha[0, :, :, 0].clamp(0, 1)
    return rgb, alpha


def masked_mean(value, mask):
    return float((value * mask).sum() / mask.sum().clamp_min(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=10)
    parser.add_argument("--render-size", type=int, default=192)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from train_geodiff_vggt_stage1 import GeoDiffBridge, load_sample, run_model, scene_gaussians
    from train_geodiff_vggt_stage2a import attach_object_masks
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    protocol = json.loads(args.protocol.read_text())
    scenes = protocol["validation_scenes"][: args.max_scenes]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    bridge = GeoDiffBridge(checkpoint["channels"])
    bridge.load_state_dict(checkpoint["bridge"])
    bridge.requires_grad_(False).cuda().eval()
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.load_state_dict(checkpoint["vggt_state"], strict=False)
    model.point_head = None
    model.track_head = None
    model.requires_grad_(False).cuda().eval()
    del checkpoint

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    rows = []
    with torch.inference_mode():
        for scene_index, scene in enumerate(scenes, 1):
            sample = load_sample(
                scene, args.manifest_root / "validation", args.cache_root,
                load_and_preprocess_images,
            )
            attach_object_masks(sample, args.manifest_root / "validation")
            prediction, _ = run_model(model, bridge, sample, device, dtype, True)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                prediction["pose_enc"].float(), (518, 518)
            )
            extrinsics = extrinsics[0].float()
            intrinsics = intrinsics[0].float()
            scale = intrinsics.new_tensor([
                [args.render_size / 518, 0, 0],
                [0, args.render_size / 518, 0],
                [0, 0, 1],
            ])
            intrinsics = torch.matmul(scale.unsqueeze(0), intrinsics)
            gs = scene_gaussians(prediction, sample["valid"])
            valid = sample["valid"].flatten(1).to(device)
            source_object = sample["object_mask"].flatten(1).to(device)[valid]
            if not bool(source_object.any()) or bool(source_object.all()):
                print(f"SKIP {scene} degenerate object selector", flush=True)
                continue
            all_selector = torch.ones_like(source_object, dtype=torch.bool)
            bottom = extrinsics.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            for slot in range(1, 8):
                viewmat = torch.cat([extrinsics[slot], bottom], dim=0).unsqueeze(0)
                intrinsic = intrinsics[slot : slot + 1]
                full_rgb, full_alpha = rasterize(gs, all_selector, viewmat, intrinsic, args.render_size)
                fg_rgb, fg_alpha = rasterize(gs, source_object, viewmat, intrinsic, args.render_size)
                bg_rgb, bg_alpha = rasterize(gs, ~source_object, viewmat, intrinsic, args.render_size)
                target = F.interpolate(
                    sample["gt"][slot : slot + 1].float().to(device),
                    (args.render_size, args.render_size), mode="bilinear",
                    align_corners=False, antialias=True,
                )[0]
                target_mask = F.interpolate(
                    sample["object_mask"][slot : slot + 1, None].float().to(device),
                    (args.render_size, args.render_size), mode="nearest",
                )[0, 0]
                if float(target_mask.sum()) == 0:
                    continue
                error = (full_rgb - target).abs().mean(0)
                mse = ((full_rgb - target) ** 2).mean(0)
                object_mse = masked_mean(mse, target_mask)
                full_minus_bg = (full_rgb - bg_rgb).abs().mean(0)
                full_minus_fg = (full_rgb - fg_rgb).abs().mean(0)
                fg_alpha_mean = masked_mean(fg_alpha, target_mask)
                bg_alpha_mean = masked_mean(bg_alpha, target_mask)
                fg_dominance = masked_mean(fg_alpha / (fg_alpha + bg_alpha + 1e-6), target_mask)
                rows.append({
                    "scene": scene,
                    "slot": slot,
                    "object_psnr": -10.0 * math.log10(max(object_mse, 1e-10)),
                    "object_l1": masked_mean(error, target_mask),
                    "full_alpha_in_object": masked_mean(full_alpha, target_mask),
                    "foreground_alpha_in_object": fg_alpha_mean,
                    "background_alpha_in_object": bg_alpha_mean,
                    "foreground_alpha_dominance": fg_dominance,
                    "foreground_causal_rgb_contribution": masked_mean(full_minus_bg, target_mask),
                    "background_causal_rgb_contribution": masked_mean(full_minus_fg, target_mask),
                })
            del prediction
            torch.cuda.empty_cache()
            print(f"CONTRIBUTION_AUDIT {scene_index}/{len(scenes)} {scene}", flush=True)

    psnr = [row["object_psnr"] for row in rows]
    l1 = [row["object_l1"] for row in rows]
    fields = [
        "full_alpha_in_object", "foreground_alpha_in_object", "background_alpha_in_object",
        "foreground_alpha_dominance", "foreground_causal_rgb_contribution",
        "background_causal_rgb_contribution",
    ]
    correlations = {}
    for field in fields:
        values = [row[field] for row in rows]
        correlations[f"{field}_vs_object_psnr"] = spearman(values, psnr)
        correlations[f"{field}_vs_object_l1"] = spearman(values, l1)
    candidate_strength = max(
        correlations["foreground_alpha_in_object_vs_object_psnr"],
        correlations["foreground_alpha_dominance_vs_object_psnr"],
        correlations["foreground_causal_rgb_contribution_vs_object_psnr"],
    )
    summary = {
        "scenes": len(set(row["scene"] for row in rows)),
        "views": len(rows),
        "means": {field: float(np.mean([row[field] for row in rows])) for field in fields},
        "correlations_spearman": correlations,
        "decision_gate": "one foreground contribution signal vs object-region PSNR >= +0.30",
        "candidate_strength": candidate_strength,
        "passes_gate": bool(candidate_strength >= 0.30),
        "rows": rows,
        "blind_accessed": False,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
