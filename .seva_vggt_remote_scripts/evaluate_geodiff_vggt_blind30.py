#!/usr/bin/env python3
"""Evaluate a frozen GeoDiff-VGGT stage-1 checkpoint on the sealed blind30."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from train_geodiff_vggt_stage1 import GeoDiffBridge, load_sample, render_loss, run_model


def depth_metrics(pred, target):
    pred, target = pred.float(), target.float()
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


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render-size", type=int, default=192)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    scenes = protocol["blind_test_scenes"]
    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.gs_optimizer import render_gaussians, ssim

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    bridge = GeoDiffBridge(checkpoint["channels"]).to(device)
    bridge.load_state_dict(checkpoint["bridge"]); bridge.eval()
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.point_head = None; model.track_head = None
    model.requires_grad_(False).to(device).eval(); model._release_depth = True

    rows = []
    with torch.inference_mode():
        for index, scene in enumerate(scenes):
            sample = load_sample(scene, args.manifests, args.cache_root, load_and_preprocess_images)
            with torch.autocast("cuda", dtype=dtype):
                teacher = model(sample["gt"][None].to(device))
            teacher_depth = teacher["depth"][0].float()
            for method, enabled in (("mvgen_baseline", False), ("geodiff_vggt", True)):
                pred, gate = run_model(model, bridge, sample, device, dtype, enabled)
                _, image_rows = render_loss(pred, sample["gt"].to(device), sample["valid"], list(range(1, 8)), pose_encoding_to_extri_intri, render_gaussians, ssim, args.render_size)
                pred_depth = pred["depth"][0].float()
                for item in image_rows:
                    slot = item["slot"]
                    row = {"scene": scene, "category": scene.split("_", 1)[0], "method": method, **item}
                    row.update(depth_metrics(pred_depth[slot], teacher_depth[slot]))
                    rows.append(row)
                del pred, gate
                torch.cuda.empty_cache()
            del teacher
            torch.cuda.empty_cache()
            print(f"BLIND {index + 1}/{len(scenes)} {scene}", flush=True)

    write_csv(args.output / "metrics_per_view.csv", rows)
    summary = {"scenes": len(scenes), "views_per_method": 7 * len(scenes), "blind_protocol": str(args.protocol), "checkpoint": str(args.checkpoint)}
    for method in ("mvgen_baseline", "geodiff_vggt"):
        subset = [r for r in rows if r["method"] == method]
        summary[method] = {key: float(np.mean([r[key] for r in subset])) for key in ("psnr", "ssim", "l1", "depth_absrel", "depth_bad20", "depth_silog")}
    summary["delta_geodiff_minus_baseline"] = {key: summary["geodiff_vggt"][key] - summary["mvgen_baseline"][key] for key in summary["mvgen_baseline"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
