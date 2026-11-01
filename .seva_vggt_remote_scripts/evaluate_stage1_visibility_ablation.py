#!/usr/bin/env python3
"""Validation-only audit of geometry fusion versus the learned visibility gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from train_geodiff_vggt_stage1 import (
    GeoDiffBridge,
    depth_metrics,
    load_sample,
    move_features,
    render_loss,
)


METRICS = ("psnr", "ssim", "l1", "depth_absrel", "depth_bad20", "depth_silog")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_variant(model, bridge, sample, device, dtype, *, use_bridge, use_visibility):
    if use_bridge:
        bridge.current_features = move_features(sample["features"], device)
        bridge.visibility_logits = None
        handle = model.aggregator.register_forward_hook(bridge.hook)
    else:
        handle = None
    try:
        with torch.autocast("cuda", dtype=dtype):
            prediction = model(sample["input"][None].to(device))
    finally:
        if handle is not None:
            handle.remove()
    gate = bridge.apply_visibility(prediction) if use_visibility else None
    return prediction, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text())
    scenes = protocol["validation_scenes"]
    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.gs_optimizer import render_gaussians, ssim

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    bridges = {}
    for epoch in (2, 3):
        checkpoint = torch.load(args.stage1 / f"epoch_{epoch}.pt", map_location="cpu")
        bridge = GeoDiffBridge(checkpoint["channels"]).to(device)
        bridge.load_state_dict(checkpoint["bridge"])
        bridge.eval()
        bridges[epoch] = bridge

    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.point_head = None
    model.track_head = None
    model.requires_grad_(False).to(device).eval()
    model._release_depth = True

    variants = (
        ("mvgen_baseline", None, False, False),
        ("epoch2_geometry_only", 2, True, False),
        ("epoch2_full", 2, True, True),
        ("epoch3_geometry_only", 3, True, False),
        ("epoch3_full", 3, True, True),
    )
    rows = []
    with torch.inference_mode():
        for scene_index, scene in enumerate(scenes, 1):
            sample = load_sample(scene, args.manifests, args.cache_root, load_and_preprocess_images)
            teacher_cache = torch.load(
                args.cache_root / scene / "vggt_real8_teacher_depth.pt", map_location="cpu"
            )
            teacher_depth = teacher_cache["raw"] if isinstance(teacher_cache, dict) else teacher_cache
            for method, epoch, use_bridge, use_visibility in variants:
                bridge = bridges[epoch] if epoch is not None else bridges[2]
                prediction, gate = run_variant(
                    model,
                    bridge,
                    sample,
                    device,
                    dtype,
                    use_bridge=use_bridge,
                    use_visibility=use_visibility,
                )
                _, image_rows = render_loss(
                    prediction,
                    sample["gt"].to(device),
                    sample["valid"],
                    list(range(1, 8)),
                    pose_encoding_to_extri_intri,
                    render_gaussians,
                    ssim,
                    args.render_size,
                )
                predicted_depth = prediction["depth"][0].float()
                opacity_mean = float(prediction["gaussians"]["opacity"].float().mean())
                gate_mean = float(gate.float().mean()) if gate is not None else None
                for item in image_rows:
                    slot = item["slot"]
                    row = {
                        "scene": scene,
                        "category": scene.split("_", 1)[0],
                        "method": method,
                        "opacity_mean": opacity_mean,
                        "visibility_gate_mean": gate_mean,
                        **item,
                    }
                    row.update(depth_metrics(predicted_depth[slot], teacher_depth[slot]))
                    rows.append(row)
                del prediction, gate
                torch.cuda.empty_cache()
            print(f"ABLATION {scene_index}/{len(scenes)} {scene}", flush=True)

    write_csv(args.output / "metrics_per_view.csv", rows)
    summary = {
        "protocol": str(args.protocol),
        "validation_scenes": len(scenes),
        "views_per_method": len(scenes) * 7,
        "methods": {},
    }
    for method, *_ in variants:
        subset = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            key: float(np.mean([row[key] for row in subset])) for key in METRICS
        }
        summary["methods"][method]["opacity_mean"] = float(
            np.mean([row["opacity_mean"] for row in subset])
        )
        gates = [row["visibility_gate_mean"] for row in subset if row["visibility_gate_mean"] is not None]
        summary["methods"][method]["visibility_gate_mean"] = float(np.mean(gates)) if gates else None
    baseline = summary["methods"]["mvgen_baseline"]
    summary["delta_vs_baseline"] = {
        method: {key: values[key] - baseline[key] for key in METRICS}
        for method, values in summary["methods"].items()
        if method != "mvgen_baseline"
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
