#!/usr/bin/env python3
"""Audit whether Stage2A rendering failures correlate with VGGT/known-camera gauge mismatch."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def pad_extrinsic(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (4, 4):
        return value
    result = np.eye(4)
    result[:3, :4] = value
    return result


def camera_centers(w2c):
    rotation = w2c[:, :3, :3]
    translation = w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", rotation.transpose(0, 2, 1), translation)


def similarity_residual(source, target):
    source_centered = source - source.mean(0)
    target_centered = target - target.mean(0)
    u, singular, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = singular.sum() / max(float((source_centered ** 2).sum()), 1e-12)
    aligned = scale * source_centered @ rotation + target.mean(0)
    path_scale = max(float(np.sqrt((target_centered ** 2).sum(1).mean())), 1e-9)
    return float(np.sqrt(((aligned - target) ** 2).sum(1).mean()) / path_scale)


def relative_rotation_error(predicted, target):
    errors = []
    for index in range(1, len(predicted)):
        pred_relative = predicted[index, :3, :3] @ predicted[0, :3, :3].T
        target_relative = target[index, :3, :3] @ target[0, :3, :3].T
        residual = pred_relative @ target_relative.T
        cosine = np.clip((np.trace(residual) - 1) / 2, -1, 1)
        errors.append(math.degrees(math.acos(cosine)))
    return float(np.mean(errors)), float(np.max(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("protocol", "manifest_root", "cache_root", "checkpoint", "validation", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from train_geodiff_vggt_stage1 import GeoDiffBridge, load_sample
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    protocol = json.loads(args.protocol.read_text())
    validation = json.loads(args.validation.read_text())
    render_rows = validation["rows"]
    render_by_scene = {}
    for scene in protocol["validation_scenes"]:
        rows = [row for row in render_rows if row["scene"] == scene]
        render_by_scene[scene] = {
            "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
            "worst_psnr": float(np.min([row["psnr"] for row in rows])),
            "mean_l1": float(np.mean([row["l1"] for row in rows])),
        }

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    bridge = GeoDiffBridge(checkpoint["channels"])
    bridge.load_state_dict(checkpoint["bridge"])
    bridge.requires_grad_(False).cuda().eval()
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.load_state_dict(checkpoint["vggt_state"], strict=False)
    model.point_head = None; model.track_head = None
    model.depth_head = None; model.dpt_feature_head = None; model.gaussian_head = None
    model.requires_grad_(False).cuda().eval()
    del checkpoint

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    rows = []
    with torch.inference_mode():
        for index, scene in enumerate(protocol["validation_scenes"], 1):
            manifest = json.loads((args.manifest_root / "validation" / f"{scene}.json").read_text())
            target_extrinsics = np.stack([pad_extrinsic(view["extrinsic"]) for view in manifest["views"][:8]])
            sample = load_sample(scene, args.manifest_root / "validation", args.cache_root, load_and_preprocess_images)
            bridge.current_features = {key: value[None].cuda() for key, value in sample["features"].items()}
            handle = model.aggregator.register_forward_hook(bridge.hook)
            with torch.autocast("cuda", dtype=dtype):
                prediction = model(sample["input"][None].cuda())
            handle.remove()
            predicted_extrinsics, _ = pose_encoding_to_extri_intri(prediction["pose_enc"].float(), (518, 518))
            predicted_extrinsics = predicted_extrinsics[0].cpu().numpy().astype(np.float64)
            predicted_padded = np.repeat(np.eye(4)[None], 8, axis=0)
            predicted_padded[:, :3, :4] = predicted_extrinsics
            rotation_mean, rotation_max = relative_rotation_error(predicted_padded, target_extrinsics)
            translation_residual = similarity_residual(
                camera_centers(predicted_padded), camera_centers(target_extrinsics)
            )
            rows.append({
                "scene": scene,
                "rotation_error_mean_deg": rotation_mean,
                "rotation_error_max_deg": rotation_max,
                "translation_similarity_residual": translation_residual,
                **render_by_scene[scene],
            })
            print(f"CAMERA_AUDIT {index}/30 {scene}", flush=True)

    rot = [row["rotation_error_mean_deg"] for row in rows]
    trans = [row["translation_similarity_residual"] for row in rows]
    mean_psnr = [row["mean_psnr"] for row in rows]
    worst_psnr = [row["worst_psnr"] for row in rows]
    mean_l1 = [row["mean_l1"] for row in rows]
    correlations = {
        "rotation_vs_mean_psnr": spearman(rot, mean_psnr),
        "rotation_vs_worst_psnr": spearman(rot, worst_psnr),
        "rotation_vs_mean_l1": spearman(rot, mean_l1),
        "translation_vs_mean_psnr": spearman(trans, mean_psnr),
        "translation_vs_worst_psnr": spearman(trans, worst_psnr),
        "translation_vs_mean_l1": spearman(trans, mean_l1),
    }
    passes = (
        min(correlations["rotation_vs_mean_psnr"], correlations["rotation_vs_worst_psnr"],
            correlations["translation_vs_mean_psnr"], correlations["translation_vs_worst_psnr"])
        <= -0.30
    )
    summary = {
        "scenes": 30,
        "camera_error_means": {
            "rotation_error_mean_deg": float(np.mean(rot)),
            "translation_similarity_residual": float(np.mean(trans)),
        },
        "spearman_correlations": correlations,
        "decision_gate": "at least one pose-error vs PSNR correlation <= -0.30",
        "passes_gate": bool(passes),
        "rows": rows,
        "blind_accessed": False,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
