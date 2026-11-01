#!/usr/bin/env python3
"""Fair validation30 depth comparison using one frozen VGGT depth head.

Reference depth is predicted from eight real GT images.  SEVA, MVGen and the
stage-1 GeoDiff model are evaluated on exactly the same target slots 1..7 with
per-view median scale alignment.  This is a GT-image/VGGT proxy, not sensor GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_geodiff_vggt_stage1 import (
    GeoDiffBridge,
    load_sample,
    move_features,
    valid_canvas_mask,
)


METRICS = ("absrel", "rmse_norm", "silog", "bad20")


def padded_object_mask(path: str, size: int = 518) -> torch.Tensor:
    image = Image.open(path).convert("L")
    width, height = image.size
    if width >= height:
        resized_width = size
        resized_height = round(height * size / width / 14) * 14
    else:
        resized_height = size
        resized_width = round(width * size / height / 14) * 14
    image = image.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    result = np.zeros((size, size), dtype=bool)
    y0 = (size - resized_height) // 2
    x0 = (size - resized_width) // 2
    result[y0:y0 + resized_height, x0:x0 + resized_width] = np.asarray(image) > 127
    return torch.from_numpy(result)


def masked_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    fixed_scale: float | None = None,
) -> dict:
    prediction = prediction.float()
    target = target.float()
    if prediction.shape[-1] == 1:
        prediction = prediction.squeeze(-1)
    if target.shape[-1] == 1:
        target = target.squeeze(-1)
    mask = mask.to(prediction.device)
    valid = (
        mask
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & (prediction > 1e-5)
        & (target > 1e-5)
    )
    pred = prediction[valid].double()
    truth = target.to(prediction.device)[valid].double()
    if pred.numel() < 100:
        return {key: math.nan for key in METRICS} | {"scale": math.nan, "pixels": int(pred.numel())}
    scale = (
        torch.as_tensor(fixed_scale, dtype=torch.double, device=pred.device)
        if fixed_scale is not None
        else torch.median(truth) / torch.median(pred).clamp_min(1e-9)
    )
    pred = pred * scale
    relative = (pred - truth).abs() / truth.clamp_min(1e-9)
    log_error = torch.log(pred.clamp_min(1e-9)) - torch.log(truth.clamp_min(1e-9))
    return {
        "absrel": float(relative.mean()),
        "rmse_norm": float(torch.sqrt((pred - truth).square().mean()) / torch.median(truth)),
        "silog": float(torch.sqrt((log_error.square().mean() - log_error.mean().square()).clamp_min(0))),
        "bad20": float((relative > 0.20).double().mean()),
        "scale": float(scale),
        "pixels": int(pred.numel()),
    }


def shared_scene_scale(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    slots=range(1, 8),
) -> float:
    pred_values = []
    target_values = []
    for slot in slots:
        pred = prediction[slot].float()
        truth = target[slot].float()
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        if truth.shape[-1] == 1:
            truth = truth.squeeze(-1)
        mask = masks[slot]
        valid = mask & torch.isfinite(pred) & torch.isfinite(truth) & (pred > 1e-5) & (truth > 1e-5)
        pred_values.append(pred[valid].double())
        target_values.append(truth[valid].double())
    pred_all = torch.cat(pred_values)
    target_all = torch.cat(target_values)
    return float(torch.median(target_all) / torch.median(pred_all).clamp_min(1e-9))


def infer_depth(model, images: torch.Tensor, device, dtype) -> torch.Tensor:
    with torch.autocast("cuda", dtype=dtype):
        prediction = model(images[None].to(device))
    depth = prediction["depth"][0].detach().float().cpu()
    del prediction
    return depth


def infer_geodiff_depth(model, bridge, sample, device, dtype) -> torch.Tensor:
    bridge.current_features = move_features(sample["features"], device)
    bridge.visibility_logits = None
    handle = model.aggregator.register_forward_hook(bridge.hook)
    try:
        return infer_depth(model, sample["input"], device, dtype)
    finally:
        handle.remove()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for alignment in sorted({row["alignment"] for row in rows}):
        result[alignment] = {}
        for region in sorted({row["region"] for row in rows}):
            result[alignment][region] = {}
            for method in sorted({row["method"] for row in rows}):
                subset = [
                    row for row in rows
                    if row["alignment"] == alignment and row["region"] == region and row["method"] == method
                ]
                result[alignment][region][method] = {
                    metric: float(np.nanmean([row[metric] for row in subset])) for metric in METRICS
                }
                result[alignment][region][method]["views"] = len(subset)
    return result


def category_aggregate(rows: list[dict]) -> dict:
    result = {}
    for category in sorted({row["category"] for row in rows}):
        result[category] = aggregate([row for row in rows if row["category"] == category])
    return result


def paired_bootstrap(rows: list[dict], seed: int = 20261030, iterations: int = 20000) -> dict:
    scene_values: dict[tuple[str, str, str, str], dict[str, float]] = {}
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["alignment"], row["region"], row["method"])].append(row)
    for key, subset in grouped.items():
        scene_values[key] = {
            metric: float(np.nanmean([row[metric] for row in subset])) for metric in METRICS
        }
    scenes = sorted({row["scene"] for row in rows})
    rng = np.random.default_rng(seed)
    result = {}
    for alignment in sorted({row["alignment"] for row in rows}):
        result[alignment] = {}
        for region in sorted({row["region"] for row in rows}):
            result[alignment][region] = {}
            for comparator in ("seva", "mvgen_baseline"):
                label = f"geodiff_epoch2_minus_{comparator}"
                result[alignment][region][label] = {}
                for metric in METRICS:
                    deltas = np.asarray([
                        scene_values[(scene, alignment, region, "geodiff_epoch2")][metric]
                        - scene_values[(scene, alignment, region, comparator)][metric]
                        for scene in scenes
                    ])
                    indices = rng.integers(0, len(scenes), size=(iterations, len(scenes)))
                    distribution = deltas[indices].mean(axis=1)
                    result[alignment][region][label][metric] = {
                        "mean_delta": float(deltas.mean()),
                        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
                        "improved_scenes": int((deltas < 0).sum()),
                        "scene_count": len(scenes),
                        "lower_is_better": True,
                    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seva-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    scenes = list(protocol["validation_scenes"])
    if len(scenes) != 30 or set(scenes) & set(protocol["blind_test_scenes"]):
        raise RuntimeError("validation30 protocol is invalid or overlaps blind30")
    for scene in scenes:
        generated = args.seva_root / scene / "generated_only"
        missing = [index for index in range(1, 8) if not (generated / f"view_{index:02d}.png").is_file()]
        if missing:
            raise FileNotFoundError(f"{scene}: missing SEVA target slots {missing}")

    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    bridge = GeoDiffBridge(checkpoint["channels"]).to(device)
    bridge.load_state_dict(checkpoint["bridge"])
    bridge.eval()

    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.point_head = None
    model.track_head = None
    # Only the shared aggregator and frozen depth head are needed here.
    model.camera_head = None
    model.dpt_feature_head = None
    model.gaussian_head = None
    model.requires_grad_(False).to(device).eval()
    model._release_depth = True

    rows = []
    with torch.inference_mode():
        for scene_index, scene in enumerate(scenes, 1):
            manifest = json.loads((args.manifest_root / f"{scene}.json").read_text())
            views = manifest["views"][:8]
            sample = load_sample(scene, args.manifest_root, args.cache_root, load_and_preprocess_images)
            teacher_cache = torch.load(
                args.cache_root / scene / "vggt_real8_teacher_depth.pt", map_location="cpu"
            )
            teacher = teacher_cache["raw"].float() if isinstance(teacher_cache, dict) else teacher_cache.float()
            mvgen = infer_depth(model, sample["input"], device, dtype)
            geodiff = infer_geodiff_depth(model, bridge, sample, device, dtype)
            seva_paths = [views[0]["image"]] + [
                str(args.seva_root / scene / "generated_only" / f"view_{index:02d}.png")
                for index in range(1, 8)
            ]
            seva_images = load_and_preprocess_images(seva_paths, mode="pad")
            seva = infer_depth(model, seva_images, device, dtype)
            candidates = {
                "seva": seva,
                "mvgen_baseline": mvgen,
                "geodiff_epoch2": geodiff,
            }
            canvas_masks = valid_canvas_mask(views)
            object_masks = torch.stack([
                padded_object_mask(view["mask"]) & canvas_masks[index]
                for index, view in enumerate(views)
            ])
            region_masks = {"valid_canvas": canvas_masks, "object_mask": object_masks}
            shared_scales = {
                (region, method): shared_scene_scale(depth, teacher, masks)
                for region, masks in region_masks.items()
                for method, depth in candidates.items()
            }
            for slot in range(1, 8):
                for region, masks in region_masks.items():
                    for method, depth in candidates.items():
                        for alignment, fixed_scale in (
                            ("per_view_median", None),
                            ("scene_shared_median", shared_scales[(region, method)]),
                        ):
                            row = {
                                "scene": scene,
                                "category": scene.split("_", 1)[0],
                                "slot": slot,
                                "alignment": alignment,
                                "region": region,
                                "method": method,
                            }
                            row.update(masked_metrics(depth[slot], teacher[slot], masks[slot], fixed_scale))
                            rows.append(row)
            del mvgen, geodiff, seva, seva_images, sample
            torch.cuda.empty_cache()
            print(f"DEPTH {scene_index}/{len(scenes)} {scene}", flush=True)

    write_csv(args.output / "metrics_per_view.csv", rows)
    summary = {
        "protocol": str(args.protocol),
        "split": "validation_scenes",
        "blind_accessed": False,
        "scene_count": len(scenes),
        "target_views_per_method": len(scenes) * 7,
        "reference": "VGGT depth head applied jointly to eight real GT RGB views",
        "warning": "GT-image-derived VGGT depth proxy; not sensor ground-truth depth",
        "scale_alignment": [
            "per_view_median: independent scale for every target view",
            "scene_shared_median: one scale shared by all seven target views in a scene and region",
        ],
        "aggregate": aggregate(rows),
        "paired_scene_bootstrap": paired_bootstrap(rows),
        "per_category": category_aggregate(rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
