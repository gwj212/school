#!/usr/bin/env python3
"""Feasibility audit: does leave-one-generated-view-out VGGT influence identify bad views?"""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=10)
    args = parser.parse_args()
    for key in ("protocol", "manifest_root", "cache_root", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sys.path[:0] = ["/root/vggt", "/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from train_geodiff_vggt_stage1 import load_sample, normalized_log_depth
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    protocol = json.loads(args.protocol.read_text())
    scenes = protocol["validation_scenes"][:args.scenes]
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    model = VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL), enable_gaussian=True)
    if hasattr(model.gaussian_head, "enable_color_head_after_init"):
        model.gaussian_head.enable_color_head_after_init()
    rp.load_reconstruction_checkpoint(model, Path("/root/vggt/ckpts/ckpt.pth"))
    model.point_head = None; model.track_head = None
    model.camera_head = None; model.dpt_feature_head = None; model.gaussian_head = None
    model.requires_grad_(False).to(device).eval()

    rows = []
    with torch.inference_mode():
        for scene_index, scene in enumerate(scenes, 1):
            sample = load_sample(
                scene, args.manifest_root / "validation", args.cache_root,
                load_and_preprocess_images,
            )
            teacher = torch.load(
                args.cache_root / scene / "vggt_real8_teacher_depth.pt", map_location="cpu"
            )["normalized"].float()
            valid = sample["valid"]
            with torch.autocast("cuda", dtype=dtype):
                full = model(sample["input"][None].to(device))
            full_depth = normalized_log_depth(full["depth"][0]).cpu()
            full_errors = []
            for view in range(8):
                mask = valid[view] & torch.isfinite(full_depth[view]) & torch.isfinite(teacher[view])
                full_errors.append(float((full_depth[view][mask] - teacher[view][mask]).abs().mean()))

            for omitted in range(1, 8):
                keep = [index for index in range(8) if index != omitted]
                with torch.autocast("cuda", dtype=dtype):
                    reduced = model(sample["input"][keep][None].to(device))
                reduced_depth = normalized_log_depth(reduced["depth"][0]).cpu()
                reduced_errors = []
                common_full_errors = []
                for reduced_index, original_index in enumerate(keep):
                    mask = valid[original_index] & torch.isfinite(reduced_depth[reduced_index]) & torch.isfinite(teacher[original_index])
                    reduced_errors.append(float((reduced_depth[reduced_index][mask] - teacher[original_index][mask]).abs().mean()))
                    common_full_errors.append(full_errors[original_index])
                error_without = float(np.mean(reduced_errors))
                error_with = float(np.mean(common_full_errors))
                rows.append({
                    "scene": scene,
                    "omitted_view": omitted,
                    "own_depth_error": full_errors[omitted],
                    "common_error_with_view": error_with,
                    "common_error_without_view": error_without,
                    "marginal_utility": error_without - error_with,
                    "harmful": error_without < error_with,
                })
                del reduced, reduced_depth
            print(f"UTILITY {scene_index}/{len(scenes)} {scene}", flush=True)
            del sample, teacher, full, full_depth
            torch.cuda.empty_cache()

    own = np.asarray([row["own_depth_error"] for row in rows])
    utility = np.asarray([row["marginal_utility"] for row in rows])
    spearman = float(np.corrcoef(ranks(own), ranks(utility))[0, 1])
    harmful = utility < 0
    high_error = own >= np.quantile(own, 0.75)
    summary = {
        "scenes": len(scenes),
        "generated_views": len(rows),
        "spearman_own_error_vs_marginal_utility": spearman,
        "harmful_view_fraction": float(harmful.mean()),
        "high_error_quartile_harmful_fraction": float(harmful[high_error].mean()),
        "mean_error_harmful": float(own[harmful].mean()) if harmful.any() else None,
        "mean_error_helpful": float(own[~harmful].mean()) if (~harmful).any() else None,
        "mean_marginal_utility": float(utility.mean()),
        "decision_gate": {
            "required_spearman_at_most": -0.25,
            "required_high_error_harmful_lift": 0.15,
            "observed_high_error_harmful_lift": float(harmful[high_error].mean() - harmful.mean()),
        },
        "passes_gate": bool(spearman <= -0.25 and harmful[high_error].mean() - harmful.mean() >= 0.15),
        "rows": rows,
        "blind_accessed": False,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
