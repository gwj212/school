#!/usr/bin/env python3
"""Compare Gaussian centers by real/generated source slots against paired real8 slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_pipeline as rp


def points_for_slots(reconstruction: dict, slots: list[int], opacity_threshold: float) -> np.ndarray:
    gaussians = reconstruction["gaussians"]
    xyz = gaussians["xyz"][0, slots].float().numpy().reshape(-1, 3)
    opacity = gaussians["opacity"][0, slots].float().numpy().reshape(-1)
    keep = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity) & (opacity >= opacity_threshold)
    return xyz[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/seva_vggt/outputs/benchmark30_fixed8"))
    parser.add_argument("--sample-points", type=int, default=20_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    scenes = json.loads((args.root / "benchmark30_scenes.json").read_text())["scenes"]
    completed = 0
    for scene_index, scene in enumerate(scenes, start=1):
        scene_root = args.root / scene
        real = torch.load(scene_root / "real8/vggt_reconstruction.pt", map_location="cpu")
        for real_count in range(1, 8):
            name = f"seva_{real_count}real_{8-real_count}gen"
            metadata = json.loads((scene_root / f"run_metadata_{name}.json").read_text())
            comparison_path = scene_root / f"comparison_real8_vs_{name}.json"
            comparison = json.loads(comparison_path.read_text())
            mixed = torch.load(scene_root / name / "vggt_reconstruction.pt", map_location="cpu")
            baseline_indices = [int(value) for value in metadata["baseline_indices"]]
            position = {view_index: slot for slot, view_index in enumerate(baseline_indices)}
            generated_slots = [position[int(value)] for value in metadata["target_indices"]]
            real_slots = [position[int(value)] for value in metadata["input_indices"]]
            real_generated_reference = points_for_slots(real, generated_slots, args.opacity_threshold)
            mixed_generated = points_for_slots(mixed, generated_slots, args.opacity_threshold)
            real_input_reference = points_for_slots(real, real_slots, args.opacity_threshold)
            mixed_real = points_for_slots(mixed, real_slots, args.opacity_threshold)
            shared_slot = position[17]
            real_shared_reference = points_for_slots(real, [shared_slot], args.opacity_threshold)
            mixed_shared = points_for_slots(mixed, [shared_slot], args.opacity_threshold)
            generated_comparison = rp.compare_pointclouds(
                real_generated_reference,
                mixed_generated,
                args.sample_points,
                args.seed,
                "real8_same_target_slots",
                "mixed_generated_source_slots",
            )
            real_comparison = rp.compare_pointclouds(
                real_input_reference,
                mixed_real,
                args.sample_points,
                args.seed,
                "real8_same_input_slots",
                "mixed_real_source_slots",
            )
            shared_comparison = rp.compare_pointclouds(
                real_shared_reference,
                mixed_shared,
                args.sample_points,
                args.seed,
                "real8_view17_slot",
                "mixed_generated_view17_slot",
            )
            comparison["source_aware"] = {
                "definition": "Per-view Gaussian centers are separated by whether their RGB input was real or SEVA-generated, then compared with the corresponding slots of real8.",
                "opacity_threshold": args.opacity_threshold,
                "generated_slot_count": len(generated_slots),
                "real_slot_count": len(real_slots),
                "generated_slots_vs_real8_same_slots": generated_comparison,
                "real_slots_vs_real8_same_slots": real_comparison,
                "shared_view17_vs_real8_same_slot": shared_comparison,
            }
            comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
            del mixed
            completed += 1
        del real
        print(f"SOURCE_GEOMETRY {scene_index}/{len(scenes)} {scene}", flush=True)
    print(json.dumps({"completed": completed, "expected": len(scenes) * 7}, indent=2))


if __name__ == "__main__":
    main()
