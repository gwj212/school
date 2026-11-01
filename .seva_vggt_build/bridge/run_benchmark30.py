#!/usr/bin/env python3
"""Run the fixed-eight SEVA/VGGT ablation on a stratified 30-scene set."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/root/seva_vggt")
DATA_ROOT = Path("/root/vggt/co3d_completion_train")
ANCHOR_SCENE = "cup_46_2479_7421"


def valid_scenes(data_root: Path) -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for scene in sorted(data_root.iterdir()):
        if not scene.is_dir():
            continue
        if not all((scene / name).exists() for name in ("images", "masks", "cameras.json")):
            continue
        images = list((scene / "images").glob("*.jpg"))
        if len(images) < 20:
            continue
        by_category[scene.name.split("_", 1)[0]].append(scene.name)
    return dict(sorted(by_category.items()))


def choose_scenes(data_root: Path, count: int, seed: int) -> list[str]:
    by_category = valid_scenes(data_root)
    rng = random.Random(seed)
    selected = [ANCHOR_SCENE] if any(ANCHOR_SCENE in values for values in by_category.values()) else []
    for category, values in by_category.items():
        candidates = [value for value in values if value not in selected]
        selected.extend(rng.sample(candidates, min(2, len(candidates))))
    category_cycle = list(by_category)
    cursor = 0
    while len(selected) < count:
        category = category_cycle[cursor % len(category_cycle)]
        candidates = [value for value in by_category[category] if value not in selected]
        if candidates:
            selected.append(rng.choice(candidates))
        cursor += 1
    return selected[:count]


def run_checked(command: list[str], log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(command, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def complete(output: Path, real_count: int) -> bool:
    branch = output / f"seva_{real_count}real_{8-real_count}gen"
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            output / f"run_metadata_seva_{real_count}real_{8-real_count}gen.json",
            output / f"comparison_real8_vs_seva_{real_count}real_{8-real_count}gen.json",
            branch / "image_metrics.json",
            branch / "point_cloud.ply",
            branch / "vggt_reconstruction.pt",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmark30_fixed8")
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "config/benchmark30")
    parser.add_argument("--vggt-side", type=int, default=518)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_root / "benchmark30_scenes.json"
    if selection_path.exists() and not args.force:
        scenes = json.loads(selection_path.read_text())["scenes"]
    else:
        scenes = choose_scenes(args.data_root, args.count, args.seed)
        selection = {
            "selection": "stratified: anchor cup scene, two seeded scenes per category, then round-robin fill",
            "seed": args.seed,
            "count": len(scenes),
            "scenes": scenes,
        }
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")

    status_path = args.output_root / "benchmark_status.json"
    status = {"started_at_unix": time.time(), "total_runs": len(scenes) * 7, "completed": [], "failed": []}
    completed_count = 0
    for scene_index, scene in enumerate(scenes, start=1):
        manifest = args.manifest_root / f"{scene}.json"
        if not manifest.exists() or args.force:
            run_checked(
                [
                    sys.executable,
                    str(args.root / "bridge/dataset_adapter.py"),
                    scene,
                    "--mode", "exact20_from_completion",
                    "--view-count", "20",
                    "--input-count", "4",
                    "--output", str(manifest),
                ]
            )
        scene_output = args.output_root / scene
        for real_count in range(1, 8):
            run_id = f"{scene}/seva_{real_count}real_{8-real_count}gen"
            if not args.force and complete(scene_output, real_count):
                completed_count += 1
                print(f"SKIP {completed_count}/{len(scenes)*7} {run_id}", flush=True)
                continue
            command = [
                sys.executable,
                str(args.root / "bridge/run_pipeline.py"),
                str(manifest),
                "--output-dir", str(scene_output),
                "--device", "cuda:0",
                "--input-count", str(real_count),
                "--total-count", "8",
                "--vggt-max-side", str(args.vggt_side),
                "--reuse-real-baseline",
            ]
            log_path = scene_output / "logs" / f"{real_count}real_{8-real_count}gen.log"
            print(
                f"RUN scene={scene_index}/{len(scenes)} ratio={real_count}+{8-real_count} {run_id}",
                flush=True,
            )
            try:
                run_checked(command, log_path)
                completed_count += 1
                status["completed"].append(run_id)
                print(f"DONE {completed_count}/{len(scenes)*7} {run_id}", flush=True)
            except subprocess.CalledProcessError as error:
                status["failed"].append({"run": run_id, "returncode": error.returncode, "log": str(log_path)})
                print(f"FAILED {run_id}; see {log_path}", flush=True)
            status["updated_at_unix"] = time.time()
            status["completed_count"] = completed_count
            status_path.write_text(json.dumps(status, indent=2) + "\n")

    status["finished_at_unix"] = time.time()
    status["completed_count"] = sum(complete(args.output_root / scene, ratio) for scene in scenes for ratio in range(1, 8))
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
