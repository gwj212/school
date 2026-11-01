#!/usr/bin/env python3
"""Resume benchmark30 with per-scene shared model loading."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_benchmark30 import ROOT, choose_scenes, complete, run_checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("/root/vggt/co3d_completion_train"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmark30_fixed8")
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "config/benchmark30")
    parser.add_argument("--vggt-side", type=int, default=518)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_root / "benchmark30_scenes.json"
    if selection_path.exists():
        scenes = json.loads(selection_path.read_text())["scenes"]
    else:
        scenes = choose_scenes(args.data_root, args.count, args.seed)
        selection_path.write_text(json.dumps({"seed": args.seed, "count": len(scenes), "scenes": scenes}, indent=2) + "\n")

    status_path = args.output_root / "benchmark_shared_status.json"
    status = {"started_at_unix": time.time(), "scene_count": len(scenes), "completed_scenes": [], "failed_scenes": []}
    for index, scene in enumerate(scenes, start=1):
        manifest = args.manifest_root / f"{scene}.json"
        if not manifest.exists():
            run_checked(
                [
                    sys.executable, str(args.root / "bridge/dataset_adapter.py"), scene,
                    "--mode", "exact20_from_completion", "--view-count", "20",
                    "--input-count", "4", "--output", str(manifest),
                ]
            )
        scene_output = args.output_root / scene
        if all(complete(scene_output, ratio) for ratio in range(1, 8)):
            status["completed_scenes"].append(scene)
            print(f"SKIP SCENE {index}/{len(scenes)} {scene}", flush=True)
            continue
        command = [
            sys.executable,
            str(args.root / "bridge/run_scene_all_ratios.py"),
            str(manifest),
            "--output-dir", str(scene_output),
            "--device", "cuda:0",
            "--vggt-max-side", str(args.vggt_side),
        ]
        log = scene_output / "shared_model_run.log"
        print(f"RUN SCENE {index}/{len(scenes)} {scene}", flush=True)
        try:
            run_checked(command, log)
            if all(complete(scene_output, ratio) for ratio in range(1, 8)):
                status["completed_scenes"].append(scene)
                print(f"DONE SCENE {index}/{len(scenes)} {scene}", flush=True)
            else:
                status["failed_scenes"].append({"scene": scene, "reason": "incomplete outputs", "log": str(log)})
        except subprocess.CalledProcessError as error:
            status["failed_scenes"].append({"scene": scene, "returncode": error.returncode, "log": str(log)})
            print(f"FAILED SCENE {scene}; see {log}", flush=True)
        status["updated_at_unix"] = time.time()
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    status["finished_at_unix"] = time.time()
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
