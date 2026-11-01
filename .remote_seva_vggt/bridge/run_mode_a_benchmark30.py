#!/usr/bin/env python3
"""Run the 30-scene single-real-view Mode-A SEVA image benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/root/seva_vggt")


def complete(scene_root: Path) -> bool:
    branch = scene_root / "seva_1real_7gen"
    metadata = scene_root / "run_metadata_seva_1real_7gen.json"
    if not metadata.is_file() or not (branch / "image_metrics.json").is_file():
        return False
    try:
        payload = json.loads(metadata.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        payload.get("seva_steps") == 50
        and payload.get("seva_context") == 21
        and payload.get("seva_chunk_strategy") == "nearest-gt"
        and payload.get("seva_short_side") == 576
        and payload.get("seva_cfg") == 4.0
        and len(list((branch / "generated_only").glob("view_*.png"))) == 7
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "outputs/benchmark30_fixed8/benchmark30_scenes.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/benchmark30_mode_a_single",
    )
    parser.add_argument(
        "--manifest-root", type=Path, default=ROOT / "config/benchmark30"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    scenes = selection["scenes"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "benchmark30_scenes.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    status_path = args.output_root / "mode_a_status.json"
    status = {
        "protocol": "1 real + 7 generated, T=21, short-side=576, 50 steps, nearest-gt, single-view CFG=4",
        "started_at_unix": time.time(),
        "completed": [],
        "failed": [],
    }
    for number, scene in enumerate(scenes, start=1):
        scene_root = args.output_root / scene
        if complete(scene_root):
            status["completed"].append(scene)
            print(f"SKIP {number}/{len(scenes)} {scene}", flush=True)
            continue
        (scene_root / "logs").mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(ROOT / "bridge/run_pipeline.py"),
            str(args.manifest_root / f"{scene}.json"),
            "--output-dir",
            str(scene_root),
            "--device",
            args.device,
            "--input-count",
            "1",
            "--total-count",
            "8",
            "--skip-reconstruction",
        ]
        print(f"RUN {number}/{len(scenes)} {scene}", flush=True)
        log_path = scene_root / "logs/mode_a.log"
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            if not complete(scene_root):
                raise RuntimeError("run returned successfully but Mode-A outputs are incomplete")
            status["completed"].append(scene)
            print(f"DONE {number}/{len(scenes)} {scene}", flush=True)
        except Exception as error:
            status["failed"].append(
                {"scene": scene, "error": str(error), "log": str(log_path)}
            )
            print(f"FAILED {number}/{len(scenes)} {scene}: {error}", flush=True)
        status["updated_at_unix"] = time.time()
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    status["finished_at_unix"] = time.time()
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
