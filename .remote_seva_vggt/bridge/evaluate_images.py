#!/usr/bin/env python3
"""Compare generated target views with held-out real images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import metric_alignment as image_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("generated_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alignment-mode", choices=("crop", "pad", "stretch"), default="crop")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if a manifest target is absent; otherwise evaluate generated files present.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    views = manifest["views"]
    rows = []
    for index in manifest["target_indices"]:
        view = views[index]
        generated_path = args.generated_dir / f"view_{int(view['index']):02d}.png"
        ground_truth_path = Path(view["ground_truth"])
        if not generated_path.is_file():
            if args.require_all:
                raise FileNotFoundError(generated_path)
            continue
        mask_path = Path(view["mask"]) if view.get("mask") else None
        metrics, alignment = image_eval.evaluate_pair(
            ground_truth_path,
            generated_path,
            mask_path,
            args.alignment_mode,
        )
        row = {
            "view": int(view["index"]),
            "pose_source": view["pose_source"],
            **metrics,
            "alignment": alignment,
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No generated target images found in {args.generated_dir}")

    summary = image_eval.evaluation_summary(
        manifest["scene_id"], manifest["mode"], rows, args.alignment_mode
    )
    output = args.output or args.generated_dir.parent / "image_metrics.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
