#!/usr/bin/env python3
"""Build a deterministic 300-train/30-validation split for GeoDiff-VGGT.

The existing 30-scene test protocol is copied verbatim and never sampled by
this script.  Development scenes come only from co3d_completion_train and are
balanced across the 14 categories as evenly as possible.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def balanced_take(groups: dict[str, list[Path]], total: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    cats = sorted(groups)
    chosen: list[Path] = []
    cursor = Counter()
    while len(chosen) < total:
        progressed = False
        for cat in cats:
            pos = cursor[cat]
            if pos < len(groups[cat]) and len(chosen) < total:
                chosen.append(groups[cat][pos])
                cursor[cat] += 1
                progressed = True
        if not progressed:
            raise RuntimeError(f"only found {len(chosen)} eligible scenes; requested {total}")
    rng.shuffle(chosen)
    return chosen


def make_manifest(scene_dir: Path, slots: int) -> dict:
    cameras = json.loads((scene_dir / "cameras.json").read_text())
    keys = sorted(cameras, key=lambda x: int(x))
    if len(keys) < slots:
        raise ValueError(f"{scene_dir.name}: only {len(keys)} camera records")
    picks = [round(i * (len(keys) - 1) / (slots - 1)) for i in range(slots)]
    views = []
    for index, source_pos in enumerate(picks):
        key = keys[source_pos]
        camera = cameras[key]
        image = scene_dir / "images" / f"{key}.jpg"
        mask = scene_dir / "masks" / f"{key}.png"
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"missing image/mask for {scene_dir.name}/{key}")
        views.append(
            {
                "index": index,
                "original_index": int(key),
                "image": str(image),
                "ground_truth": str(image),
                "mask": str(mask),
                "extrinsic": camera["extrinsic"],
                "intrinsic": camera["intrinsic"],
                "height": camera["height"],
                "width": camera["width"],
                "frame_number": camera.get("frame_number"),
                "co3d_path": camera.get("co3d_path"),
                "pose_source": "co3d_completion_train/cameras.json",
            }
        )
    return {
        "scene_id": scene_dir.name,
        "category": scene_dir.name.split("_", 1)[0],
        "mode": "geodiff_vggt_1real7target",
        "dataset_source": str(scene_dir),
        "input_indices": [0],
        "target_indices": list(range(1, slots)),
        "views": views,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("/root/vggt/co3d_completion_train"))
    p.add_argument("--blind-protocol", type=Path, default=Path("/root/seva_vggt/config/test30_mvgen/selection_protocol.json"))
    p.add_argument("--output", type=Path, default=Path("/root/seva_vggt/config/geodiff_vggt_300"))
    p.add_argument("--train-scenes", type=int, default=300)
    p.add_argument("--validation-scenes", type=int, default=30)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261029)
    args = p.parse_args()

    blind = json.loads(args.blind_protocol.read_text())
    blind_scenes = set(blind["scenes"])
    groups: dict[str, list[Path]] = defaultdict(list)
    rejected = []
    for scene in sorted(args.source.iterdir()):
        if not scene.is_dir() or scene.name in blind_scenes:
            continue
        try:
            cameras = json.loads((scene / "cameras.json").read_text())
            if len(cameras) < args.views:
                raise ValueError("too few cameras")
            groups[scene.name.split("_", 1)[0]].append(scene)
        except Exception as exc:  # recorded instead of silently discarded
            rejected.append({"scene": scene.name, "reason": str(exc)})

    train = balanced_take({k: list(v) for k, v in groups.items()}, args.train_scenes, args.seed)
    train_names = {p.name for p in train}
    remaining = {k: [p for p in values if p.name not in train_names] for k, values in groups.items()}
    val = balanced_take(remaining, args.validation_scenes, args.seed + 1)
    selected = train + val
    args.output.mkdir(parents=True, exist_ok=True)
    for split, scenes in (("train", train), ("validation", val)):
        split_dir = args.output / split
        split_dir.mkdir(exist_ok=True)
        for scene in scenes:
            manifest = make_manifest(scene, args.views)
            (split_dir / f"{scene.name}.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )

    protocol = {
        "protocol": "geodiff_vggt_300_v1",
        "seed": args.seed,
        "train_scenes": [p.name for p in train],
        "validation_scenes": [p.name for p in val],
        "blind_test_protocol": str(args.blind_protocol),
        "blind_test_scenes": blind["scenes"],
        "train_category_counts": dict(Counter(p.name.split("_", 1)[0] for p in train)),
        "validation_category_counts": dict(Counter(p.name.split("_", 1)[0] for p in val)),
        "blind_category_counts": blind.get("category_counts"),
        "scene_overlap_checks": {
            "train_validation": len(set(p.name for p in train) & set(p.name for p in val)),
            "development_blind": len(set(p.name for p in selected) & blind_scenes),
        },
        "rejected": rejected,
    }
    (args.output / "selection_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
