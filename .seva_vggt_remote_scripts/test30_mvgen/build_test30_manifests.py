#!/usr/bin/env python3
"""Select a balanced 30-scene held-out set and recover its closest saved cameras."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PREVIOUS = {
    "chair_386_46082_92005", "car_599_92296_183295",
    "couch_349_36574_67726", "toybus_517_74123_144462",
    "ball_113_13373_24790",
}


def thumb(path: Path, size: int = 96) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def monotonic_match(test_images: list[Path], reference_images: list[Path]) -> tuple[list[int], list[float]]:
    """Minimum-cost strictly increasing assignment of 8 test images to 24 reference frames."""
    a = np.stack([thumb(p) for p in test_images])
    b = np.stack([thumb(p) for p in reference_images])
    cost = np.mean((a[:, None] - b[None]) ** 2, axis=(2, 3, 4))
    n, m = cost.shape
    dp = np.full((n, m), np.inf, dtype=np.float64)
    parent = np.full((n, m), -1, dtype=np.int32)
    dp[0] = cost[0]
    for i in range(1, n):
        best_value, best_index = np.inf, -1
        for j in range(m):
            if j > 0 and dp[i - 1, j - 1] < best_value:
                best_value, best_index = dp[i - 1, j - 1], j - 1
            if best_index >= 0:
                dp[i, j] = best_value + cost[i, j]
                parent[i, j] = best_index
    end = int(np.argmin(dp[-1]))
    indices = [end]
    for i in range(n - 1, 0, -1):
        end = int(parent[i, end])
        indices.append(end)
    indices.reverse()
    return indices, [float(cost[i, j]) for i, j in enumerate(indices)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", type=Path, default=Path("/root/vggt/my_test_multiview"))
    ap.add_argument("--camera-root", type=Path, default=Path("/root/vggt/co3d_completion_test"))
    ap.add_argument("--output", type=Path, default=Path("/root/seva_vggt/config/test30_mvgen"))
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()
    groups: dict[str, list[str]] = defaultdict(list)
    missing = []
    for scene_dir in sorted(args.test_root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name in PREVIOUS:
            continue
        if not (args.camera_root / scene_dir.name / "cameras.json").exists():
            missing.append(scene_dir.name)
            continue
        groups[scene_dir.name.split("_")[0]].append(scene_dir.name)
    rng = random.Random(args.seed)
    selected = []
    for category in sorted(groups):
        choices = sorted(groups[category])
        rng.shuffle(choices)
        selected.extend(choices[:2])
    # 14 categories -> 28; add one chair and one plant while preserving deterministic sampling.
    for category in ("chair", "plant"):
        choices = [s for s in groups[category] if s not in selected]
        rng.shuffle(choices)
        selected.append(choices[0])
    if len(selected) != 30 or len(set(selected)) != 30:
        raise RuntimeError(f"Expected 30 unique scenes, got {len(selected)}")
    selected = sorted(selected, key=lambda s: (s.split("_")[0], s))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for scene in selected:
        test_dir = args.test_root / scene
        camera_dir = args.camera_root / scene
        test_images = sorted(test_dir.glob("*.jpg"))
        ref_images = sorted((camera_dir / "images").glob("*.jpg"))
        if len(test_images) != 8 or len(ref_images) != 24:
            raise RuntimeError(f"{scene}: expected 8/24 frames, got {len(test_images)}/{len(ref_images)}")
        matched, costs = monotonic_match(test_images, ref_images)
        cameras = json.loads((camera_dir / "cameras.json").read_text())
        views = []
        for slot, (test_image, ref_pos, mse) in enumerate(zip(test_images, matched, costs)):
            key = ref_images[ref_pos].stem
            camera = cameras[key]
            test_w, test_h = Image.open(test_image).size
            views.append({
                "index": slot,
                "original_index": int(key),
                "image": str(test_image),
                "ground_truth": str(test_image),
                "mask": str(camera_dir / "masks" / f"{key}.png"),
                "camera_reference_image": str(ref_images[ref_pos]),
                "camera_match_mse_96": mse,
                "extrinsic": camera["extrinsic"],
                "intrinsic": camera["intrinsic"],
                "height": camera["height"],
                "width": camera["width"],
                "test_image_height": test_h,
                "test_image_width": test_w,
                "frame_number": camera.get("frame_number"),
                "pose_source": "closest_monotonic_frame_in_co3d_completion_test",
            })
        payload = {
            "scene_id": scene,
            "category": scene.split("_")[0],
            "mode": "my_test_multiview_8_mvgen_1real7gen",
            "dataset_source": str(test_dir),
            "camera_source": str(camera_dir),
            "camera_mapping_note": "my_test_multiview stores RGB only; cameras are recovered by minimum-MSE monotonic matching to the 24-frame completion-test copy of the same CO3D sequence.",
            "input_indices": [0], "target_indices": list(range(1, 8)), "views": views,
        }
        (args.output / f"{scene}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        records.append({"scene": scene, "category": payload["category"], "matched_reference_indices": matched,
                        "mean_match_mse_96": float(np.mean(costs)), "max_match_mse_96": float(np.max(costs))})
    protocol = {
        "seed": args.seed, "scene_count": len(selected), "scenes": selected,
        "category_counts": {c: sum(r["category"] == c for r in records) for c in sorted(groups)},
        "excluded_previous_small5": sorted(PREVIOUS), "missing_camera_scenes_excluded": missing,
        "selection": "2 scenes per each of 14 categories, plus one extra chair and plant; deterministic seeded shuffle; previous small5 excluded.",
        "camera_mapping": "strictly increasing dynamic-programming match minimizing 96x96 RGB MSE against 24 saved completion frames",
        "records": records,
    }
    (args.output / "selection_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"scene_count": len(selected), "category_counts": protocol["category_counts"],
                      "mean_mapping_mse": float(np.mean([r["mean_match_mse_96"] for r in records])),
                      "max_mapping_mse": float(np.max([r["max_match_mse_96"] for r in records]))}, indent=2))


if __name__ == "__main__":
    main()
