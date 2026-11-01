#!/usr/bin/env python3
"""Build a SEVA/VGGT scene manifest without changing the source datasets.

The legacy 20-view dataset contains RGB images only.  The newer completion
dataset contains 32 views with cameras and masks from the same CO3D sequence.
Both were sampled uniformly from that sequence, so camera poses for the legacy
20 views can be approximated by interpolating the 32-view camera trajectory.
The manifest records whether every pose is exact or interpolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def as_homogeneous(extrinsic: Any) -> np.ndarray:
    matrix = np.asarray(extrinsic, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3] = matrix
        return result
    raise ValueError(f"Expected 3x4 or 4x4 extrinsic, got {matrix.shape}")


def interpolate_w2c(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate world-to-camera matrices via camera-to-world SE(3)."""
    if alpha <= 1e-12:
        return left.copy()
    if alpha >= 1.0 - 1e-12:
        return right.copy()
    left_c2w = np.linalg.inv(left)
    right_c2w = np.linalg.inv(right)
    rotations = Rotation.from_matrix(
        np.stack([left_c2w[:3, :3], right_c2w[:3, :3]], axis=0)
    )
    rotation = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
    translation = (
        (1.0 - alpha) * left_c2w[:3, 3] + alpha * right_c2w[:3, 3]
    )
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation
    return np.linalg.inv(c2w)


def load_camera_track(scene: Path) -> list[dict[str, Any]]:
    camera_file = scene / "cameras.json"
    if not camera_file.is_file():
        raise FileNotFoundError(f"Missing camera file: {camera_file}")
    raw = json.loads(camera_file.read_text())
    keys = sorted(raw, key=lambda value: int(value))
    track = []
    for key in keys:
        entry = raw[key]
        track.append(
            {
                "key": key,
                "w2c": as_homogeneous(entry["extrinsic"]),
                "K": np.asarray(entry["intrinsic"], dtype=np.float64),
                "height": int(entry["height"]),
                "width": int(entry["width"]),
                "frame_number": int(entry["frame_number"]),
                "co3d_path": entry.get("co3d_path"),
            }
        )
    if len(track) < 2:
        raise ValueError(f"Camera track is too short: {camera_file}")
    return track


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_camera(entry: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "extrinsic": entry["w2c"][:3].tolist(),
        "intrinsic": entry["K"].tolist(),
        "height": entry["height"],
        "width": entry["width"],
        "frame_number_estimate": entry["frame_number"],
        "pose_source": source,
        "camera_neighbors": [entry["key"], entry["key"]],
        "camera_alpha": 0.0,
    }


def camera_at(track: list[dict[str, Any]], normalized_position: float) -> dict[str, Any]:
    continuous = normalized_position * (len(track) - 1)
    left_index = int(math.floor(continuous))
    right_index = int(math.ceil(continuous))
    alpha = float(continuous - left_index)
    left = track[left_index]
    right = track[right_index]
    w2c = interpolate_w2c(left["w2c"], right["w2c"], alpha)
    intrinsics = (1.0 - alpha) * left["K"] + alpha * right["K"]
    exact = left_index == right_index or alpha <= 1e-12
    frame_number = int(round((1.0 - alpha) * left["frame_number"] + alpha * right["frame_number"]))
    return {
        "extrinsic": w2c[:3].tolist(),
        "intrinsic": intrinsics.tolist(),
        "height": int(round((1.0 - alpha) * left["height"] + alpha * right["height"])),
        "width": int(round((1.0 - alpha) * left["width"] + alpha * right["width"])),
        "frame_number_estimate": frame_number,
        "pose_source": "exact_completion_camera" if exact else "interpolated_completion_track",
        "camera_neighbors": [left["key"], right["key"]],
        "camera_alpha": alpha,
    }


def uniform_ring_indices(count: int, selected: int) -> list[int]:
    if selected < 1 or selected >= count:
        raise ValueError(f"input-count must be in [1, {count - 1}], got {selected}")
    indices = [int(math.floor(index * count / selected)) for index in range(selected)]
    if len(set(indices)) != selected:
        raise RuntimeError(f"Could not choose {selected} unique indices from {count}")
    return indices


def exact_subset_indices(available: int, requested: int) -> list[int]:
    if requested > available:
        raise ValueError(f"Requested {requested} views but only {available} exist")
    values = np.linspace(0, available - 1, requested)
    indices = [int(round(value)) for value in values]
    if len(set(indices)) != requested:
        raise RuntimeError("Exact subset selection produced duplicate indices")
    return indices


def build_interpolated_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rgb_scene = args.image_root / args.scene
    camera_scene = args.camera_root / args.scene
    images = image_files(rgb_scene)
    if len(images) != args.view_count:
        raise ValueError(
            f"{rgb_scene} has {len(images)} images; expected exactly {args.view_count}"
        )
    track = load_camera_track(camera_scene)
    completion_images = camera_scene / "images"
    digest_to_track_index = {
        file_digest(completion_images / f"{entry['key']}.jpg"): track_index
        for track_index, entry in enumerate(track)
        if (completion_images / f"{entry['key']}.jpg").is_file()
    }
    views = []
    for index, image in enumerate(images):
        normalized = index / (len(images) - 1)
        matched_index = digest_to_track_index.get(file_digest(image))
        camera = (
            exact_camera(track[matched_index], "exact_image_hash_match")
            if matched_index is not None
            else camera_at(track, normalized)
        )
        exact = camera["pose_source"].startswith("exact")
        mask = camera_scene / "masks" / f"{camera['camera_neighbors'][0]}.png"
        views.append(
            {
                "index": index,
                "original_index": index,
                "image": str(image),
                "ground_truth": str(image),
                "mask": str(mask) if exact and mask.is_file() else None,
                "mask_source": "exact_completion_mask" if exact and mask.is_file() else "unavailable_for_interpolated_pose",
                **camera,
            }
        )
    return {
        "mode": "interpolated20",
        "pose_assumption": (
            "The 20-view and completion datasets are deterministic uniform samples "
            "of the same ordered CO3D sequence. Intermediate poses are SE(3) "
            "interpolations of the completion camera track."
        ),
        "views": views,
    }


def build_exact_manifest(args: argparse.Namespace) -> dict[str, Any]:
    scene = args.camera_root / args.scene
    track = load_camera_track(scene)
    selected = exact_subset_indices(len(track), args.view_count)
    views = []
    for manifest_index, original_index in enumerate(selected):
        camera = track[original_index]
        image = scene / "images" / f"{camera['key']}.jpg"
        mask = scene / "masks" / f"{camera['key']}.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        views.append(
            {
                "index": manifest_index,
                "original_index": original_index,
                "image": str(image),
                "ground_truth": str(image),
                "mask": str(mask) if mask.is_file() else None,
                "mask_source": "exact_completion_mask" if mask.is_file() else None,
                "extrinsic": camera["w2c"][:3].tolist(),
                "intrinsic": camera["K"].tolist(),
                "height": camera["height"],
                "width": camera["width"],
                "frame_number_estimate": camera["frame_number"],
                "pose_source": "exact_completion_camera",
                "camera_neighbors": [camera["key"], camera["key"]],
                "camera_alpha": 0.0,
            }
        )
    return {
        "mode": "exact20_from_completion",
        "pose_assumption": "All RGB images, masks, intrinsics and extrinsics are exact completion-dataset entries.",
        "views": views,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", help="Scene directory name")
    parser.add_argument(
        "--mode",
        choices=("interpolated20", "exact20_from_completion"),
        default="interpolated20",
    )
    parser.add_argument("--image-root", type=Path, default=Path("/root/vggt/my_train_20v"))
    parser.add_argument(
        "--camera-root", type=Path, default=Path("/root/vggt/co3d_completion_train")
    )
    parser.add_argument("--view-count", type=int, default=20)
    parser.add_argument("--input-count", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = (
        build_interpolated_manifest(args)
        if args.mode == "interpolated20"
        else build_exact_manifest(args)
    )
    input_indices = uniform_ring_indices(len(payload["views"]), args.input_count)
    target_indices = [
        index for index in range(len(payload["views"])) if index not in input_indices
    ]
    payload.update(
        {
            "schema_version": 1,
            "scene_id": args.scene,
            "input_indices": input_indices,
            "target_indices": target_indices,
            "coordinate_convention": "extrinsic is OpenCV world-to-camera [R|t]; pipeline converts it to camera-to-world",
        }
    )
    output = args.output or Path("/root/seva_vggt/config/scenes") / f"{args.scene}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    exact = sum(view["pose_source"].startswith("exact") for view in payload["views"])
    print(
        json.dumps(
            {
                "manifest": str(output),
                "scene": args.scene,
                "mode": payload["mode"],
                "views": len(payload["views"]),
                "inputs": input_indices,
                "targets": len(target_indices),
                "exact_poses": exact,
                "interpolated_poses": len(payload["views"]) - exact,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
