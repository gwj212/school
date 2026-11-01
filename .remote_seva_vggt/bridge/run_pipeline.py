#!/usr/bin/env python3
"""Run camera-conditioned SEVA generation followed by unchanged VGGT-G reconstruction."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

import metric_alignment as image_eval


ROOT = Path("/root/seva_vggt")
DEFAULT_VGGT_MODEL = Path(
    "/root/vggt/ckpts/models--facebook--VGGT-1B/snapshots/"
    "860abec7937da0a4c03c41d3c269c366e82abdf9"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--seva-root", type=Path, default=ROOT / "seva")
    parser.add_argument("--vggt-root", type=Path, default=Path("/root/vggt"))
    parser.add_argument(
        "--seva-model", type=Path, default=ROOT / "models/stable-virtual-camera"
    )
    parser.add_argument("--seva-version", type=float, default=1.1)
    parser.add_argument("--vggt-model", type=Path, default=DEFAULT_VGGT_MODEL)
    parser.add_argument(
        "--vggt-checkpoint", type=Path, default=Path("/root/vggt/ckpts/ckpt.pth")
    )
    parser.add_argument("--vae-model", type=Path, default=ROOT / "models/sd-vae-ft-mse")
    parser.add_argument(
        "--clip-weights",
        type=Path,
        default=ROOT
        / "models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seva-size",
        type=int,
        default=None,
        help="Legacy square canvas. If set, overrides --seva-short-side.",
    )
    parser.add_argument(
        "--seva-short-side",
        type=int,
        default=576,
        help="Mode-A native-aspect short side; 576 matches the official SEVA demo.",
    )
    parser.add_argument("--seva-context", type=int, default=21)
    parser.add_argument("--chunk-strategy", default="nearest-gt")
    parser.add_argument("--vggt-max-side", type=int, default=294)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument(
        "--single-cfg",
        type=float,
        default=4.0,
        help="Classifier-free guidance used only when there is one real input.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--target-count",
        type=int,
        default=8,
        help="Number of held-out cameras SEVA generates; 0 uses every manifest target.",
    )
    parser.add_argument(
        "--input-count",
        type=int,
        help="Real inputs in a fixed-total split; requires --total-count.",
    )
    parser.add_argument(
        "--total-count",
        type=int,
        help="Total real+generated reconstruction views selected uniformly from the scene.",
    )
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--comparison-points", type=int, default=20_000)
    parser.add_argument(
        "--reuse-real-baseline",
        action="store_true",
        help="Reuse an existing realN point cloud/metrics instead of reconstructing it again.",
    )
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_file_size(path: Path, expected: int, description: str) -> None:
    require_path(path, description)
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(
            f"Incomplete {description}: {path} has {actual} bytes; expected {expected}"
        )


def as_w2c(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float32)
        result[:3] = matrix
        return result
    raise ValueError(f"Bad extrinsic shape: {matrix.shape}")


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    views = payload["views"]
    inputs = [views[index] for index in payload["input_indices"]]
    targets = [views[index] for index in payload["target_indices"]]
    if not inputs or not targets:
        raise ValueError("Manifest must contain at least one input and one target view")
    for view in views:
        require_path(Path(view["image"]), "view image")
        as_w2c(view["extrinsic"])
        intrinsic = np.asarray(view["intrinsic"])
        if intrinsic.shape != (3, 3):
            raise ValueError(f"Bad intrinsic shape for view {view['index']}: {intrinsic.shape}")
    return payload, inputs, targets


def select_targets(
    payload: dict[str, Any], targets: list[dict[str, Any]], requested: int
) -> list[dict[str, Any]]:
    """Uniformly select held-out target cameras around the ordered ring."""
    if requested < 0:
        raise ValueError("--target-count must be >= 0")
    if requested == 0 or requested >= len(targets):
        selected = list(targets)
    else:
        positions = [
            min(len(targets) - 1, int(math.floor((index + 0.5) * len(targets) / requested)))
            for index in range(requested)
        ]
        selected = [targets[position] for position in positions]
    if len(payload["input_indices"]) + len(selected) > 21:
        raise ValueError("SEVA fast pass supports at most 21 total frames")
    return selected


def uniform_indices(count: int, selected: int) -> list[int]:
    if selected < 1 or selected > count:
        raise ValueError(f"selected count must be in [1, {count}], got {selected}")
    indices = [int(math.floor(index * count / selected)) for index in range(selected)]
    if len(set(indices)) != selected:
        raise RuntimeError(f"Could not select {selected} unique indices from {count}")
    return indices


def select_fixed_total_split(
    payload: dict[str, Any], input_count: int, total_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep total cameras fixed while varying the real/SEVA composition."""
    views = payload["views"]
    if total_count > 21:
        raise ValueError("SEVA fast pass supports at most 21 total frames")
    if input_count < 1 or input_count >= total_count:
        raise ValueError("--input-count must be in [1, total-count - 1]")
    slot_indices = uniform_indices(len(views), total_count)
    baseline_views = [views[index] for index in slot_indices]
    input_positions = set(uniform_indices(total_count, input_count))
    inputs = [view for position, view in enumerate(baseline_views) if position in input_positions]
    targets = [view for position, view in enumerate(baseline_views) if position not in input_positions]
    return baseline_views, inputs, targets


def tensor_cameras(views: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    c2ws = [np.linalg.inv(as_w2c(view["extrinsic"])) for view in views]
    intrinsics = [np.asarray(view["intrinsic"], dtype=np.float32) for view in views]
    return torch.from_numpy(np.stack(c2ws)).float(), torch.from_numpy(np.stack(intrinsics)).float()


def save_generated(images: torch.Tensor, targets: list[dict[str, Any]], directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    images = ((images.detach().cpu() + 1.0) * 127.5).clamp(0, 255).byte()
    if len(images) != len(targets):
        raise RuntimeError(
            f"SEVA returned {len(images)} frames for {len(targets)} target cameras"
        )
    paths = []
    for image, target in zip(images, targets):
        path = directory / f"view_{int(target['index']):02d}.png"
        Image.fromarray(image.permute(1, 2, 0).numpy()).save(path)
        paths.append(str(path))
    return paths


def save_branch_images(
    views: list[dict[str, Any]],
    directory: Path,
    generated_by_index: dict[int, str] | None = None,
) -> list[str]:
    """Copy a branch's exact reconstruction inputs into a self-contained folder."""
    directory.mkdir(parents=True, exist_ok=True)
    generated_by_index = generated_by_index or {}
    paths = []
    for view in sorted(views, key=lambda item: int(item["index"])):
        index = int(view["index"])
        generated = index in generated_by_index
        source = Path(generated_by_index[index] if generated else view["image"])
        suffix = source.suffix.lower() if source.suffix else ".png"
        role = "generated" if generated else "real"
        destination = directory / f"view_{index:02d}_{role}{suffix}"
        shutil.copy2(source, destination)
        paths.append(str(destination))
    return paths


def read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    squared = (reference - prediction) ** 2
    if mask is not None:
        squared = squared[mask]
        if squared.size == 0:
            return float("nan")
    mse = float(squared.mean())
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def evaluate_generated(
    payload: dict[str, Any],
    targets: list[dict[str, Any]],
    generated_by_index: dict[int, str],
    alignment_mode: str = "crop",
) -> dict[str, Any]:
    rows = []
    for view in targets:
        index = int(view["index"])
        ground_truth_path = Path(view["ground_truth"])
        generated_path = Path(generated_by_index[index])
        mask_path = Path(view["mask"]) if view.get("mask") else None
        metrics, alignment = image_eval.evaluate_pair(
            ground_truth_path,
            generated_path,
            mask_path,
            alignment_mode,
        )
        row: dict[str, Any] = {
            "view": index,
            "pose_source": view["pose_source"],
            **metrics,
            "alignment": alignment,
        }
        rows.append(row)
    return image_eval.evaluation_summary(
        payload["scene_id"], payload["mode"], rows, alignment_mode
    )


def write_pointcloud_ply(
    path: Path, xyz: np.ndarray, color: np.ndarray, opacity: np.ndarray
) -> None:
    """Write a compact binary PLY containing Gaussian centers as colored points."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    rgb = (np.clip(color, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    vertices["alpha"] = (np.clip(opacity.reshape(-1), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar alpha\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def read_pointcloud_xyz(path: Path) -> np.ndarray:
    """Read XYZ from the compact binary PLY emitted by write_pointcloud_ply."""
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header: {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Unsupported PLY format: {path}")
        count = int(next(line.split()[-1] for line in header if line.startswith("element vertex ")))
        dtype = np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
            ]
        )
        data = np.fromfile(handle, dtype=dtype, count=count)
    return np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)


def gaussian_statistics(
    gaussians: dict[str, Any],
    opacity_threshold: float,
    max_points: int,
    seed: int,
    pointcloud_path: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    xyz = gaussians["xyz"].detach().float().cpu().numpy().reshape(-1, 3)
    color = gaussians["color"].detach().float().cpu().numpy().reshape(-1, 3)
    opacity = gaussians["opacity"].detach().float().cpu().numpy().reshape(-1, 1)
    scale = gaussians["scale"].detach().float().cpu().numpy().reshape(-1, 3)
    finite = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(color).all(axis=1)
        & np.isfinite(opacity).all(axis=1)
        & np.isfinite(scale).all(axis=1)
    )
    keep = finite & (opacity[:, 0] >= opacity_threshold)
    kept_indices = np.flatnonzero(keep)
    above_threshold = int(len(kept_indices))
    if max_points > 0 and len(kept_indices) > max_points:
        rng = np.random.default_rng(seed)
        kept_indices = np.sort(rng.choice(kept_indices, max_points, replace=False))
    if not len(kept_indices):
        raise RuntimeError(
            f"No finite Gaussians remain at opacity threshold {opacity_threshold}"
        )
    exported_xyz = xyz[kept_indices]
    write_pointcloud_ply(
        pointcloud_path, exported_xyz, color[kept_indices], opacity[kept_indices]
    )
    bbox_min = exported_xyz.min(axis=0)
    bbox_max = exported_xyz.max(axis=0)
    finite_opacity = opacity[finite, 0]
    finite_scale = scale[finite]
    metrics = {
        "raw_gaussians": int(len(xyz)),
        "finite_gaussians": int(finite.sum()),
        "above_opacity_threshold": above_threshold,
        "exported_points": int(len(kept_indices)),
        "opacity_threshold": float(opacity_threshold),
        "opacity": {
            "mean": float(finite_opacity.mean()),
            "p50": float(np.quantile(finite_opacity, 0.5)),
            "p90": float(np.quantile(finite_opacity, 0.9)),
            "max": float(finite_opacity.max()),
        },
        "scale": {
            "mean": [float(value) for value in finite_scale.mean(axis=0)],
            "p50": [float(value) for value in np.quantile(finite_scale, 0.5, axis=0)],
        },
        "bbox_min": [float(value) for value in bbox_min],
        "bbox_max": [float(value) for value in bbox_max],
        "bbox_diagonal": float(np.linalg.norm(bbox_max - bbox_min)),
        "point_cloud": str(pointcloud_path),
    }
    return metrics, exported_xyz


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def resize_vggt_images(images: torch.Tensor, max_side: int) -> torch.Tensor:
    height, width = images.shape[-2:]
    if max_side <= 0 or max(height, width) <= max_side:
        return images
    scale = max_side / max(height, width)
    target_height = max(14, int(round(height * scale / 14)) * 14)
    target_width = max(14, int(round(width * scale / 14)) * 14)
    return torch.nn.functional.interpolate(
        images,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def load_reconstruction_checkpoint(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.gaussian_head.load_state_dict(
        state["gaussian_head_state_dict"], strict=False
    )
    if "dpt_feature_head_state_dict" in state:
        model.dpt_feature_head.load_state_dict(
            state["dpt_feature_head_state_dict"], strict=False
        )
    print(
        f"VGGT checkpoint step={state.get('global_step')} "
        f"gaussian_missing={len(missing)} gaussian_unexpected={len(unexpected)}"
    )
    return {
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "metrics": state.get("metrics"),
    }


def reconstruct_branch(
    model: torch.nn.Module,
    image_paths: list[str],
    branch_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_side: int,
    opacity_threshold: float,
    max_points: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    from vggt.utils.load_fn import load_and_preprocess_images

    branch_started = time.perf_counter()
    preprocess_started = time.perf_counter()
    reconstruction_input = load_and_preprocess_images(image_paths, mode="pad")
    reconstruction_input = resize_vggt_images(reconstruction_input, max_side).to(device)
    preprocess_seconds = time.perf_counter() - preprocess_started
    inference_started = time.perf_counter()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        reconstruction = model(reconstruction_input.unsqueeze(0))
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    export_started = time.perf_counter()
    torch.save(cpu_tree(reconstruction), branch_dir / "vggt_reconstruction.pt")
    reconstruction_metrics, exported_xyz = gaussian_statistics(
        reconstruction["gaussians"],
        opacity_threshold,
        max_points,
        seed,
        branch_dir / "point_cloud.ply",
    )
    export_seconds = time.perf_counter() - export_started
    reconstruction_metrics.update(
        {
            "input_count": len(image_paths),
            "input_shape": list(reconstruction_input.shape),
            "output_keys": list(reconstruction),
            "elapsed_seconds": time.perf_counter() - branch_started,
            "timing_seconds": {
                "input_preprocess": preprocess_seconds,
                "model_inference": inference_seconds,
                "serialize_and_export": export_seconds,
            },
        }
    )
    del reconstruction, reconstruction_input
    torch.cuda.empty_cache()
    return reconstruction_metrics, exported_xyz


def compare_pointclouds(
    reference: np.ndarray,
    prediction: np.ndarray,
    sample_count: int,
    seed: int,
    reference_name: str,
    prediction_name: str,
) -> dict[str, Any]:
    """Compare same-coordinate reconstructions with deterministic NN samples."""
    rng = np.random.default_rng(seed)

    def sample(points: np.ndarray) -> np.ndarray:
        if sample_count <= 0 or len(points) <= sample_count:
            return points
        return points[rng.choice(len(points), sample_count, replace=False)]

    ref = sample(reference)
    pred = sample(prediction)
    pred_to_ref = cKDTree(ref).query(pred, k=1, workers=-1)[0]
    ref_to_pred = cKDTree(pred).query(ref, k=1, workers=-1)[0]
    diagonal = float(np.linalg.norm(ref.max(axis=0) - ref.min(axis=0)))
    symmetric_mean = float((pred_to_ref.mean() + ref_to_pred.mean()) / 2.0)
    symmetric_l2 = float(((pred_to_ref ** 2).mean() + (ref_to_pred ** 2).mean()) / 2.0)
    normalized_pred_to_ref = pred_to_ref / diagonal if diagonal > 0 else pred_to_ref
    normalized_ref_to_pred = ref_to_pred / diagonal if diagonal > 0 else ref_to_pred
    f_scores = {}
    for threshold in (0.005, 0.01, 0.02, 0.05):
        precision = float(np.mean(normalized_pred_to_ref <= threshold))
        recall = float(np.mean(normalized_ref_to_pred <= threshold))
        f_score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f_scores[f"{threshold:.3f}"] = {
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
        }
    return {
        "reference": reference_name,
        "prediction": prediction_name,
        "coordinate_alignment": "raw VGGT coordinates; both branches start with the same real view",
        "sampled_reference_points": int(len(ref)),
        "sampled_prediction_points": int(len(pred)),
        "prediction_to_reference_mean": float(pred_to_ref.mean()),
        "reference_to_prediction_mean": float(ref_to_pred.mean()),
        "prediction_to_reference_p50": float(np.quantile(pred_to_ref, 0.50)),
        "prediction_to_reference_p95": float(np.quantile(pred_to_ref, 0.95)),
        "reference_to_prediction_p50": float(np.quantile(ref_to_pred, 0.50)),
        "reference_to_prediction_p95": float(np.quantile(ref_to_pred, 0.95)),
        "symmetric_chamfer_l1": symmetric_mean,
        "symmetric_chamfer_l1_normalized": symmetric_mean / diagonal if diagonal > 0 else None,
        "symmetric_chamfer_l2": symmetric_l2,
        "symmetric_chamfer_l2_normalized": symmetric_l2 / (diagonal ** 2) if diagonal > 0 else None,
        "hausdorff_max": float(max(pred_to_ref.max(), ref_to_pred.max())),
        "hausdorff_max_normalized": float(max(normalized_pred_to_ref.max(), normalized_ref_to_pred.max())),
        "hausdorff_p95": float(max(np.quantile(pred_to_ref, 0.95), np.quantile(ref_to_pred, 0.95))),
        "hausdorff_p95_normalized": float(
            max(np.quantile(normalized_pred_to_ref, 0.95), np.quantile(normalized_ref_to_pred, 0.95))
        ),
        "f_scores_at_normalized_threshold": f_scores,
        "reference_bbox_diagonal": diagonal,
    }


def validate_paths(args: argparse.Namespace) -> None:
    require_path(args.manifest, "scene manifest")
    require_path(args.seva_root / "seva/eval.py", "modified SEVA source")
    require_file_size(
        args.seva_model / "modelv1.1.safetensors", 5_056_009_640, "SEVA v1.1 weight"
    )
    require_path(args.seva_model / "config.yaml", "SEVA config")
    require_file_size(
        args.vae_model / "diffusion_pytorch_model.safetensors", 334_643_276, "VAE weight"
    )
    require_path(args.vae_model / "config.json", "VAE config")
    require_file_size(args.clip_weights, 3_944_692_325, "CLIP weight")
    require_path(args.clip_weights.parent / "open_clip_config.json", "CLIP config")
    require_path(args.vggt_root / "vggt/models/vggt.py", "VGGT source")
    require_path(args.vggt_model / "model.safetensors", "VGGT-1B weight")
    require_path(args.vggt_checkpoint, "VGGT-G reconstruction checkpoint")


def main() -> None:
    args = parse_args()
    payload, manifest_inputs, manifest_targets = load_manifest(args.manifest)
    if (args.input_count is None) != (args.total_count is None):
        raise ValueError("--input-count and --total-count must be provided together")
    if args.total_count is not None:
        baseline_views, inputs, targets = select_fixed_total_split(
            payload, args.input_count, args.total_count
        )
    else:
        baseline_views = payload["views"]
        inputs = manifest_inputs
        targets = select_targets(payload, manifest_targets, args.target_count)
    validate_paths(args)
    output_dir = args.output_dir or (
        args.workspace / "outputs" / payload["scene_id"] / payload["mode"]
    )
    real_name = f"real{len(baseline_views)}"
    generated_name = f"seva_{len(inputs)}real_{len(targets)}gen"
    real_dir = output_dir / real_name
    generated_dir = output_dir / generated_name

    if args.dry_run:
        print(
            json.dumps(
                {
                    "ready": True,
                    "scene": payload["scene_id"],
                    "mode": payload["mode"],
                    "baseline_indices": [view["index"] for view in baseline_views],
                    "inputs": [view["index"] for view in inputs],
                    "target_indices": [view["index"] for view in targets],
                    "target_count": len(targets),
                    "real_output": str(real_dir),
                    "generated_output": str(generated_dir),
                    "cuda_available": torch.cuda.is_available(),
                },
                indent=2,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this container. Restore NVIDIA runtime access, "
            "verify nvidia-smi, then rerun the same command."
        )
    if args.seva_size is not None and args.seva_size % 64:
        raise ValueError("--seva-size must be divisible by 64")
    if args.seva_short_side % 64:
        raise ValueError("--seva-short-side must be divisible by 64")
    if args.seva_context < len(inputs) + len(targets):
        raise ValueError("--seva-context cannot be smaller than the requested frame count")
    if not 0.0 <= args.opacity_threshold <= 1.0:
        raise ValueError("--opacity-threshold must be in [0, 1]")
    if args.max_points < 1:
        raise ValueError("--max-points must be >= 1")

    os.environ.setdefault("HF_HOME", str(args.workspace / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
    os.environ["SEVA_WRITE_VIDEO"] = "0"
    os.environ["SEVA_VAE_MODEL"] = str(args.vae_model)
    os.environ["SEVA_CLIP_MODEL"] = str(args.clip_weights)
    sys.path.insert(0, str(args.seva_root))
    sys.path.insert(0, str(args.vggt_root))

    from seva.eval import run_one_scene, set_lowvram_mode
    from seva.model import SGMWrapper
    from seva.modules.autoencoder import AutoEncoder
    from seva.modules.conditioner import CLIPConditioner
    from seva.sampling import DiscreteDenoiser
    from seva.utils import load_model as load_seva_model

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    ordered_views = inputs + targets
    all_c2ws, all_intrinsics = tensor_cameras(ordered_views)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats(device)

    stage_started = time.perf_counter()
    set_lowvram_mode(True)
    seva_model = SGMWrapper(
        load_seva_model(
            model_version=args.seva_version,
            pretrained_model_name_or_path=str(args.seva_model),
            device="cpu",
            dtype=dtype,
        ).eval()
    )
    autoencoder = AutoEncoder(chunk_size=1).eval()
    conditioner = CLIPConditioner().eval()
    denoiser = DiscreteDenoiser(num_idx=1000, device=str(device))
    timings["seva_model_load"] = time.perf_counter() - stage_started
    total_frames = len(ordered_views)
    effective_cfg = args.single_cfg if len(inputs) == 1 else args.cfg
    square_canvas = args.seva_size is not None
    base_size = args.seva_size if square_canvas else args.seva_short_side
    version = {
        "H": base_size,
        "W": base_size,
        "T": args.seva_context,
        "C": 4,
        "f": 8,
        "options": {
            "chunk_strategy": args.chunk_strategy,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": 1,
            "cfg": effective_cfg,
            "camera_scale": 2.0,
            "num_steps": args.steps,
            "cfg_min": 1.2,
            "encoding_t": 1,
            "decoding_t": 1,
            "save_input": False,
            "save_output": False,
            "write_video": False,
            "return_tensors": True,
            "L_short": -1 if square_canvas else args.seva_short_side,
            "transform_input": "crop",
            "transform_target": "crop",
        },
    }
    image_cond = {
        "img": [view["image"] for view in inputs] + [None] * len(targets),
        "input_indices": list(range(len(inputs))),
        "prior_indices": None,
    }
    camera_cond = {
        "c2w": all_c2ws.clone(),
        "K": all_intrinsics.clone(),
        "input_indices": list(range(total_frames)),
    }
    torch.cuda.synchronize(device)
    stage_started = time.perf_counter()
    result = next(
        run_one_scene(
            "img2img",
            version,
            model=seva_model,
            ae=autoencoder,
            conditioner=conditioner,
            denoiser=denoiser,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=str(generated_dir / "seva_internal"),
            use_traj_prior=False,
            traj_prior_Ks=None,
            traj_prior_c2ws=None,
            seed=args.seed,
        )
    )
    generated = next(
        value for key, value in result.items() if key.startswith("samples-rgb")
    ).detach().cpu()
    torch.cuda.synchronize(device)
    timings["seva_generation"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    generated_paths = save_generated(generated, targets, generated_dir / "generated_only")
    generated_by_index = {
        int(view["index"]): path for view, path in zip(targets, generated_paths)
    }
    real_image_paths = save_branch_images(baseline_views, real_dir / "images")
    generated_views = sorted(inputs + targets, key=lambda view: int(view["index"]))
    generated_input_paths = save_branch_images(
        generated_views, generated_dir / "images", generated_by_index
    )
    image_metrics = evaluate_generated(
        payload, targets, generated_by_index, alignment_mode="crop"
    )
    (generated_dir / "image_metrics.json").write_text(
        json.dumps(image_metrics, indent=2)
    )
    timings["generated_outputs_and_image_metrics"] = time.perf_counter() - stage_started
    del seva_model, autoencoder, conditioner, denoiser, result, generated
    torch.cuda.empty_cache()

    metadata: dict[str, Any] = {
        "scene_id": payload["scene_id"],
        "manifest": str(args.manifest),
        "baseline_indices": [view["index"] for view in baseline_views],
        "input_indices": [view["index"] for view in inputs],
        "target_indices": [view["index"] for view in targets],
        "generated_paths": generated_paths,
        "real_branch": str(real_dir),
        "generated_branch": str(generated_dir),
        "seva_steps": args.steps,
        "seva_cfg": effective_cfg,
        "seva_multi_view_cfg": args.cfg,
        "seva_single_view_cfg": args.single_cfg,
        "seva_context": args.seva_context,
        "seva_chunk_strategy": args.chunk_strategy,
        "seva_short_side": None if square_canvas else args.seva_short_side,
        "seva_square_size": args.seva_size,
        "seva_output_canvas": list(Image.open(generated_paths[0]).size),
        "dtype": str(dtype),
        "gpu_capability": list(capability),
        "timing_seconds": timings,
    }

    if not args.skip_reconstruction:
        from vggt.models.vggt import VGGT

        stage_started = time.perf_counter()
        model = VGGT.from_pretrained(str(args.vggt_model), enable_gaussian=True)
        if hasattr(model.gaussian_head, "enable_color_head_after_init"):
            model.gaussian_head.enable_color_head_after_init()
        model.eval()
        model.point_head = None
        model.track_head = None
        checkpoint_metadata = load_reconstruction_checkpoint(
            model, args.vggt_checkpoint
        )
        metadata["vggt_checkpoint"] = checkpoint_metadata
        model.to(device)
        torch.cuda.synchronize(device)
        timings["vggt_model_load"] = time.perf_counter() - stage_started

        real_metrics_path = real_dir / "metrics.json"
        real_pointcloud_path = real_dir / "point_cloud.ply"
        reuse_real = (
            args.reuse_real_baseline
            and real_metrics_path.is_file()
            and real_pointcloud_path.is_file()
        )
        if reuse_real:
            real_metrics = json.loads(real_metrics_path.read_text())
            real_reconstruction = real_metrics["reconstruction"]
            stage_started = time.perf_counter()
            real_points = read_pointcloud_xyz(real_pointcloud_path)
            timings["real_baseline_load"] = time.perf_counter() - stage_started
            timings["real_baseline_reconstruction"] = 0.0
        else:
            real_reconstruction, real_points = reconstruct_branch(
                model,
                real_image_paths,
                real_dir,
                device,
                dtype,
                args.vggt_max_side,
                args.opacity_threshold,
                args.max_points,
                args.seed,
            )
            timings["real_baseline_load"] = 0.0
            timings["real_baseline_reconstruction"] = real_reconstruction["elapsed_seconds"]
            real_metrics = {
                "branch": real_name,
                "description": f"VGGT-G reconstruction from {len(baseline_views)} real images.",
                "vggt_checkpoint": checkpoint_metadata,
                "reconstruction": real_reconstruction,
            }
            real_metrics_path.write_text(json.dumps(real_metrics, indent=2))
        metadata["real_baseline_reused"] = reuse_real

        generated_reconstruction, generated_points = reconstruct_branch(
            model,
            generated_input_paths,
            generated_dir,
            device,
            dtype,
            args.vggt_max_side,
            args.opacity_threshold,
            args.max_points,
            args.seed,
        )
        timings["mixed_vggt_reconstruction"] = generated_reconstruction["elapsed_seconds"]
        generated_metrics = {
            "branch": generated_name,
            "description": (
                f"VGGT-G reconstruction from {len(inputs)} real inputs and "
                f"{len(targets)} SEVA-generated views."
            ),
            "vggt_checkpoint": checkpoint_metadata,
            "image_generation": image_metrics,
            "reconstruction": generated_reconstruction,
        }
        (generated_dir / "metrics.json").write_text(
            json.dumps(generated_metrics, indent=2)
        )

        stage_started = time.perf_counter()
        comparison = compare_pointclouds(
            real_points,
            generated_points,
            args.comparison_points,
            args.seed,
            real_name,
            generated_name,
        )
        timings["geometry_comparison"] = time.perf_counter() - stage_started
        comparison.update(
            {
                "real_metrics": str(real_dir / "metrics.json"),
                "generated_metrics": str(generated_dir / "metrics.json"),
                "note": (
                    f"The {real_name} reconstruction is the reference, not a ground-truth point cloud. "
                    "Use this metric for paired ablations, not absolute geometry accuracy."
                ),
            }
        )
        comparison_path = output_dir / (
            f"comparison_{real_name}_vs_{generated_name}.json"
        )
        comparison_path.write_text(json.dumps(comparison, indent=2))
        metadata["comparison_metrics"] = str(comparison_path)
        metadata["real_reconstruction"] = real_reconstruction
        metadata["generated_reconstruction"] = generated_reconstruction
        timings["warm_end_to_end"] = (
            timings["seva_generation"]
            + timings["generated_outputs_and_image_metrics"]
            + timings["mixed_vggt_reconstruction"]
        )
        timings["cold_end_to_end"] = (
            timings["seva_model_load"]
            + timings["vggt_model_load"]
            + timings["warm_end_to_end"]
        )
        del model, real_points, generated_points
        torch.cuda.empty_cache()

    metadata["elapsed_seconds"] = time.perf_counter() - started
    timings["total_wall"] = metadata["elapsed_seconds"]
    metadata["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    run_metadata_path = output_dir / (
        f"run_metadata_{generated_name}.json"
    )
    run_metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
