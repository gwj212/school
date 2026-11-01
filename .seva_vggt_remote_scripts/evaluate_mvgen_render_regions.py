#!/usr/bin/env python3
"""Region-aware evaluation of saved MVGenMaster->VGGT-G Gaussian reconstructions."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.metrics import structural_similarity


ROOT = Path("/root/seva_vggt/outputs/model_strategy_test5")
MANIFEST_ROOT = Path("/root/seva_vggt/config/strategy_test5")
OUT = ROOT / "vggt_render_region_eval"
SCENES = [
    "chair_386_46082_92005",
    "car_599_92296_183295",
    "couch_349_36574_67726",
    "toybus_517_74123_144462",
    "ball_113_13373_24790",
]
ALPHA_THRESHOLD = 0.05
DEPTH_CONFLICT_REL = 0.20


def preprocess_mask(path: str, target: int = 518) -> np.ndarray:
    image = Image.open(path).convert("L")
    width, height = image.size
    if width >= height:
        new_width = target
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target
        new_width = round(width * (new_height / height) / 14) * 14
    image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
    array = np.asarray(image) > 127
    top = (target - new_height) // 2
    bottom = target - new_height - top
    left = (target - new_width) // 2
    right = target - new_width - left
    return np.pad(array, ((top, bottom), (left, right)), constant_values=False)


def viewmat_and_k(extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    view = torch.eye(4, device=extrinsic.device, dtype=extrinsic.dtype)
    view[:3, :3] = extrinsic[:3, :3]
    view[:3, 3] = extrinsic[:3, 3]
    k = torch.zeros((3, 3), device=intrinsic.device, dtype=intrinsic.dtype)
    k[0, 0], k[1, 1] = intrinsic[0, 0], intrinsic[1, 1]
    k[0, 2], k[1, 2], k[2, 2] = intrinsic[0, 2], intrinsic[1, 2], 1
    return view[None], k[None]


def rasterize_rgb_depth_alpha(g: dict[str, torch.Tensor], extrinsic: torch.Tensor,
                              intrinsic: torch.Tensor, height: int, width: int):
    from gsplat import rasterization
    view, k = viewmat_and_k(extrinsic, intrinsic)
    rendered, alpha, _ = rasterization(
        means=g["xyz"], quats=g["rotation"], scales=g["scale"],
        opacities=g["opacity"].squeeze(-1), colors=g["color"],
        viewmats=view, Ks=k, width=width, height=height,
        near_plane=0.01, far_plane=1e10, sh_degree=None,
        render_mode="RGB+ED",
    )
    return (
        rendered[0, ..., :3].permute(2, 0, 1).clamp(0, 1),
        rendered[0, ..., 3],
        alpha[0, ..., 0].clamp(0, 1),
    )


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def masked_rgb_metrics(reference: torch.Tensor, prediction: torch.Tensor, mask_np: np.ndarray,
                       lpips_model: torch.nn.Module) -> dict[str, float | None]:
    mask = torch.from_numpy(mask_np).to(reference.device)
    count = int(mask.sum().item())
    if count < 64:
        return {"pixel_count": count, "psnr": None, "ssim": None, "lpips": None}
    ref = reference.float().clamp(0, 1)
    pred = prediction.float().clamp(0, 1)
    diff = (ref - pred)[:, mask]
    mse = torch.mean(diff * diff).item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    ref_np = ref.permute(1, 2, 0).detach().cpu().numpy()
    pred_np = pred.permute(1, 2, 0).detach().cpu().numpy()
    _, ssim_map = structural_similarity(ref_np, pred_np, channel_axis=2, data_range=1.0, full=True)
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=2)
    ssim = float(ssim_map[mask_np].mean())
    ys, xs = np.where(mask_np)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop_mask = mask[y0:y1, x0:x1].float()[None]
    ref_crop = ref[:, y0:y1, x0:x1] * crop_mask + 0.5 * (1 - crop_mask)
    pred_crop = pred[:, y0:y1, x0:x1] * crop_mask + 0.5 * (1 - crop_mask)
    with torch.inference_mode():
        distance = lpips_model(ref_crop[None] * 2 - 1, pred_crop[None] * 2 - 1)
    return {
        "pixel_count": count,
        "psnr": finite(psnr),
        "ssim": finite(ssim),
        "lpips": finite(float(distance.flatten()[0].detach().cpu())),
    }


def bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def silhouette_metrics(gt: np.ndarray, rendered: np.ndarray) -> dict[str, float | int | None]:
    intersection = gt & rendered
    union = gt | rendered
    gt_n, rd_n, inter_n = int(gt.sum()), int(rendered.sum()), int(intersection.sum())
    if gt_n == 0:
        return {
            "gt_area_pixels": 0, "render_area_pixels": rd_n, "intersection_pixels": 0,
            "iou": None, "dice": None, "precision": None, "recall": None,
            "render_to_gt_area_ratio": None, "centroid_shift_normalized": None,
            "bbox_width_ratio": None, "bbox_height_ratio": None,
            "boundary_chamfer_normalized": None,
        }
    precision = inter_n / rd_n if rd_n else None
    recall = inter_n / gt_n if gt_n else None
    dice = 2 * inter_n / (gt_n + rd_n) if gt_n + rd_n else None
    gt_box, rd_box = bbox(gt), bbox(rendered)
    diagonal = math.hypot(*gt.shape)
    centroid_shift = None
    width_ratio = height_ratio = None
    if gt_n and rd_n:
        gy, gx = np.argwhere(gt).mean(axis=0)
        ry, rx = np.argwhere(rendered).mean(axis=0)
        centroid_shift = math.hypot(float(gx - rx), float(gy - ry)) / diagonal
    if gt_box and rd_box:
        width_ratio = (rd_box[2] - rd_box[0]) / max(gt_box[2] - gt_box[0], 1)
        height_ratio = (rd_box[3] - rd_box[1]) / max(gt_box[3] - gt_box[1], 1)
    gt_edge = gt ^ binary_erosion(gt)
    rd_edge = rendered ^ binary_erosion(rendered)
    boundary_chamfer = None
    if gt_edge.any() and rd_edge.any():
        d_to_rd = distance_transform_edt(~rd_edge)[gt_edge].mean()
        d_to_gt = distance_transform_edt(~gt_edge)[rd_edge].mean()
        boundary_chamfer = float((d_to_rd + d_to_gt) * 0.5 / diagonal)
    return {
        "gt_area_pixels": gt_n,
        "render_area_pixels": rd_n,
        "intersection_pixels": inter_n,
        "iou": inter_n / int(union.sum()) if union.any() else None,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "render_to_gt_area_ratio": rd_n / gt_n if gt_n else None,
        "centroid_shift_normalized": centroid_shift,
        "bbox_width_ratio": width_ratio,
        "bbox_height_ratio": height_ratio,
        "boundary_chamfer_normalized": boundary_chamfer,
    }


def completion_metrics(gt: np.ndarray, render_mask: np.ndarray, alpha: np.ndarray,
                       object_depth: np.ndarray, input_depth: np.ndarray,
                       depth_conf: np.ndarray) -> tuple[dict[str, float | int | None], np.ndarray]:
    if not gt.any():
        empty = np.zeros_like(gt, dtype=bool)
        return {
            "area_pixels": None, "area_image_ratio": None, "area_render_ratio": None,
            "alpha_mean": None, "depth_confidence_mean": None, "depth_valid_pixels": None,
            "depth_absrel_mean": None, "depth_conflict_rate_gt20pct": None,
            "front_conflict_rate_gt20pct": None, "behind_conflict_rate_gt20pct": None,
        }, empty
    completion = render_mask & ~gt
    n = int(completion.sum())
    result: dict[str, float | int | None] = {
        "area_pixels": n,
        "area_image_ratio": n / completion.size,
        "area_render_ratio": n / max(int(render_mask.sum()), 1),
        "alpha_mean": float(alpha[completion].mean()) if n else None,
        "depth_confidence_mean": float(depth_conf[completion].mean()) if n else None,
    }
    valid = completion & np.isfinite(object_depth) & np.isfinite(input_depth) & (object_depth > 0) & (input_depth > 0)
    if valid.any():
        rel = np.abs(object_depth[valid] - input_depth[valid]) / np.maximum(input_depth[valid], 1e-6)
        result.update({
            "depth_valid_pixels": int(valid.sum()),
            "depth_absrel_mean": float(rel.mean()),
            "depth_conflict_rate_gt20pct": float((rel > DEPTH_CONFLICT_REL).mean()),
            "front_conflict_rate_gt20pct": float((object_depth[valid] < input_depth[valid] * (1 - DEPTH_CONFLICT_REL)).mean()),
            "behind_conflict_rate_gt20pct": float((object_depth[valid] > input_depth[valid] * (1 + DEPTH_CONFLICT_REL)).mean()),
        })
    else:
        result.update({
            "depth_valid_pixels": 0,
            "depth_absrel_mean": None,
            "depth_conflict_rate_gt20pct": None,
            "front_conflict_rate_gt20pct": None,
            "behind_conflict_rate_gt20pct": None,
        })
    return result, completion


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    identity = {"scene", "slot", "view_index", "source_type", "gt_mask_valid"}
    output = []
    for scope, selected in (
        ("all_views", rows),
        ("generated_views_only", [r for r in rows if r["source_type"] == "generated"]),
    ):
        for key in sorted(set().union(*(r.keys() for r in selected)) - identity):
            values = [float(r[key]) for r in selected if r.get(key) is not None and math.isfinite(float(r[key]))]
            if not values:
                continue
            a = np.asarray(values)
            output.append({
                "scope": scope, "metric": key, "count": len(a),
                "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
                "sum": float(a.sum()), "median": float(np.median(a)),
                "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            })
    return output


def to_u8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def make_sheet(scene: str, rows_visual: list[tuple], path: Path) -> None:
    thumb = 180
    label = 90
    header = 45
    columns = ("Real GT", "MVGenMaster", "VGGT render", "GT mask", "Object alpha", "Completion")
    sheet = Image.new("RGB", (label + len(columns) * thumb, header + len(rows_visual) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    for c, title in enumerate(columns):
        draw.text((label + c * thumb + 12, 15), title, fill="black")
    for r, (view_index, source_type, real, mixed, render, gt, alpha, completion) in enumerate(rows_visual):
        y = header + r * thumb
        draw.multiline_text((5, y + 8), f"view {view_index:02d}\n{source_type}", fill="black")
        gt_rgb = np.repeat((gt[..., None] * 255).astype(np.uint8), 3, axis=2)
        alpha_rgb = np.repeat((np.clip(alpha, 0, 1)[..., None] * 255).astype(np.uint8), 3, axis=2)
        comp_rgb = np.zeros((*completion.shape, 3), dtype=np.uint8)
        comp_rgb[..., 0] = completion.astype(np.uint8) * 255
        images = (to_u8(real), to_u8(mixed), to_u8(render), gt_rgb, alpha_rgb, comp_rgb)
        for c, array in enumerate(images):
            image = Image.fromarray(array).resize((thumb, thumb), Image.Resampling.NEAREST if c >= 3 else Image.Resampling.LANCZOS)
            sheet.paste(image, (label + c * thumb, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def main() -> None:
    sys.path.insert(0, "/root/vggt")
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    all_rows: list[dict] = []
    for scene in SCENES:
        generation = json.loads((ROOT / "mvgenmaster" / scene / "generation_metadata.json").read_text())
        view_indices = [int(generation["source_manifest_index"])] + [int(v) for v in generation["target_manifest_indices"]]
        manifest = json.loads((MANIFEST_ROOT / f"{scene}.json").read_text())
        gt_masks = np.stack([preprocess_mask(manifest["views"][i]["mask"]) for i in view_indices])
        reconstruction = torch.load(ROOT / "mvgenmaster_vggt" / scene / "vggt_reconstruction.pt", map_location="cpu")
        real_reconstruction = torch.load(ROOT / "seva" / scene / "real8" / "vggt_reconstruction.pt", map_location="cpu")
        mixed = reconstruction["images"][0].float().to(device)
        real = real_reconstruction["images"][0].float().to(device)
        height, width = mixed.shape[-2:]
        if gt_masks.shape[1:] != (height, width):
            raise RuntimeError(f"mask mismatch {gt_masks.shape} vs {(height, width)}")
        extrinsics, intrinsics = pose_encoding_to_extri_intri(reconstruction["pose_enc"].float(), (height, width))
        extrinsics, intrinsics = extrinsics[0].to(device), intrinsics[0].to(device)
        g_all = {
            key: value[0].reshape(-1, value.shape[-1]).float().to(device)
            for key, value in reconstruction["gaussians"].items()
            if torch.is_tensor(value) and value.ndim >= 2
        }
        selector = torch.from_numpy(gt_masks.reshape(-1)).to(device)
        if len(selector) != len(g_all["xyz"]):
            raise RuntimeError(f"gaussian/mask layout mismatch {len(selector)} vs {len(g_all['xyz'])}")
        g_object = {key: value[selector] for key, value in g_all.items()}
        depth_input = reconstruction["depth"][0, ..., 0].float().numpy()
        depth_conf = reconstruction["depth_conf"][0].float().numpy()
        scene_rows = []
        visuals = []
        for slot, view_index in enumerate(view_indices):
            with torch.inference_mode():
                render_rgb, _, _ = rasterize_rgb_depth_alpha(g_all, extrinsics[slot], intrinsics[slot], height, width)
                _, object_depth_t, object_alpha_t = rasterize_rgb_depth_alpha(g_object, extrinsics[slot], intrinsics[slot], height, width)
            object_alpha = object_alpha_t.cpu().numpy()
            object_depth = object_depth_t.cpu().numpy()
            render_mask = object_alpha >= ALPHA_THRESHOLD
            gt_mask = gt_masks[slot]
            visible = gt_mask & render_mask
            r_to_m = masked_rgb_metrics(mixed[slot], render_rgb, visible, lpips_model)
            r_to_g = masked_rgb_metrics(real[slot], render_rgb, visible, lpips_model)
            m_to_g = masked_rgb_metrics(real[slot], mixed[slot], visible, lpips_model)
            outline = silhouette_metrics(gt_mask, render_mask)
            completion, completion_mask = completion_metrics(
                gt_mask, render_mask, object_alpha, object_depth, depth_input[slot], depth_conf[slot]
            )
            row = {"scene": scene, "slot": slot, "view_index": view_index,
                   "source_type": "real" if slot == 0 else "generated",
                   "gt_mask_valid": int(gt_mask.sum() >= 64)}
            for prefix, values in (
                ("visible_render_to_mvgen", r_to_m),
                ("visible_render_to_real", r_to_g),
                ("visible_mvgen_to_real", m_to_g),
                ("silhouette", outline),
                ("completion", completion),
            ):
                row.update({f"{prefix}.{key}": value for key, value in values.items()})
            scene_rows.append(row)
            all_rows.append(row)
            visuals.append((view_index, row["source_type"], real[slot].cpu(), mixed[slot].cpu(),
                            render_rgb.cpu(), gt_mask, object_alpha, completion_mask))
            del render_rgb, object_depth_t, object_alpha_t
        scene_dir = OUT / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        write_csv(scene_dir / "per_view_metrics.csv", scene_rows)
        make_sheet(scene, visuals, scene_dir / f"{scene}_region_evaluation.jpg")
        (scene_dir / "metrics.json").write_text(json.dumps({
            "protocol": "mvgen_vggtg_region_evaluation_v1",
            "alpha_threshold": ALPHA_THRESHOLD,
            "depth_conflict_relative_threshold": DEPTH_CONFLICT_REL,
            "object_alpha_note": "Object-only Gaussians selected by paired GT masks for evaluation; not a deployment-time mask.",
            "views": scene_rows,
        }, indent=2) + "\n", encoding="utf-8")
        del reconstruction, real_reconstruction, mixed, real, g_all, g_object, selector
        torch.cuda.empty_cache()
        print(f"COMPLETE {scene}", flush=True)
    summary = aggregate(all_rows)
    write_csv(OUT / "region_metrics_per_view.csv", all_rows)
    write_csv(OUT / "region_metrics_summary.csv", summary)
    (OUT / "protocol.json").write_text(json.dumps({
        "protocol": "mvgen_vggtg_region_evaluation_v1",
        "scene_count": len(SCENES), "view_count": len(all_rows),
        "visible_region": "GT object mask intersect object-only render alpha mask; RGB metrics computed here only.",
        "silhouette_region": "GT object mask versus object-only render alpha mask.",
        "completion_region": "object-only render alpha mask minus GT visible object mask; no RGB PSNR/SSIM/LPIPS.",
        "completion_depth": "Object-render expected depth versus saved VGGT per-view depth in the same reconstruction.",
        "confidence": "Saved VGGT depth_conf plus rendered object alpha.",
        "alpha_threshold": ALPHA_THRESHOLD,
        "depth_conflict_relative_threshold": DEPTH_CONFLICT_REL,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"rows={len(all_rows)} summary_rows={len(summary)}", flush=True)


if __name__ == "__main__":
    main()
