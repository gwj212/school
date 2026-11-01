#!/usr/bin/env python3
"""Evaluate selective target-view depth-consistency opacity post-processing on test30 real8."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


MODEL_CANVAS = 518
PATCH = 14
ALPHA_REFERENCE_THRESHOLD = 0.75
FRONT_RELATIVE_THRESHOLD = 0.03
CONFLICT_OPACITY_WEIGHT = 0.25
MANIFEST = Path("/root/seva_vggt/config/test30_mvgen/selection_protocol.json")
ROOT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
RECON_ROOT = ROOT / "reconstruction/scenes"
BASELINE_ROOT = ROOT / "evaluation_native_masked_real8"
OUTPUT = ROOT / "evaluation_zconsistency_real8"


def preprocess_geometry(width: int, height: int) -> dict:
    if width >= height:
        rw, rh = MODEL_CANVAS, round(height * MODEL_CANVAS / width / PATCH) * PATCH
    else:
        rh, rw = MODEL_CANVAS, round(width * MODEL_CANVAS / height / PATCH) * PATCH
    pw, ph = MODEL_CANVAS - rw, MODEL_CANVAS - rh
    return {"w": width, "h": height, "rw": int(rw), "rh": int(rh),
            "pl": int(pw // 2), "pr": int(pw - pw // 2),
            "pt": int(ph // 2), "pb": int(ph - ph // 2)}


def native_intrinsic(k: torch.Tensor, g: dict) -> torch.Tensor:
    out = k.clone()
    sx, sy = g["w"] / g["rw"], g["h"] / g["rh"]
    out[0, 0], out[1, 1] = k[0, 0] * sx, k[1, 1] * sy
    out[0, 2], out[1, 2] = (k[0, 2] - g["pl"]) * sx, (k[1, 2] - g["pt"]) * sy
    return out


def pil_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def to_pil_chw(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def calculate_metrics(gt: torch.Tensor, pred: torch.Tensor, perceptual) -> dict:
    gt, pred = gt.float().clamp(0, 1), pred.float().clamp(0, 1)
    mse = torch.mean((gt - pred) ** 2).item()
    gt_np = gt.permute(1, 2, 0).cpu().numpy()
    pred_np = pred.permute(1, 2, 0).cpu().numpy()
    ssim = structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0)
    with torch.inference_mode():
        distance = perceptual(gt[None] * 2 - 1, pred[None] * 2 - 1)
    return {"mse": mse, "psnr": -10 * math.log10(max(mse, 1e-12)),
            "ssim": float(ssim), "lpips": float(distance.flatten()[0])}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def fitted(image: Image.Image, width: int, height: int) -> Image.Image:
    result = Image.new("RGB", (width, height), "white")
    copy = image.convert("RGB")
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    result.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return result


def make_sheet(scene: str, visual: list[dict], path: Path) -> None:
    cell_w, cell_h, label_w, header = 250, 210, 92, 50
    titles = ("GT", "unpad baseline", "z-consistency", "alpha")
    sheet = Image.new("RGB", (label_w + 4 * cell_w, header + 8 * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), scene, fill="black")
    for col, title in enumerate(titles):
        draw.text((label_w + col * cell_w + 5, 25), title, fill="black")
    for row in visual:
        y = header + row["slot"] * cell_h
        draw.text((5, y + 8), f"view {row['slot']:02d}", fill="black")
        draw.text((5, y + 27), f"dP {row['psnr_gain']:+.2f}", fill="black")
        draw.text((5, y + 46), f"dS {row['ssim_gain']:+.3f}", fill="black")
        for col, key in enumerate(("gt", "baseline", "processed", "alpha")):
            sheet.paste(fitted(row[key], cell_w, cell_h - 25), (label_w + col * cell_w, y))
    sheet.save(path, quality=94)


def render_source_depth(rasterization, flat, target, view, K, g):
    out, alpha, _ = rasterization(
        means=flat["xyz"][target], quats=flat["rotation"][target],
        scales=flat["scale"][target], opacities=flat["opacity"][target, 0],
        colors=flat["color"][target], viewmats=view[None], Ks=K[None],
        width=g["w"], height=g["h"], near_plane=0.01, far_plane=1e10,
        render_mode="RGB+ED",
    )
    return out[0, ..., 3], alpha[0, ..., 0]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "scenes").mkdir(exist_ok=True)
    (OUTPUT / "comparisons").mkdir(exist_ok=True)
    sys.path.insert(0, "/root/vggt")
    from gsplat import rasterization
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    protocol = {
        "name": "target_view_front_conflict_soft_opacity",
        "input": "real8 padding-masked Gaussians",
        "reference": "target source-view expected depth rendered at native resolution",
        "reference_alpha_threshold": ALPHA_REFERENCE_THRESHOLD,
        "front_relative_threshold": FRONT_RELATIVE_THRESHOLD,
        "conflict_rule": "non-target Gaussian z < 0.97 * target-source expected z",
        "conflict_opacity_weight": CONFLICT_OPACITY_WEIGHT,
        "target_source_weight": 1.0,
        "notes": "No GT is used by filtering. GT is used only for evaluation.",
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = json.loads(MANIFEST.read_text())["scenes"]
    baseline = pd.read_csv(BASELINE_ROOT / "metrics_per_view.csv", encoding="utf-8-sig")
    baseline_map = {(r.scene, int(r.slot)): r for r in baseline.itertuples(index=False)}
    device = torch.device("cuda:0")
    perceptual = lpips.LPIPS(net="alex").eval().to(device)
    status_path = OUTPUT / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"completed": [], "failed": {}}

    for scene in scenes:
        scene_dir = OUTPUT / "scenes" / scene
        scene_metrics_path = scene_dir / "metrics.json"
        if scene in status.get("completed", []) and scene_metrics_path.exists():
            print(f"SKIP {scene}", flush=True)
            continue
        started = time.perf_counter()
        try:
            scene_dir.mkdir(parents=True, exist_ok=True)
            branch = RECON_ROOT / scene / "real8"
            rec = torch.load(branch / "vggt_reconstruction.pt", map_location="cpu")
            paths = json.loads((branch / "metrics.json").read_text())["input_paths"]
            images = [Image.open(path).convert("RGB") for path in paths]
            geoms = [preprocess_geometry(*image.size) for image in images]
            valid = torch.zeros((8, MODEL_CANVAS, MODEL_CANVAS), dtype=torch.bool, device=device)
            for source, g in enumerate(geoms):
                valid[source, g["pt"]:MODEL_CANVAS-g["pb"], g["pl"]:MODEL_CANVAS-g["pr"]] = True
            grid = {
                key: value[0].float().reshape(8, MODEL_CANVAS, MODEL_CANVAS, value.shape[-1]).to(device)
                for key, value in rec["gaussians"].items() if torch.is_tensor(value) and value.ndim >= 2
            }
            flat = {key: value[valid] for key, value in grid.items()}
            source_grid = torch.arange(8, device=device)[:, None, None].expand_as(valid)
            source_ids = source_grid[valid]
            ext, intr = pose_encoding_to_extri_intri(rec["pose_enc"].float(), (MODEL_CANVAS, MODEL_CANVAS))
            ext, intr = ext[0].to(device), intr[0].to(device)
            scene_rows, visual = [], []

            for slot in range(8):
                view_started = time.perf_counter()
                g = geoms[slot]
                K = native_intrinsic(intr[slot], g)
                view = torch.eye(4, device=device)
                view[:3, :3], view[:3, 3] = ext[slot, :3, :3], ext[slot, :3, 3]
                target = source_ids == slot
                with torch.inference_mode():
                    ref_depth, ref_alpha = render_source_depth(rasterization, flat, target, view, K, g)
                    cam = flat["xyz"] @ view[:3, :3].T + view[:3, 3]
                    z = cam[:, 2]
                    u = K[0, 0] * cam[:, 0] / z.clamp_min(1e-8) + K[0, 2]
                    v = K[1, 1] * cam[:, 1] / z.clamp_min(1e-8) + K[1, 2]
                    inside = (z > 0.01) & (u >= 0) & (u < g["w"]) & (v >= 0) & (v < g["h"])
                    px = u.round().long().clamp(0, g["w"] - 1)
                    py = v.round().long().clamp(0, g["h"] - 1)
                    sampled_depth = ref_depth[py, px]
                    sampled_alpha = ref_alpha[py, px]
                    rel = (z - sampled_depth) / sampled_depth.clamp_min(1e-6)
                    reference = inside & (~target) & (sampled_alpha >= ALPHA_REFERENCE_THRESHOLD) & torch.isfinite(sampled_depth)
                    front_conflict = reference & (rel < -FRONT_RELATIVE_THRESHOLD)
                    weights = torch.ones_like(flat["opacity"][:, 0])
                    weights[front_conflict] = CONFLICT_OPACITY_WEIGHT
                    out, alpha, _ = rasterization(
                        means=flat["xyz"], quats=flat["rotation"], scales=flat["scale"],
                        opacities=flat["opacity"][:, 0] * weights, colors=flat["color"],
                        viewmats=view[None], Ks=K[None], width=g["w"], height=g["h"],
                        near_plane=0.01, far_plane=1e10, render_mode="RGB",
                    )
                pred = out[0, ..., :3].permute(2, 0, 1).clamp(0, 1)
                gt = pil_tensor(images[slot], device)
                result = calculate_metrics(gt, pred, perceptual)
                render_pil = to_pil_chw(pred)
                alpha_pil = Image.fromarray((alpha[0, ..., 0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
                render_pil.save(scene_dir / f"view_{slot:02d}_render.png")
                alpha_pil.save(scene_dir / f"view_{slot:02d}_alpha.png")
                base = baseline_map[(scene, slot)]
                row = {
                    "scene": scene, "category": scene.split("_")[0], "slot": slot,
                    "width": g["w"], "height": g["h"],
                    "gaussian_count": int(len(source_ids)),
                    "target_source_gaussians": int(target.sum()),
                    "front_conflict_gaussians": int(front_conflict.sum()),
                    "front_conflict_fraction_all": float(front_conflict.float().mean()),
                    "front_conflict_fraction_non_target": float(front_conflict.sum() / (~target).sum()),
                    "reference_alpha_mean": float(ref_alpha.mean()),
                    "processed_alpha_mean": float(alpha.mean()),
                    "baseline_psnr": float(base.masked_psnr),
                    "baseline_ssim": float(base.masked_ssim),
                    "baseline_lpips": float(base.masked_lpips),
                    "baseline_mse": float(base.masked_mse),
                    **{f"processed_{k}": v for k, v in result.items()},
                    "elapsed_seconds": time.perf_counter() - view_started,
                }
                row["psnr_gain"] = row["processed_psnr"] - row["baseline_psnr"]
                row["ssim_gain"] = row["processed_ssim"] - row["baseline_ssim"]
                row["lpips_reduction"] = row["baseline_lpips"] - row["processed_lpips"]
                row["mse_reduction"] = row["baseline_mse"] - row["processed_mse"]
                scene_rows.append(row)
                baseline_image = Image.open(BASELINE_ROOT / "scenes" / scene / f"view_{slot:02d}_render_masked.png").convert("RGB")
                visual.append({"slot": slot, "gt": images[slot], "baseline": baseline_image,
                               "processed": render_pil, "alpha": alpha_pil,
                               "psnr_gain": row["psnr_gain"], "ssim_gain": row["ssim_gain"]})
                print(scene, slot, f"PSNR={row['processed_psnr']:.3f}", f"gain={row['psnr_gain']:+.3f}",
                      f"conflicts={row['front_conflict_gaussians']}", flush=True)
                del out, alpha, pred, gt, cam, z, u, v, sampled_depth, sampled_alpha, rel, weights
                torch.cuda.empty_cache()

            (scene_dir / "metrics.json").write_text(json.dumps(scene_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            make_sheet(scene, visual, OUTPUT / "comparisons" / f"{scene}_comparison.jpg")
            if scene not in status["completed"]:
                status["completed"].append(scene)
            status["failed"].pop(scene, None)
            status["last_scene_seconds"] = time.perf_counter() - started
            print(f"COMPLETE {scene} seconds={status['last_scene_seconds']:.1f}", flush=True)
            del rec, grid, flat
            torch.cuda.empty_cache()
        except Exception as error:
            import traceback
            status["failed"][scene] = {"error": repr(error), "traceback": traceback.format_exc()}
            print(f"FAILED {scene}: {error!r}", flush=True)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for scene in scenes:
        path = OUTPUT / "scenes" / scene / "metrics.json"
        if path.exists(): rows.extend(json.loads(path.read_text()))
    write_csv(OUTPUT / "metrics_per_view.csv", rows)
    scene_rows = []
    for scene in scenes:
        subset = [row for row in rows if row["scene"] == scene]
        if not subset: continue
        item = {"scene": scene, "category": scene.split("_")[0], "view_count": len(subset)}
        for key in ("baseline_psnr", "baseline_ssim", "baseline_lpips", "baseline_mse",
                    "processed_psnr", "processed_ssim", "processed_lpips", "processed_mse",
                    "psnr_gain", "ssim_gain", "lpips_reduction", "mse_reduction",
                    "front_conflict_fraction_non_target", "processed_alpha_mean", "elapsed_seconds"):
            item[f"{key}_mean"] = float(np.mean([row[key] for row in subset]))
        scene_rows.append(item)
    write_csv(OUTPUT / "metrics_per_scene.csv", scene_rows)

    summary = {"protocol": protocol, "scene_count": len(scene_rows), "view_count": len(rows),
               "failed": status["failed"], "baseline": {}, "processed": {}, "improvement": {}}
    for metric in ("psnr", "ssim", "lpips", "mse"):
        for prefix in ("baseline", "processed"):
            values = np.asarray([row[f"{prefix}_{metric}"] for row in rows], float)
            summary[prefix][metric] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                       "std": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max())}
    for key in ("psnr_gain", "ssim_gain", "lpips_reduction", "mse_reduction"):
        values = np.asarray([row[key] for row in rows], float)
        summary["improvement"][key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                        "improved_views": int((values > 0).sum()), "total_views": len(values)}
    conflicts = np.asarray([row["front_conflict_fraction_non_target"] for row in rows], float)
    summary["front_conflict_fraction_non_target"] = {"mean": float(conflicts.mean()),
                                                       "median": float(np.median(conflicts)),
                                                       "min": float(conflicts.min()), "max": float(conflicts.max())}
    summary["elapsed_seconds_sum"] = float(sum(row["elapsed_seconds"] for row in rows))
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 30场景目标视角深度一致性后处理报告

## 实验设置

- 数据：my_test_multiview测试集固定30场景，每场景8张真实图。
- 输入：VGGT-G输出，先硬移除来源图padding Gaussian。
- 后处理：目标来源视角渲染期望深度；仅对目标视角alpha≥{ALPHA_REFERENCE_THRESHOLD:.2f}的可靠区域，检测位于参考表面前方超过{FRONT_RELATIVE_THRESHOLD*100:.1f}%的其他来源Gaussian，并将其opacity乘{CONFLICT_OPACITY_WEIGHT:.2f}。
- 目标来源Gaussian不降权；不使用GT参与筛选，GT只用于计算指标。

## 汇总结果

| 指标 | 去padding基线 | 深度一致性后处理 | 变化 |
|---|---:|---:|---:|
| PSNR↑ | {summary['baseline']['psnr']['mean']:.4f} | {summary['processed']['psnr']['mean']:.4f} | {summary['improvement']['psnr_gain']['mean']:+.4f} |
| SSIM↑ | {summary['baseline']['ssim']['mean']:.4f} | {summary['processed']['ssim']['mean']:.4f} | {summary['improvement']['ssim_gain']['mean']:+.4f} |
| LPIPS↓ | {summary['baseline']['lpips']['mean']:.4f} | {summary['processed']['lpips']['mean']:.4f} | {-summary['improvement']['lpips_reduction']['mean']:+.4f} |
| MSE↓ | {summary['baseline']['mse']['mean']:.6f} | {summary['processed']['mse']['mean']:.6f} | {-summary['improvement']['mse_reduction']['mean']:+.6f} |

- PSNR改善：{summary['improvement']['psnr_gain']['improved_views']}/{len(rows)}视角。
- SSIM改善：{summary['improvement']['ssim_gain']['improved_views']}/{len(rows)}视角。
- LPIPS改善：{summary['improvement']['lpips_reduction']['improved_views']}/{len(rows)}视角。
- 非目标来源Gaussian中，被判定为前景深度冲突的比例均值：{summary['front_conflict_fraction_non_target']['mean']*100:.2f}%。

## 解释

该方法不删除原始重建文件中的Gaussian，只在目标视角渲染时选择性降低深度冲突Gaussian的opacity。因此它是可逆的视角相关后处理，避免硬删除造成新孔洞。逐场景、逐视角数值与比较图应结合使用：平均指标改善只能说明总体趋势，薄结构和遮挡边界仍需检查局部视觉结果。
"""
    (OUTPUT / "深度一致性后处理报告.md").write_text(report, encoding="utf-8")
    status["summary"] = summary
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if status["failed"] or len(rows) != 240:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
