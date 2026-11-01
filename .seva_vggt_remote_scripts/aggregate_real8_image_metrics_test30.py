#!/usr/bin/env python3
"""Aggregate native-resolution image metrics for the real8 VGGT-G branch."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "/root/seva_vggt/outputs/test30_mvgen_vggt/"
    "evaluation_native_v2/metrics_per_view_native.csv"
)
OUTPUT_DIR = Path(
    "/root/seva_vggt/outputs/test30_mvgen_vggt/"
    "evaluation_native_v2/real8_image_metrics"
)

METRICS = {
    "psnr": "real8_render_to_real_psnr",
    "ssim": "real8_render_to_real_ssim",
    "lpips": "real8_render_to_real_lpips",
    "mse": "real8_render_to_real_mse",
}


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(x) for x in column if str(x))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def grouped_stats(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    named = {name: (column, stat) for name, column in METRICS.items() for stat in []}
    del named
    result = data.groupby(keys, dropna=False).agg(
        view_count=("slot", "count"),
        psnr_mean=(METRICS["psnr"], "mean"),
        psnr_median=(METRICS["psnr"], "median"),
        psnr_std=(METRICS["psnr"], "std"),
        ssim_mean=(METRICS["ssim"], "mean"),
        ssim_median=(METRICS["ssim"], "median"),
        ssim_std=(METRICS["ssim"], "std"),
        lpips_mean=(METRICS["lpips"], "mean"),
        lpips_median=(METRICS["lpips"], "median"),
        lpips_std=(METRICS["lpips"], "std"),
        mse_mean=(METRICS["mse"], "mean"),
        mse_median=(METRICS["mse"], "median"),
        mse_std=(METRICS["mse"], "std"),
    )
    return result.reset_index()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required = {
        "scene",
        "category",
        "slot",
        "gt_path",
        "gt_width",
        "gt_height",
        "real8_render_size_matches_gt",
        *METRICS.values(),
    }
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise RuntimeError(f"Missing columns: {missing_columns}")
    if len(data) != 240:
        raise RuntimeError(f"Expected 240 rows (30 scenes x 8 views), found {len(data)}")
    if data["scene"].nunique() != 30:
        raise RuntimeError(f"Expected 30 scenes, found {data['scene'].nunique()}")
    if not bool((data["real8_render_size_matches_gt"] == 1).all()):
        raise RuntimeError("At least one real8 render size does not match its GT")
    for column in METRICS.values():
        if not np.isfinite(data[column].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite values found in {column}")

    per_view = data[
        [
            "scene",
            "category",
            "slot",
            "gt_path",
            "gt_width",
            "gt_height",
            "real8_render_size_matches_gt",
            METRICS["psnr"],
            METRICS["ssim"],
            METRICS["lpips"],
            METRICS["mse"],
        ]
    ].copy()
    per_view = per_view.rename(
        columns={
            METRICS["psnr"]: "psnr",
            METRICS["ssim"]: "ssim",
            METRICS["lpips"]: "lpips",
            METRICS["mse"]: "mse",
            "real8_render_size_matches_gt": "render_size_matches_gt",
        }
    )
    per_view.to_csv(OUTPUT_DIR / "real8_metrics_per_view.csv", index=False, encoding="utf-8-sig")

    per_scene = grouped_stats(data, ["scene", "category"])
    per_scene.to_csv(OUTPUT_DIR / "real8_metrics_per_scene.csv", index=False, encoding="utf-8-sig")

    by_category = grouped_stats(data, ["category"])
    by_category.insert(1, "scene_count", data.groupby("category")["scene"].nunique().reindex(by_category["category"]).to_numpy())
    by_category.to_csv(OUTPUT_DIR / "real8_metrics_by_category.csv", index=False, encoding="utf-8-sig")

    by_slot = grouped_stats(data, ["slot"])
    by_slot.to_csv(OUTPUT_DIR / "real8_metrics_by_slot.csv", index=False, encoding="utf-8-sig")

    scene_metric_columns = {
        "psnr": "psnr_mean",
        "ssim": "ssim_mean",
        "lpips": "lpips_mean",
        "mse": "mse_mean",
    }
    summary_rows = []
    summary_json = {
        "protocol": {
            "dataset": "/root/vggt/my_test_multiview",
            "scene_count": 30,
            "views_per_scene": 8,
            "view_count": 240,
            "comparison": "VGGT-G real8 reconstruction native-resolution render vs corresponding real GT image",
            "size_validation": "240/240 render sizes exactly match GT sizes",
            "psnr": "10*log10(1/MSE), RGB normalized to [0,1]",
            "ssim": "structural similarity on aligned RGB images",
            "lpips": "LPIPS AlexNet v0.1 on aligned RGB images",
            "aggregation": "view-level descriptive statistics plus 95% CI over 30 scene means",
        },
        "metrics": {},
    }
    for metric, source in METRICS.items():
        view_values = data[source].to_numpy(dtype=np.float64)
        scene_values = per_scene[scene_metric_columns[metric]].to_numpy(dtype=np.float64)
        scene_std = float(np.std(scene_values, ddof=1))
        ci_half = 1.96 * scene_std / math.sqrt(len(scene_values))
        record = {
            "metric": metric,
            "view_count": int(len(view_values)),
            "view_mean": float(np.mean(view_values)),
            "view_median": float(np.median(view_values)),
            "view_std": float(np.std(view_values, ddof=1)),
            "view_min": float(np.min(view_values)),
            "view_max": float(np.max(view_values)),
            "scene_count": int(len(scene_values)),
            "scene_mean": float(np.mean(scene_values)),
            "scene_std": scene_std,
            "scene_ci95_low": float(np.mean(scene_values) - ci_half),
            "scene_ci95_high": float(np.mean(scene_values) + ci_half),
        }
        summary_rows.append(record)
        summary_json["metrics"][metric] = record

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "real8_metric_summary.csv", index=False, encoding="utf-8-sig"
    )
    with (OUTPUT_DIR / "real8_metric_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, ensure_ascii=False, indent=2)

    metric_map = {row["metric"]: row for row in summary_rows}
    best_psnr = per_scene.loc[per_scene["psnr_mean"].idxmax()]
    worst_psnr = per_scene.loc[per_scene["psnr_mean"].idxmin()]
    report = f"""# 30场景八张真图重建的图像指标

## 评测对象

- 数据集：`/root/vggt/my_test_multiview` 中当前测试清单的30个场景。
- 每场景8张真实图，共240个输入视角。
- 对比对象：八张真实图输入 VGGT-G 后得到的三维高斯，在各输入相机视角按 GT 原始宽高直接渲染，再与对应真实图逐像素比较。
- 尺寸检查：240/240张渲染图与对应 GT 的宽、高完全一致。
- 注意：这不是“真实图与自身”的指标；后者 PSNR 为无穷大，没有评测意义。

## 总体结果

| 指标 | 240视角均值 | 中位数 | 标准差 | 30场景均值的95%CI |
|---|---:|---:|---:|---:|
| PSNR↑ | {metric_map['psnr']['view_mean']:.4f} dB | {metric_map['psnr']['view_median']:.4f} | {metric_map['psnr']['view_std']:.4f} | [{metric_map['psnr']['scene_ci95_low']:.4f}, {metric_map['psnr']['scene_ci95_high']:.4f}] |
| SSIM↑ | {metric_map['ssim']['view_mean']:.4f} | {metric_map['ssim']['view_median']:.4f} | {metric_map['ssim']['view_std']:.4f} | [{metric_map['ssim']['scene_ci95_low']:.4f}, {metric_map['ssim']['scene_ci95_high']:.4f}] |
| LPIPS↓ | {metric_map['lpips']['view_mean']:.4f} | {metric_map['lpips']['view_median']:.4f} | {metric_map['lpips']['view_std']:.4f} | [{metric_map['lpips']['scene_ci95_low']:.4f}, {metric_map['lpips']['scene_ci95_high']:.4f}] |
| MSE↓ | {metric_map['mse']['view_mean']:.6f} | {metric_map['mse']['view_median']:.6f} | {metric_map['mse']['view_std']:.6f} | [{metric_map['mse']['scene_ci95_low']:.6f}, {metric_map['mse']['scene_ci95_high']:.6f}] |

## 场景范围

- 场景平均PSNR最高：`{best_psnr['scene']}`，{best_psnr['psnr_mean']:.4f} dB。
- 场景平均PSNR最低：`{worst_psnr['scene']}`，{worst_psnr['psnr_mean']:.4f} dB。
- 逐场景、类别和槽位结果分别保存为CSV，便于定位类别难度与特定相机视角偏差。

## 解释

八张真实图重建仍不会与输入图完全一致，因为VGGT-G需要用一套共享三维高斯同时解释八个视角；遮挡边界、薄结构、反光区域、背景以及高斯筛选都会造成损失。这组结果应作为“当前模型在完整真实多视图输入下的重建上限基线”，用于比较1真+7生成、2真+6生成等方案。
"""
    (OUTPUT_DIR / "八真图像指标报告.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
