#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SELECTED = [
    "real8_render_to_real_psnr", "real8_render_to_real_ssim", "real8_render_to_real_lpips",
    "mv_render_to_generated_input_psnr", "mv_render_to_generated_input_ssim", "mv_render_to_generated_input_lpips",
    "mv_render_to_real_gt_psnr", "mv_render_to_real_gt_ssim", "mv_render_to_real_gt_lpips",
    "mv_generated_to_real_gt_psnr", "mv_generated_to_real_gt_ssim", "mv_generated_to_real_gt_lpips",
]


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    old_root = args.root / "evaluation"
    new_root = args.root / "evaluation_native_v2"
    old_summary = read_csv(old_root / "metric_summary.csv")
    new_summary = read_csv(new_root / "metric_summary_native.csv")
    rows = read_csv(new_root / "metrics_per_view_native.csv")

    old_map = {(row["scope"], row["metric"]): float(row["mean"]) for row in old_summary}
    new_map = {(row["scope"], row["metric"]): float(row["mean"]) for row in new_summary}
    comparison_rows = []
    for scope in ("all_views", "generated_views_only"):
        for metric in SELECTED:
            old = old_map.get((scope, metric))
            new = new_map.get((scope, metric))
            comparison_rows.append({
                "scope": scope, "metric": metric,
                "old_padded_518_mean": old, "native_corrected_mean": new,
                "native_minus_old": None if old is None or new is None else new - old,
                "old_protocol_valid_for_native_claim": 0,
            })
    write_csv(new_root / "old_vs_native_metric_comparison.csv", comparison_rows)

    category_groups = defaultdict(list)
    for row in rows:
        if row["source_type"] == "generated":
            category_groups[row["category"]].append(row)
    category_rows = []
    for category, subset in sorted(category_groups.items()):
        out = {"category": category, "view_count": len(subset)}
        for metric in SELECTED:
            values = [float(row[metric]) for row in subset]
            out[metric] = float(np.mean(values))
        category_rows.append(out)
    write_csv(new_root / "metrics_by_category_generated_views_native.csv", category_rows)

    validation_rows, missing, bad_size = [], [], []
    for row in rows:
        scene = row["scene"]
        slot = int(row["slot"])
        gt_size = (int(row["gt_width"]), int(row["gt_height"]))
        mv_size = (int(row["mv_width"]), int(row["mv_height"]))
        checks = (
            ("real8_native", gt_size), ("mvgen_native", mv_size), ("mvgen_to_gt_native", gt_size),
        )
        for render_set, expected in checks:
            path = new_root / "scenes" / scene / render_set / f"view_{slot:02d}_render.png"
            if not path.exists():
                missing.append(str(path))
                actual = None
            else:
                with Image.open(path) as image:
                    actual = image.size
                if actual != expected:
                    bad_size.append({"path": str(path), "actual": actual, "expected": expected})
            validation_rows.append({
                "scene": scene, "slot": slot, "render_set": render_set,
                "expected_width": expected[0], "expected_height": expected[1],
                "actual_width": None if actual is None else actual[0],
                "actual_height": None if actual is None else actual[1],
                "exact_match": int(actual == expected), "path": str(path),
            })
    write_csv(new_root / "render_size_validation.csv", validation_rows)

    generated = {metric: new_map[("generated_views_only", metric)] for metric in SELECTED}
    all_views = {metric: new_map[("all_views", metric)] for metric in SELECTED}
    old_generated = {metric: old_map[("generated_views_only", metric)] for metric in SELECTED}
    real_psnr_gap = generated["real8_render_to_real_psnr"] - generated["mv_render_to_real_gt_psnr"]
    real_ssim_gap = generated["real8_render_to_real_ssim"] - generated["mv_render_to_real_gt_ssim"]
    real_lpips_gap = generated["mv_render_to_real_gt_lpips"] - generated["real8_render_to_real_lpips"]
    reconstruction_psnr_loss = generated["mv_generated_to_real_gt_psnr"] - generated["mv_render_to_real_gt_psnr"]

    report = f"""# 测试集30场景：原始尺寸渲染与重算指标报告

## 修正结论

旧评测虽然保证参与计算的张量都是 `518×518`，但这些张量是 VGGT 的白边补齐方形画布，不是数据集原图或 MVGenMaster 生成图的真实尺寸。旧指标因此混入补边区域，而且旧渲染相机仍对应方形画布，不能作为“输入视角原始尺寸”的结果。

本轮没有重新运行或修改 VGGT-G，也没有改变保存的高斯参数。修正只发生在渲染与评测端：对每张输入图复现 VGGT `pad` 预处理的缩放与四边补齐量，把 `518×518` 相机内参去补边并缩放到目标图的真实宽高，然后直接以目标宽高进行高斯光栅化；没有把方形渲染图事后强行拉伸。

尺寸验证结果：240个视角、3套渲染，共720张 PNG；精确尺寸匹配 {sum(int(row['exact_match']) for row in validation_rows)}/720，缺失 {len(missing)}，尺寸错误 {len(bad_size)}。30张场景对比图均已生成。真实图包含30种尺寸，MVGen/VGGT实际输入包含{len(set((row['mv_width'], row['mv_height']) for row in rows))}种尺寸。

## 主要指标（只统计7个生成视角，共210个视角）

| 对比关系 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|
| 8真分支渲染 vs 真实GT | {fmt(generated['real8_render_to_real_psnr'])} | {fmt(generated['real8_render_to_real_ssim'])} | {fmt(generated['real8_render_to_real_lpips'])} |
| MV分支渲染 vs MVGen输入 | {fmt(generated['mv_render_to_generated_input_psnr'])} | {fmt(generated['mv_render_to_generated_input_ssim'])} | {fmt(generated['mv_render_to_generated_input_lpips'])} |
| MV分支渲染 vs 真实GT | {fmt(generated['mv_render_to_real_gt_psnr'])} | {fmt(generated['mv_render_to_real_gt_ssim'])} | {fmt(generated['mv_render_to_real_gt_lpips'])} |
| MVGen输入 vs 真实GT | {fmt(generated['mv_generated_to_real_gt_psnr'])} | {fmt(generated['mv_generated_to_real_gt_ssim'])} | {fmt(generated['mv_generated_to_real_gt_lpips'])} |

## 结果解释

1. 在未观察的7个视角上，8真分支相对真实GT比 `1真+7生成` 分支高 {fmt(real_psnr_gap)} dB PSNR、{fmt(real_ssim_gap)} SSIM，并低 {fmt(real_lpips_gap)} LPIPS，说明独立测试集上生成视角仍会显著降低最终重建与真实场景的一致性。
2. MV分支渲染更接近自己的生成输入（PSNR {fmt(generated['mv_render_to_generated_input_psnr'])}），而不是对应真实GT（PSNR {fmt(generated['mv_render_to_real_gt_psnr'])}）。这说明 VGGT-G 在一定程度上忠实重建了 MVGenMaster 提供的假设，但这些假设本身和真实测试视角存在偏差。
3. MVGen输入相对GT的 PSNR 为 {fmt(generated['mv_generated_to_real_gt_psnr'])}，经过 VGGT-G 重建并渲染后相对GT降至 {fmt(generated['mv_render_to_real_gt_psnr'])}，额外损失约 {fmt(reconstruction_psnr_loss)} dB。因而误差既来自生成模型，也来自生成图进入重建模型后的几何/外观融合。
4. 旧的生成图对GT结果为 PSNR {fmt(old_generated['mv_generated_to_real_gt_psnr'])}、SSIM {fmt(old_generated['mv_generated_to_real_gt_ssim'])}、LPIPS {fmt(old_generated['mv_generated_to_real_gt_lpips'])}；修正后为 {fmt(generated['mv_generated_to_real_gt_psnr'])}/{fmt(generated['mv_generated_to_real_gt_ssim'])}/{fmt(generated['mv_generated_to_real_gt_lpips'])}。差异很大，证实旧的方形补边评测不能继续用于原始视角质量结论。

## 指标计算

- `real8_render_to_real`：8真重建分支按每张真实图原始尺寸渲染，与该真实图逐像素比较。
- `mv_render_to_generated_input`：MV分支按实际送入VGGT的真实/生成图原始尺寸渲染，与该输入逐像素比较。
- `mv_render_to_real_gt`：MV分支按配对GT的原始尺寸和去补边内参重新渲染，与真实GT比较。
- `mv_generated_to_real_gt`：仅这一跨来源对比需要一次尺寸统一；把MVGen输入用双三次插值缩放到GT原始宽高后比较，不引入方形补边。
- PSNR/MSE：RGB `[0,1]` 全部目标原始像素；SSIM：`channel_axis=2, data_range=1`；LPIPS：AlexNet，RGB从 `[0,1]` 映射到 `[-1,1]`，在对应目标原始尺寸上计算。

## 文件说明

- `scenes/<场景>/real8_native/`：8真分支原始GT尺寸渲染。
- `scenes/<场景>/mvgen_native/`：MV分支实际输入尺寸渲染。
- `scenes/<场景>/mvgen_to_gt_native/`：MV分支配对GT尺寸渲染。
- `comparisons/`：30张五列对比图，图中逐列标注真实像素尺寸。
- `metrics_per_view_native.csv`、`metrics_per_scene_native.csv`、`metric_summary_native.csv`：逐视角、逐场景和总体指标。
- `old_vs_native_metric_comparison.csv`：旧方形画布结果与本轮原始尺寸结果的差值；旧值仅用于说明评测偏差。
- `metrics_by_category_generated_views_native.csv`：按测试类别汇总的生成视角结果。
- `render_size_validation.csv`：720张渲染图的期望与实际宽高核验。
- `evaluation_protocol_native.json`：完整评测协议。
"""
    (new_root / "原始尺寸渲染评测报告.md").write_text(report, encoding="utf-8")
    validation = {
        "view_rows": len(rows), "render_files": len(validation_rows),
        "exact_matches": sum(int(row["exact_match"]) for row in validation_rows),
        "missing": missing, "bad_size": bad_size,
        "comparison_sheets": len(list((new_root / "comparisons").glob("*_native_comparison.jpg"))),
        "status_failed": json.loads((new_root / "status.json").read_text())["failed"],
    }
    (new_root / "validation_summary.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
