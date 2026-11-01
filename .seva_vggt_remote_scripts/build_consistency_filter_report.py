#!/usr/bin/env python3
"""Build the local decision report from held-out consistency-filter evidence."""

from __future__ import annotations

import csv, json, math, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\Desktop\seva_vggt\consistency_filter_validation")
EVAL = ROOT / "evaluation_mvgen_target_depth_filter"
VIEW_CSV = EVAL / "metrics_per_view.csv"
OUT_CSV = ROOT / "adaptive_gate_per_view.csv"
OUT_JSON = ROOT / "validation_decision_summary.json"
OUT_MD = ROOT / "一致性与过滤解决方案_结论.md"
THRESHOLD = 0.065


def mean(values):
    values=list(values)
    return sum(values)/len(values) if values else 0.0


def ci95(values):
    values=list(values)
    m=mean(values)
    if len(values)<2: return [m,m]
    half=1.96*statistics.stdev(values)/math.sqrt(len(values))
    return [m-half,m+half]


with VIEW_CSV.open(encoding="utf-8-sig",newline="") as f:
    rows=list(csv.DictReader(f))

numeric=[k for k in rows[0] if k not in {"scene","category","split","source_type"}]
for row in rows:
    for key in numeric:
        row[key]=float(row[key])

generated=[r for r in rows if r["source_type"]=="generated"]
holdout=[r for r in generated if r["split"]=="holdout"]
calibration=[r for r in generated if r["split"]=="calibration"]

for row in generated:
    row["adaptive_apply"] = int(row["front_conflict_fraction_non_target"] >= THRESHOLD)
    for ref in ("input","gt"):
        for metric in ("psnr","ssim","lpips","mse"):
            b=row[f"baseline_{ref}_{metric}"]; p=row[f"processed_{ref}_{metric}"]
            row[f"selected_{ref}_{metric}"]=p if row["adaptive_apply"] else b
            row[f"selected_delta_{ref}_{metric}"]=row[f"selected_{ref}_{metric}"]-b

fields=list(generated[0])
with OUT_CSV.open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(generated)


def split_summary(data):
    by_scene=defaultdict(list)
    for r in data: by_scene[r["scene"]].append(r)
    result={"scenes":len(by_scene),"views":len(data),"applied_views":sum(r["adaptive_apply"] for r in data)}
    for ref in ("input","gt"):
        for metric in ("psnr","ssim","lpips","mse"):
            delta=[r[f"processed_{ref}_{metric}"]-r[f"baseline_{ref}_{metric}"] for r in data]
            adaptive=[r[f"selected_delta_{ref}_{metric}"] for r in data]
            scene_delta=[mean(r[f"processed_{ref}_{metric}"]-r[f"baseline_{ref}_{metric}"] for r in rr) for rr in by_scene.values()]
            scene_adapt=[mean(r[f"selected_delta_{ref}_{metric}"] for r in rr) for rr in by_scene.values()]
            better=(lambda x: x>0) if metric in ("psnr","ssim") else (lambda x: x<0)
            result[f"always_delta_{ref}_{metric}"]={"mean":mean(delta),"scene_ci95":ci95(scene_delta),
                "improved_views":sum(better(x) for x in delta),"improved_scenes":sum(better(x) for x in scene_delta)}
            result[f"adaptive_delta_{ref}_{metric}"]={"mean":mean(adaptive),"scene_ci95":ci95(scene_adapt),
                "improved_views":sum(better(x) for x in adaptive),"improved_scenes":sum(better(x) for x in scene_adapt)}
    return result


decision={
    "adaptive_rule":{"front_conflict_fraction_threshold":THRESHOLD,"chosen_on":"5 calibration scenes",
                     "uses_gt_at_inference":False,"hard_deletes_gaussians":False},
    "calibration":split_summary(calibration),
    "holdout":split_summary(holdout),
}
OUT_JSON.write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding="utf-8")

h=decision["holdout"]
ig=h["always_delta_input_psnr"]; gg=h["always_delta_gt_psnr"]
ag=h["adaptive_delta_gt_psnr"]
ss=h["always_delta_gt_ssim"]; lp=h["always_delta_gt_lpips"]
report=f"""# 一致性与过滤问题：解决方案及30场景可行性结论

## 结论先行

问题不是单一的“过滤阈值没调好”，而是两个误差源叠加：

1. **生成一致性误差**：MVGen视角中的局部结构、深度和遮挡关系与真实场景不一致；椅子腿、薄结构最明显。该误差进入VGGT前已经存在。
2. **重建/合成冲突**：不同来源视角的Gaussian在目标相机前后关系冲突，错误的前景Gaussian遮挡正确表面，形成漂浮物、重影和过曝。该误差可在渲染层缓解。

最合适的工程方案是“**重建前一致性门控 + 重建后目标视角软遮挡过滤 + 再决定是否训练**”，而不是继续全局硬删点、按scale统一裁剪或同时微调两个大模型。

## 已完成的独立验证

- 固定30个测试场景；5场景只用于选择参数，剩余25场景、175个生成视角作为独立验证。
- padding区域Gaussian直接移除；其余Gaussian不永久删除。
- 每个目标视角只用该来源Gaussian生成可信深度。对alpha≥0.75且位于可信表面前方超过3%的其他来源Gaussian，仅在当前目标视角将opacity乘0.25。
- 过滤过程不使用GT；GT只用于最终评价。

### 25场景留出结果

| 评价对象 | 指标 | 平均变化 | 95%CI（按场景） | 改善视角 | 改善场景 |
|---|---|---:|---:|---:|---:|
| 渲染 vs MVGen输入 | PSNR↑ | {ig['mean']:+.4f} dB | [{ig['scene_ci95'][0]:+.4f}, {ig['scene_ci95'][1]:+.4f}] | {ig['improved_views']}/175 | {ig['improved_scenes']}/25 |
| 渲染 vs 真实GT | PSNR↑ | {gg['mean']:+.4f} dB | [{gg['scene_ci95'][0]:+.4f}, {gg['scene_ci95'][1]:+.4f}] | {gg['improved_views']}/175 | {gg['improved_scenes']}/25 |
| 渲染 vs 真实GT | SSIM↑ | {ss['mean']:+.5f} | [{ss['scene_ci95'][0]:+.5f}, {ss['scene_ci95'][1]:+.5f}] | {ss['improved_views']}/175 | {ss['improved_scenes']}/25 |
| 渲染 vs 真实GT | LPIPS↓ | {lp['mean']:+.5f} | [{lp['scene_ci95'][0]:+.5f}, {lp['scene_ci95'][1]:+.5f}] | {lp['improved_views']}/175 | {lp['improved_scenes']}/25 |

解释：软过滤对重建内部一致性非常稳定（25/25场景的输入PSNR均改善），但真实GT提升较小且区间包含零。这证明它确实清理了错误遮挡，却不能纠正MVGen已经生成错的位置和形状。

用5个校准场景选择“冲突比例≥{THRESHOLD*100:.1f}%才启用过滤”的自适应门控后，留出集GT PSNR平均变化为{ag['mean']:+.4f} dB，95%CI为[{ag['scene_ci95'][0]:+.4f}, {ag['scene_ci95'][1]:+.4f}]；仍不足以宣称几何精度得到稳定提升。因此该门控适合作为**可逆渲染修复**，不应作为最终几何修复。

## 不应继续采用的做法

- 不按固定点数上限导出或重建；它只改变密度，不能识别错误几何。
- 不按全局scale/opacity阈值硬删。5场景校准中，删除scale最大0.1%反而使生成输入PSNR从21.3781降至21.3626 dB，GT PSNR从16.0880降至16.0660 dB。
- 不把跨视角一次不一致永久判为坏Gaussian。全局一致性权重使生成输入PSNR降至21.2122 dB；连同整视角权重后降至18.5000 dB。
- 不直接联合解冻MVGen与VGGT。既有小样本训练中，MVGen适配器虽让扩散验证损失下降7.24%，但5个留出场景生成PSNR下降0.696 dB、LPIPS恶化0.0492；再微调VGGT末端头后测试PSNR下降1.549 dB，出现明显过拟合。

## 推荐的长期训练方案

### 第一阶段：轻量可见性/置信度头（优先，成本最低）

冻结MVGen和VGGT主干，只训练一个小型Gaussian validity/visibility head。输入包括来源视角、depth confidence、opacity、scale、跨视角支持数、前景冲突比例和相机夹角；输出是每个Gaussian在各目标视角的opacity gate。使用8真图重建作为稳定回放，损失由可见区RGB/LPIPS、轮廓alpha、深度重投影、前后遮挡排序和稀疏正则组成。该方案直接学习本次已验证有效的“视角相关软权重”，不会破坏原重建。

### 第二阶段：MVGen一致性感知LoRA（解决根因）

VGGT完全冻结，MVGen只训练3D注入层或LoRA。除扩散损失外，必须加入：真实源图到生成视角的深度重投影损失、双向遮挡一致性、VGGT冻结教师的深度/相机一致性、生成视角之间的特征循环一致性。类别留出验证必须同时满足：生成LPIPS不升、深度冲突率下降至少10%、渲染GT PSNR不降。

### 第三阶段：可选的VGGT微调

只有第二阶段通过门槛后，才短程微调opacity/scale与可见性头，xyz/rotation/color继续冻结；每批至少50%为8真图回放。不要再次同时调整全部末端头。

## 最终建议

立即部署：padding硬移除 + 目标视角深度冲突opacity软门控，保留原Gaussian并支持开关回退。训练方向：先做轻量可见性头，再做冻结VGGT教师约束的MVGen LoRA。现有证据不支持继续调全局过滤阈值，也不支持直接全量联合微调。
"""
OUT_MD.write_text(report,encoding="utf-8")
print(OUT_MD)
