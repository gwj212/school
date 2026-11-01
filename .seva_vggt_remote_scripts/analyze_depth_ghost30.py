from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path(os.environ.get(
    "DEPTH_GHOST_ROOT",
    r"D:\Desktop\seva_vggt\depth_ghost30_cfg2_t8_s12",
))
RNG = np.random.default_rng(23)


def paired_test(df: pd.DataFrame, left: str, right: str, label: str,
                higher_is_better: bool | None = None) -> dict:
    pair = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    a = pair[left].to_numpy(float)
    b = pair[right].to_numpy(float)
    delta = b - a
    if len(delta) < 2:
        return {"comparison": label, "n": len(delta)}
    boot = np.empty(5000, dtype=float)
    for i in range(len(boot)):
        boot[i] = RNG.choice(delta, size=len(delta), replace=True).mean()
    try:
        w = wilcoxon(b, a, zero_method="wilcox")
        wp = float(w.pvalue)
    except ValueError:
        wp = None
    t = ttest_rel(b, a, nan_policy="omit")
    improved = None
    if higher_is_better is not None:
        improved = float(np.mean(delta > 0 if higher_is_better else delta < 0))
    return {
        "comparison": label,
        "left_metric": left,
        "right_metric": right,
        "n": len(delta),
        "left_mean": float(a.mean()),
        "right_mean": float(b.mean()),
        "paired_delta_right_minus_left_mean": float(delta.mean()),
        "paired_delta_median": float(np.median(delta)),
        "paired_delta_ci95_low": float(np.quantile(boot, 0.025)),
        "paired_delta_ci95_high": float(np.quantile(boot, 0.975)),
        "cohen_dz": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) > 0 else None,
        "paired_t_p": float(t.pvalue),
        "wilcoxon_p": wp,
        "improved_rate": improved,
    }


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n.a."
    return f"{float(value):.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n.a."
    return f"{100 * float(value):.{digits}f}%"


def main() -> None:
    view = pd.read_csv(ROOT / "per_view_metrics.csv")
    scene = pd.read_csv(ROOT / "per_scene_metrics.csv")
    ablation = pd.read_csv(ROOT / "opacity_gate_ablation_per_view.csv")

    tests = [
        paired_test(
            view,
            "source_warp_vs_gt_class.match_rate",
            "source_warp_vs_seva_class.match_rate",
            "真实输入深度重投影：与真实目标深度匹配率 vs 与SEVA目标深度匹配率",
            higher_is_better=True,
        ),
        paired_test(
            view,
            "source_warp_vs_gt_class.front_conflict_rate",
            "source_warp_vs_seva_class.front_conflict_rate",
            "真实输入深度重投影：真实目标前向冲突率 vs SEVA目标前向冲突率",
            higher_is_better=False,
        ),
        paired_test(
            view,
            "pose.real_vs_gt.rotation_deg",
            "pose.mixed_vs_gt.rotation_deg",
            "相机旋转误差：8真实输入 vs 1真实+7SEVA",
            higher_is_better=False,
        ),
        paired_test(
            view,
            "real_depth_conf_mean",
            "mixed_depth_conf_mean",
            "物体区域VGGT深度置信度：真实图 vs SEVA图",
            higher_is_better=True,
        ),
        paired_test(
            scene,
            "dedup_real_raw_keep_ratio",
            "dedup_mixed_raw_keep_ratio",
            "离线深度去重原始保留率：8真实输入 vs 1真实+7SEVA",
            higher_is_better=None,
        ),
        paired_test(
            scene,
            "dedup_real_high_opacity_rejected_rate",
            "dedup_mixed_high_opacity_rejected_rate",
            "应被深度去重但仍为高opacity的比例：8真实输入 vs 1真实+7SEVA",
            higher_is_better=False,
        ),
    ]

    original = ablation[ablation.condition == "original"].set_index(["scene", "slot"])
    for condition in [x for x in ablation.condition.unique() if x != "original"]:
        current = ablation[ablation.condition == condition].set_index(["scene", "slot"])
        joined = original[["psnr", "ssim", "lpips"]].join(
            current[["psnr", "ssim", "lpips"]], lsuffix="_original", rsuffix="_condition", how="inner"
        )
        for metric, higher in (("psnr", True), ("ssim", True), ("lpips", False)):
            tests.append(paired_test(
                joined,
                f"{metric}_original",
                f"{metric}_condition",
                f"门控消融 {condition}: {metric} vs original",
                higher_is_better=higher,
            ))
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(ROOT / "hypothesis_tests.csv", index=False, encoding="utf-8-sig")

    scene_ablation = ablation.groupby(["scene", "condition"], as_index=False).agg(
        psnr=("psnr", "mean"), ssim=("ssim", "mean"), lpips=("lpips", "mean"),
        removed_fraction=("removed_fraction_of_generated_slot", "mean"),
    )
    base = scene_ablation[scene_ablation.condition == "original"].set_index("scene")
    scene_ablation["delta_psnr"] = scene_ablation.apply(
        lambda r: r.psnr - base.loc[r.scene, "psnr"], axis=1
    )
    scene_ablation["delta_ssim"] = scene_ablation.apply(
        lambda r: r.ssim - base.loc[r.scene, "ssim"], axis=1
    )
    scene_ablation["delta_lpips"] = scene_ablation.apply(
        lambda r: r.lpips - base.loc[r.scene, "lpips"], axis=1
    )
    scene_ablation.to_csv(ROOT / "opacity_gate_ablation_by_scene.csv", index=False, encoding="utf-8-sig")

    category = scene.groupby("category", as_index=False).agg(
        scene_count=("scene", "count"),
        depth_absrel=("mean.gt_vs_seva_depth.object.absrel", "mean"),
        boundary_absrel=("mean.gt_vs_seva_depth.boundary.absrel", "mean"),
        warp_gt_front_conflict=("mean.source_warp_vs_gt_class.front_conflict_rate", "mean"),
        warp_seva_front_conflict=("mean.source_warp_vs_seva_class.front_conflict_rate", "mean"),
        pose_rotation_drift_deg=("mean.pose.mixed_vs_real.rotation_deg", "mean"),
        pose_translation_direction_drift_deg=("mean.pose.mixed_vs_real.translation_direction_deg", "mean"),
        confidence_ratio=("mean.mixed_to_real_conf_ratio", "mean"),
        dedup_keep_real=("dedup_real_raw_keep_ratio", "mean"),
        dedup_keep_mixed=("dedup_mixed_raw_keep_ratio", "mean"),
    )
    category.to_csv(ROOT / "category_summary.csv", index=False, encoding="utf-8-sig")

    protocol = {
        "experiment_name": "SEVA to VGGT depth and ghosting diagnosis",
        "scene_count": 30,
        "target_view_count": 210,
        "baseline": "8 real images",
        "test_branch": "1 real image + 7 SEVA images",
        "seva": {"T": 8, "sampling_steps": 12, "CFG": 2, "seed": 23,
                 "short_side": 576, "preserve_aspect_ratio": True, "view_mapping": "nearest-gt"},
        "vggt": {"saved_input_size": [518, 518], "depth": "VGGT predicted depth",
                 "camera": "VGGT predicted camera with exact dataset camera audit",
                 "checkpoint": "existing improved VGGT-G checkpoint used by cfg sweep"},
        "depth_alignment": "Scale-only alignment by median depth ratio on high-confidence pixels in the shared real source view",
        "source_warp": "Forward projection with 2x2 z-buffer splatting; only source-visible support is evaluated",
        "regions": ["valid image", "object mask", "mask boundary", "lower-object proxy", "background"],
        "mask_quality": {"empty_exact_masks": int(scene.object_mask_fallback_view_count.sum()),
                         "fallback": "valid image region", "affected_scenes": 4},
        "opacity_gate_ablation": ["original", "GT-image-depth oracle 10%", "GT-image-depth oracle 20%",
                                    "GT-image-depth oracle 30%", "source-warp front-conflict 20%"],
        "important_limitation": "GT depth denotes VGGT depth estimated from a real target image, not a sensor-measured metric depth map.",
    }
    (ROOT / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    v = view
    depth_absrel = v["gt_vs_seva_depth.object.absrel"].mean()
    boundary_absrel = v["gt_vs_seva_depth.boundary.absrel"].mean()
    bad20 = v["gt_vs_seva_depth.object.inconsistent_rate_0.20"].mean()
    gt_match = v["source_warp_vs_gt_class.match_rate"].mean()
    seva_match = v["source_warp_vs_seva_class.match_rate"].mean()
    gt_front = v["source_warp_vs_gt_class.front_conflict_rate"].mean()
    seva_front = v["source_warp_vs_seva_class.front_conflict_rate"].mean()
    pose_rot_median = v["pose.mixed_vs_real.rotation_deg"].median()
    pose_trans_median = v["pose.mixed_vs_real.translation_direction_deg"].median()
    conf_ratio_median = v["mixed_to_real_conf_ratio"].median()
    dedup_real = scene["dedup_real_raw_keep_ratio"].mean()
    dedup_mixed = scene["dedup_mixed_raw_keep_ratio"].mean()
    fallback_real = int(scene["dedup_real_fallback_triggered"].sum())
    fallback_mixed = int(scene["dedup_mixed_fallback_triggered"].sum())
    ab20 = scene_ablation[scene_ablation.condition == "gt_depth_oracle_0p20"][
        ["psnr", "ssim", "lpips", "removed_fraction", "delta_psnr", "delta_ssim", "delta_lpips"]
    ].mean()
    sw20 = scene_ablation[scene_ablation.condition == "source_warp_front_0p20"][
        ["psnr", "ssim", "lpips", "removed_fraction", "delta_psnr", "delta_ssim", "delta_lpips"]
    ].mean()
    corr = pd.read_csv(ROOT / "correlations.csv")
    corr_row = corr[(corr.x == "gt_vs_seva_depth.object.absrel") &
                    (corr.y == "source_warp_vs_seva_class.front_conflict_rate")].iloc[0]

    chairs = scene[scene.category == "chair"].set_index("scene")
    chair = chairs.loc["chair_385_45726_91419"]
    chair_ab = scene_ablation[(scene_ablation.scene == "chair_385_45726_91419") &
                              (scene_ablation.condition == "gt_depth_oracle_0p20")].iloc[0]

    report = f"""# SEVA→VGGT 深度与重影排查报告

## 实验设置

- 场景：30个，覆盖12类物体。
- 对照组：同一视角集合的8张真实图。
- 实验组：1张真实图+7张SEVA生成图，共210个生成目标视角。
- SEVA：T=8、12步、CFG=2、seed=23、短边576并保持比例、nearest-gt。
- VGGT：复用CFG扫描实验保存的改进版VGGT-G输出，不重新推理、不修改模型。
- 深度：全部来自VGGT。实验中的“GT深度”指真实目标图经VGGT估计的深度，不是传感器真值。
- 混合分支深度使用共享真实输入视角的高置信区域做仅尺度对齐，不使用平移项。
- 重投影：输入真实图深度经VGGT相机投影至目标视角，采用2×2 Z-buffer splat并只评价有效覆盖。
- 消融：仅离线把深度冲突生成像素对应的高斯opacity置零，不改权重、不重新训练。
- 数据质量：12/240个所需mask为空，涉及4个场景；这些视角退化为有效图像区域，并在逐视角表中标记。

## 主要结果

### 1. 重投影链条本身基本有效

真实输入深度投影到真实目标图后，平均深度匹配率为{pct(gt_match)}，前向冲突率仅{pct(gt_front)}。这说明坐标约定、深度尺度对齐和重投影实现总体可信，虽然大视角变化和遮挡仍会造成不可见区域。

### 2. SEVA输入后出现显著深度不一致

真实目标图与SEVA图的物体区域深度AbsRel均值为{pct(depth_absrel)}，边界区域升至{pct(boundary_absrel)}；平均{pct(bad20)}的物体像素深度相对误差超过20%。同一真实输入重投影与SEVA深度的匹配率降至{pct(seva_match)}，前向冲突率升至{pct(seva_front)}。

深度AbsRel与“真实输入重投影→SEVA深度”的前向冲突率呈明显正相关：Pearson r={fmt(corr_row.pearson_r)}, p={corr_row.pearson_p:.2e}；Spearman ρ={fmt(corr_row.spearman_rho)}, p={corr_row.spearman_p:.2e}。因此SEVA生成图对应的VGGT深度与真实几何不一致不是单个场景偶然现象。

### 3. 相机漂移是第二个独立因素

1真+7生成分支相对8真实分支的相对旋转漂移中位数为{fmt(pose_rot_median, 2)}°，平移方向漂移中位数为{fmt(pose_trans_median, 2)}°。混合分支物体区域深度置信度仅为真实分支的{pct(conf_ratio_median)}（中位数）。因此重影不能完全归因于像素深度，SEVA图还会影响VGGT的多视角相机估计和整体置信度。

### 4. 当前“去重”不是推理阶段的硬去重

代码审计确认：深度去重掩码只在训练时监督opacity；推理时GaussianHead仍为每个视角、每个像素输出一个高斯。离线重新计算掩码时，8真实分支平均保留率为{pct(dedup_real)}，混合分支升至{pct(dedup_mixed)}，说明混合输入更难确认重复表面。fallback在真实分支触发{fallback_real}/30个场景，在混合分支触发{fallback_mixed}/30个场景。

### 5. 深度门控能改善结果，但改善幅度较小

使用真实目标图深度作为诊断oracle、阈值20%时，平均移除{pct(ab20.removed_fraction)}的生成槽位像素，高斯渲染PSNR平均变化{ab20.delta_psnr:+.3f} dB，SSIM变化{ab20.delta_ssim:+.4f}，LPIPS变化{ab20.delta_lpips:+.4f}。使用部署时可获得的真实输入重投影门控时，PSNR平均变化{sw20.delta_psnr:+.3f} dB，SSIM变化{sw20.delta_ssim:+.4f}，LPIPS变化{sw20.delta_lpips:+.4f}。

oracle门控的PSNR和LPIPS总体改善，但幅度很小，且SSIM没有稳定改善。这说明深度冲突确实参与了重影形成，但直接删除冲突高斯会同时制造空洞，深度并不是唯一原因。

## 当前椅子场景

`chair_385_45726_91419`的物体深度AbsRel为{pct(chair['mean.gt_vs_seva_depth.object.absrel'])}，边界AbsRel为{pct(chair['mean.gt_vs_seva_depth.boundary.absrel'])}。真实输入重投影对真实目标的前向冲突率为{pct(chair['mean.source_warp_vs_gt_class.front_conflict_rate'])}，对SEVA深度升至{pct(chair['mean.source_warp_vs_seva_class.front_conflict_rate'])}。混合分支相对旋转漂移为{fmt(chair['mean.pose.mixed_vs_real.rotation_deg'], 2)}°，深度置信度只剩真实分支的{pct(chair['mean.mixed_to_real_conf_ratio'])}。

该场景20% oracle门控后，PSNR变化{chair_ab.delta_psnr:+.3f} dB、SSIM变化{chair_ab.delta_ssim:+.4f}、LPIPS变化{chair_ab.delta_lpips:+.4f}。可视化中椅背、座面和腿部的深度色带确实不同，但简单删除高残差高斯只能轻微缓解，不能彻底消除双层结构。

## 结论

1. **深度问题成立。** 30场景、210视角上，SEVA图对应的VGGT深度与真实图深度存在系统差异，物体边界最明显。
2. **SEVA→VGGT深度不一致会促成重影。** 重投影验证、相关性和oracle门控都支持这一因果链，但作用量不是全部。
3. **相机漂移同样重要。** 生成图改变了VGGT的多视角相机解，部分重影来自同一表面被放进不同坐标位置。
4. **当前去重机制存在训练—推理差异。** 模型只学习用opacity模拟去重，推理时没有重新施加深度一致性掩码；当生成域深度置信度下降时，冲突高斯仍可能保留较高opacity。
5. **简单硬删除不是最终方案。** 它可小幅改善PSNR/LPIPS，却可能降低SSIM并产生空洞。后续更合适的方向是软置信度门控、相机/深度联合一致性权重，或在训练中加入真实—生成配对的跨视角深度与位姿一致性损失。

## 建议的下一步（本轮未执行）

- 先把深度残差、相机残差和VGGT深度置信度作为连续权重输入Gaussian融合，而不是二值删除。
- 微调时加入真实目标深度估计与生成目标深度估计的一致性损失，并对物体边界、薄结构提高权重。
- 对相机头增加真实视角锚定，防止大量生成图改变全局相机解。
- 在推理阶段显式执行软去重，避免完全依赖训练得到的opacity泛化。
"""
    (ROOT / "实验报告_深度与重影排查.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
