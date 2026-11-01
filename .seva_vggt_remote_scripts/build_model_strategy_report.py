#!/usr/bin/env python3
"""Build compact tables and a Chinese report for the model-strategy test."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/seva_vggt/outputs/model_strategy_test5")
MANIFESTS = Path("/root/seva_vggt/config/strategy_test5")
SCENES = [
    "chair_386_46082_92005",
    "car_599_92296_183295",
    "couch_349_36574_67726",
    "toybus_517_74123_144462",
    "ball_113_13373_24790",
]
TARGETS = [2, 5, 7, 10, 12, 15, 17]


def image_scene_summary():
    payload = json.load(open(ROOT / "evaluation/image_comparison.json", encoding="utf-8"))
    df = pd.DataFrame(payload["rows"])
    return df.groupby(["scene", "method"])[["psnr", "ssim", "lpips", "object_psnr", "background_psnr"]].mean().reset_index(), payload["summary"]


def geometry(scene, method):
    if method == "SEVA":
        d = json.load(open(ROOT / f"seva/{scene}/comparison_real8_vs_seva_1real_7gen.json"))
    else:
        d = json.load(open(ROOT / f"mvgenmaster_vggt/{scene}/metrics.json"))["geometry_vs_real8"]
    return d["symmetric_chamfer_l1_normalized"], d["f_scores_at_normalized_threshold"]["0.010"]["f_score"]


def timing(scene, method):
    if method == "SEVA":
        d = json.load(open(ROOT / f"seva/{scene}/run_metadata_seva_1real_7gen.json"))
        t = d["timing_seconds"]
        return t["seva_generation"], t["mixed_vggt_reconstruction"], t["warm_end_to_end"], t["cold_end_to_end"], d["peak_cuda_memory_mb"] / 1024
    g = json.load(open(ROOT / f"mvgenmaster/{scene}/generation_metadata.json"))
    v = json.load(open(ROOT / f"mvgenmaster_vggt/{scene}/metrics.json"))
    vg = v["reconstruction"]["elapsed_seconds"]
    return g["generation_seconds"], vg, g["generation_seconds"] + vg, g["cold_process_work_seconds"] + vg, max(g["peak_allocated_gib"], v["peak_allocated_gib"])


def make_sheet(scene, out_path):
    manifest = json.load(open(MANIFESTS / f"{scene}.json", encoding="utf-8"))
    thumb_w, thumb_h = 320, 192
    margin, label_w, header_h = 8, 90, 34
    canvas = Image.new("RGB", (label_w + 3 * thumb_w + 4 * margin, header_h + 7 * (thumb_h + margin) + margin), "white")
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(("Real GT", "SEVA", "MVGenMaster")):
        draw.text((label_w + margin + col * (thumb_w + margin) + 8, 8), title, fill="black")
    for row, idx in enumerate(TARGETS):
        view = manifest["views"][idx]
        paths = [
            Path(view["ground_truth"]),
            ROOT / f"seva/{scene}/seva_1real_7gen/images/view_{idx:02d}_generated.png",
            ROOT / f"mvgenmaster/{scene}/images/view_{idx:02d}_generated.png",
        ]
        y = header_h + row * (thumb_h + margin)
        draw.text((8, y + 8), f"view {idx:02d}", fill="black")
        for col, path in enumerate(paths):
            im = Image.open(path).convert("RGB")
            im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            frame = Image.new("RGB", (thumb_w, thumb_h), (238, 238, 238))
            frame.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
            canvas.paste(frame, (label_w + margin + col * (thumb_w + margin), y))
    canvas.save(out_path, quality=92)


def main():
    report_dir = ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    img_scene, image_overall = image_scene_summary()
    depth_frames = {
        "SEVA": pd.read_csv(ROOT / "depth_ghost_seva/per_scene_metrics.csv"),
        "MVGenMaster": pd.read_csv(ROOT / "depth_ghost_mvgen/per_scene_metrics.csv"),
    }
    rows = []
    for scene in SCENES:
        for method in ("SEVA", "MVGenMaster"):
            image_row = img_scene[(img_scene.scene == scene) & (img_scene.method == method)].iloc[0]
            depth_row = depth_frames[method][depth_frames[method].scene == scene].iloc[0]
            chamfer, fscore = geometry(scene, method)
            gen, vg, warm, cold, vram = timing(scene, method)
            rows.append({
                "scene": scene, "method": method,
                "psnr": image_row.psnr, "ssim": image_row.ssim, "lpips": image_row.lpips,
                "object_psnr": image_row.object_psnr, "background_psnr": image_row.background_psnr,
                "depth_absrel_object": depth_row["mean.gt_vs_seva_depth.object.absrel"],
                "depth_absrel_boundary": depth_row["mean.gt_vs_seva_depth.boundary.absrel"],
                "depth_inconsistent_rate_0p20": depth_row["mean.gt_vs_seva_depth.object.inconsistent_rate_0.20"],
                "source_warp_front_conflict": depth_row["mean.source_warp_vs_seva_class.front_conflict_rate"],
                "pose_rotation_deg": depth_row["mean.pose.mixed_vs_real.rotation_deg"],
                "pose_translation_direction_deg": depth_row["mean.pose.mixed_vs_real.translation_direction_deg"],
                "gaussian_front_conflict": depth_row["mean.gaussian_to_source.front_conflict_rate"],
                "normalized_chamfer_l1": chamfer, "fscore_0p010": fscore,
                "generation_s": gen, "vggt_s": vg, "warm_total_s": warm, "cold_total_s": cold,
                "peak_vram_gib": vram,
            })
    per_scene = pd.DataFrame(rows)
    per_scene.to_csv(report_dir / "test5_per_scene.csv", index=False, encoding="utf-8-sig")
    numeric = [c for c in per_scene.columns if c not in ("scene", "method")]
    summary = per_scene.groupby("method")[numeric].mean().reset_index()
    summary.to_csv(report_dir / "test5_method_summary.csv", index=False, encoding="utf-8-sig")

    schemes = [
        {"方案": "1 直接替换SEVA", "立即可执行": "是", "当前硬件可行性": "高", "先验解决概率": "65%-75%", "测试后概率": "70%-80%改善整体管线；55%-70%彻底消除重影", "结论": "当前最优，推荐MVGenMaster进入30场景复验"},
        {"方案": "2 SEVA与VGGT全量联合微调", "立即可执行": "否（需改梯度链）", "当前硬件可行性": "低", "先验解决概率": "10%-25%全量；55%-70%分阶段LoRA", "测试后概率": "不变", "结论": "不做全量端到端；后续只做交替LoRA+VGGT小头"},
        {"方案": "3 从VGGT训练新扩散模型", "立即可执行": "否", "当前硬件可行性": "很低", "先验解决概率": "<10%从零训练；VGGT-Diff六视图70%-85%", "测试后概率": "单/双视图仅约5%-15%，且VGGT-Omega权重受限", "结论": "当前不选，等待权重授权或改成变量源视图适配"},
        {"方案": "补充：几何分层融合", "立即可执行": "是", "当前硬件可行性": "高", "先验解决概率": "75%-85%（完整方案）", "测试后概率": "朴素warp融合未优于纯MVGenMaster", "结论": "否决朴素覆盖；保留深度门控/视图筛选版本"},
    ]
    pd.DataFrame(schemes).to_csv(report_dir / "scheme_assessment.csv", index=False, encoding="utf-8-sig")
    for scene in SCENES:
        make_sheet(scene, report_dir / f"{scene}_comparison.jpg")

    sm = {r.method: r for _, r in summary.iterrows()}
    s, m = sm["SEVA"], sm["MVGenMaster"]
    report = f"""# SEVA替换/联合微调/VGGT扩散方案评估与小样本测试

## 结论

当前最优方案是**方案1：用MVGenMaster替换SEVA，并保持VGGT-G不变**。独立 `co3d_completion_test` 的5场景结果显示，MVGenMaster不仅图像指标更好，而且整体几何指标没有退化：PSNR提高 `{m.psnr-s.psnr:.3f} dB`，SSIM提高 `{m.ssim-s.ssim:.4f}`，LPIPS降低 `{s.lpips-m.lpips:.4f}`，F-score@0.010提高 `{m.fscore_0p010-s.fscore_0p010:.4f}`。物体深度AbsRel由 `{s.depth_absrel_object:.4f}` 降至 `{m.depth_absrel_object:.4f}`，源视图重投影前向冲突率由 `{s.source_warp_front_conflict:.4f}` 降至 `{m.source_warp_front_conflict:.4f}`。

但它还不是“可直接发布”的最终方案：`car_599_92296_183295` 的深度和Chamfer明显变差，说明替换模型仍存在类别/场景失败。建议先扩展到30个**独立测试场景**确认，再决定是否进入LoRA微调。

## 测试前备份

- 备份目录：`/root/seva_vggt/backups/pre_model_strategy_20261014_2055`
- 内容：SEVA-VGGT代码配置、VGGT代码、未提交补丁、Git状态、文件SHA256清单。
- 所有候选代码位于 `/root/seva_vggt/candidates`，输出位于 `/root/seva_vggt/outputs/model_strategy_test5`；未覆盖原工程和原实验目录。

## 三个方案的工程概率

概率表示“在当前数据、单张RTX A6000和现有代码基础上，明显改善一致性/重影且不显著损伤重建”的工程判断，不是统计置信度。

|方案|概率与判断|
|---|---|
|直接替换SEVA|测试前65%-75%；测试后估计70%-80%可改善整体管线，彻底解决重影约55%-70%。当前首选。|
|SEVA+VGGT全量联合微调|当前全量端到端仅10%-25%；改为“SEVA LoRA + VGGT深度/DPT/Gaussian小头交替训练”为55%-70%。不建议直接全量训练。|
|基于VGGT新训练扩散模型|单卡从零训练<10%。公开VGGT-Diff在6输入视图条件下潜力70%-85%，但当前目标是1/2输入视图，直接适配约5%-15%，且依赖的VGGT-Omega权重需要授权。|

## 独立小样本设置

- 场景来源：`/root/vggt/co3d_completion_test`，不是此前30场景所在的train目录。
- 该划分能验证“未参与当前VGGT-G训练/此前benchmark”的管线留出效果；但MVGenMaster公开训练数据包含CO3Dv2，当前无法证明这5个场景未被其预训练见过。因此这里不是严格的外部数据集泛化证明。
- 5类场景：chair、car、couch、toybus、ball；每场景20张真实图及精确相机参数。
- 输入组合：1张真实图 + 7张生成图；目标manifest索引为 `2,5,7,10,12,15,17`。
- SEVA基线：`T=8`，12步，CFG=2，短边576、保持比例，seed=23。
- MVGenMaster：精确目标相机，DepthPro源深度按源物体中值和相机半径做尺度对齐，50步，CFG=2，seed=23，输出保持比例（短边320）。
- VGGT-G：原模型与step 35999高斯权重不变，最大边518；PLY导出上限50万点；几何比较固定采样2万点。
- 图像比较：两种生成结果都缩放到MVGenMaster原生画布，GT按同一中心裁剪规则对齐，避免分辨率造成偏差。
- 几何“GT”：real-8的VGGT重建为配对伪参考，并非扫描真值。

## Test-5平均结果

|指标|SEVA|MVGenMaster|变化|
|---|---:|---:|---:|
|PSNR↑|{s.psnr:.4f}|{m.psnr:.4f}|{m.psnr-s.psnr:+.4f}|
|SSIM↑|{s.ssim:.4f}|{m.ssim:.4f}|{m.ssim-s.ssim:+.4f}|
|LPIPS↓|{s.lpips:.4f}|{m.lpips:.4f}|{m.lpips-s.lpips:+.4f}|
|物体深度AbsRel↓|{s.depth_absrel_object:.4f}|{m.depth_absrel_object:.4f}|{m.depth_absrel_object-s.depth_absrel_object:+.4f}|
|边界深度AbsRel↓|{s.depth_absrel_boundary:.4f}|{m.depth_absrel_boundary:.4f}|{m.depth_absrel_boundary-s.depth_absrel_boundary:+.4f}|
|深度不一致率>20%↓|{s.depth_inconsistent_rate_0p20:.4f}|{m.depth_inconsistent_rate_0p20:.4f}|{m.depth_inconsistent_rate_0p20-s.depth_inconsistent_rate_0p20:+.4f}|
|源重投影前向冲突↓|{s.source_warp_front_conflict:.4f}|{m.source_warp_front_conflict:.4f}|{m.source_warp_front_conflict-s.source_warp_front_conflict:+.4f}|
|归一化Chamfer-L1↓|{s.normalized_chamfer_l1:.5f}|{m.normalized_chamfer_l1:.5f}|{m.normalized_chamfer_l1-s.normalized_chamfer_l1:+.5f}|
|F-score@0.010↑|{s.fscore_0p010:.4f}|{m.fscore_0p010:.4f}|{m.fscore_0p010-s.fscore_0p010:+.4f}|
|生成时间(s)↓|{s.generation_s:.2f}|{m.generation_s:.2f}|{m.generation_s-s.generation_s:+.2f}|
|热端到端(s)↓|{s.warm_total_s:.2f}|{m.warm_total_s:.2f}|{m.warm_total_s-s.warm_total_s:+.2f}|
|冷端到端估计(s)↓|{s.cold_total_s:.2f}|{m.cold_total_s:.2f}|{m.cold_total_s-s.cold_total_s:+.2f}|
|峰值显存(GiB)↓|{s.peak_vram_gib:.2f}|{m.peak_vram_gib:.2f}|{m.peak_vram_gib-s.peak_vram_gib:+.2f}|

时间口径说明：两者“生成时间”均为模型生成阶段，最适合直接比较。SEVA的“热端到端”还包含保存生成结果和图像指标计算，而MVGenMaster当前记录的是生成+VGGT重建；因此端到端列只能视为工程估计，不能据此声称严格加速。后续30场景复验应统一计时边界后再做速度结论。

## 失败案例与解释

- `car_599_92296_183295`：MVGenMaster的深度AbsRel和归一化Chamfer均恶化，是当前最主要反例。
- 平均旋转误差基本没有改善，说明图像扩散模型替换不能自动修复VGGT的全部相机估计误差。
- MVGenMaster原生分辨率较低，虽然LPIPS/PSNR提升，但细节上限仍受320短边限制。
- 初始train/benchmark 5场景中，MVGenMaster图像和Chamfer也改善，但深度AbsRel恶化；由于可能与MVGenMaster的CO3Dv2训练集重合，该组不作为泛化结论。

## 补充方案测试

测试了“源图深度+精确相机前向warp覆盖MVGenMaster”的朴素几何分层融合。在初始5场景中，PSNR继续提高，但SSIM/LPIPS、Chamfer和深度没有同时改善，因此**不采用直接像素覆盖**。更合理的后续形式应是：只用重投影深度冲突作为软置信度/视图筛选，不把warp RGB硬覆盖到生成图。

## 推荐路线

1. 先把MVGenMaster作为SEVA的候选替换，在30个独立test场景上复验；保留逐场景/类别统计，不只看总平均。
2. 生成阶段增加“源深度重投影冲突分数”，对冲突高的生成视图降权或丢弃，避免一张坏图污染VGGT。
3. 若30场景仍稳定，再做轻量联合微调：冻结VGGT主体，只训练SEVA/MVGenMaster LoRA和VGGT深度/DPT/Gaussian小头，采用单步扩散监督，不展开12/50步反传。
4. 暂不从零训练VGGT扩散模型；等VGGT-Omega授权后再评估VGGT-Diff，并先补变量源视图/单视图适配。
"""
    (report_dir / "模型方案评估与小样本测试.md").write_text(report, encoding="utf-8")
    print(report_dir)


if __name__ == "__main__":
    main()
