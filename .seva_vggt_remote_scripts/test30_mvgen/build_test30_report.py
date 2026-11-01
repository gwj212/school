#!/usr/bin/env python3
"""Create category tables and a concise Chinese protocol/results report."""

from pathlib import Path
import json
import pandas as pd

ROOT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
EVAL = ROOT / "evaluation"

df = pd.read_csv(EVAL / "metrics_per_view.csv")
generated = df[df.source_type == "generated"].copy()
cols = [
    "real8_render_to_real_psnr", "real8_render_to_real_ssim", "real8_render_to_real_lpips",
    "mv_render_to_generated_input_psnr", "mv_render_to_generated_input_ssim", "mv_render_to_generated_input_lpips",
    "mv_render_to_real_gt_psnr", "mv_render_to_real_gt_ssim", "mv_render_to_real_gt_lpips",
    "mv_generated_to_real_gt_psnr", "mv_generated_to_real_gt_ssim", "mv_generated_to_real_gt_lpips",
]
category = generated.groupby("category")[cols].mean().reset_index()
category.to_csv(EVAL / "metrics_by_category_generated_views.csv", index=False, encoding="utf-8-sig")
selection = json.loads(Path("/root/seva_vggt/config/test30_mvgen/selection_protocol.json").read_text())
summary = pd.read_csv(EVAL / "metric_summary.csv")
timing = pd.read_csv(EVAL / "timing_summary.csv")

def metric(name, scope="generated_views_only"):
    return float(summary[(summary.scope == scope) & (summary.metric == name)].iloc[0]["mean"])

def seconds(name):
    return float(timing[timing.metric == name].iloc[0]["mean"])

scene = generated.groupby("scene")[cols].mean()
best = scene.sort_values("mv_generated_to_real_gt_psnr", ascending=False).head(5).index.tolist()
worst = scene.sort_values("mv_generated_to_real_gt_psnr").head(5).index.tolist()
mvload = json.loads((ROOT / "mvgen/status.json").read_text())["model_load_seconds"]
vgload = json.loads((ROOT / "reconstruction/status.json").read_text())["model_load_seconds"]

report = f"""# `my_test_multiview` 30场景 MVGenMaster→VGGT-G 测试报告

## 结论摘要

本轮已经在 `/root/vggt/my_test_multiview` 的独立测试划分完成30场景验证，共14类；此前小样本的5个场景全部排除。每个场景同时运行两条分支：`8张真实图→VGGT-G` 与 `1张真实图+7张MVGenMaster生成图→VGGT-G`。30个场景、60次重建、480张原尺寸渲染、60份无点数上限PLY均完整，无失败。

在7个生成视角上，MVGen图本身相对测试真图为 PSNR `{metric('mv_generated_to_real_gt_psnr'):.4f}`、SSIM `{metric('mv_generated_to_real_gt_ssim'):.4f}`、LPIPS `{metric('mv_generated_to_real_gt_lpips'):.4f}`。将这些图送入VGGT-G后，渲染相对MVGen输入为 PSNR `{metric('mv_render_to_generated_input_psnr'):.4f}`、SSIM `{metric('mv_render_to_generated_input_ssim'):.4f}`、LPIPS `{metric('mv_render_to_generated_input_lpips'):.4f}`；渲染相对真实测试图进一步降至 PSNR `{metric('mv_render_to_real_gt_psnr'):.4f}`、SSIM `{metric('mv_render_to_real_gt_ssim'):.4f}`、LPIPS `{metric('mv_render_to_real_gt_lpips'):.4f}`。这说明误差并不只来自生成模型，VGGT-G把生成视图融合成高斯时又引入了明显的渲染误差。

真实8图分支在同样7个非源视角上的渲染相对真实输入为 PSNR `{metric('real8_render_to_real_psnr'):.4f}`、SSIM `{metric('real8_render_to_real_ssim'):.4f}`、LPIPS `{metric('real8_render_to_real_lpips'):.4f}`。它明显好于生成分支渲染到GT，但绝对PSNR仍不高，原因包括高斯边界/背景伪影以及VGGT-G预测相机与输入像素并非严格重投影最优；不能把“视觉上物体完整”直接等价为高全图PSNR。

## 场景与数据来源

- 测试源：`/root/vggt/my_test_multiview`，每场景8张真实图。
- 选样：固定随机种子 `{selection['seed']}`；14类每类2场景，chair与plant各增加1场景，共30场景；排除此前5场景。
- 类别数：`{json.dumps(selection['category_counts'], ensure_ascii=False)}`。
- `my_test_multiview` 只保存RGB，不保存相机。MVGen目标相机从同一CO3D序列的 `/root/vggt/co3d_completion_test` 24帧副本中，以严格递增动态规划、最小96×96 RGB MSE匹配获得。平均匹配MSE为 `{sum(r['mean_match_mse_96'] for r in selection['records'])/30:.6f}`；这是最接近的可恢复相机，但不是原8图目录直接携带的相机真值，结论中必须保留这一限制。

## 模型与输出设置

- MVGenMaster：单图条件，1真+7生成，目标相机按上述匹配；50步，CFG=2，seed=23；原生比例画布。
- VGGT-G：用户现有改进模型，checkpoint step 35999，`depth_unproject`，输入长边518并pad到精确 `518×518`。
- PLY不设置点数上限。筛选依次为：XYZ/RGB/opacity/scale均有限、scale三轴均大于0、opacity≥0.10、删除场景内最大scale最高0.1%的异常高斯。PLY保存全部剩余中心点及RGB。
- 每一张用于指标的GT、生成输入和渲染图，都来自VGGT保存的精确 `518×518` 输入张量或直接在该相机画布上渲染；程序逐场景断言形状完全一致，未做事后拉伸对齐。

## 指标如何计算

- PSNR：RGB归一化到 `[0,1]`，在全部 `518×518×3` 元素上计算 `MSE=mean((I_ref-I_pred)^2)`，再算 `-10 log10(MSE)`；越高越好。
- SSIM：`skimage.metrics.structural_similarity`，`channel_axis=2`、`data_range=1.0`，整张RGB图计算；越高越好。
- LPIPS：AlexNet版LPIPS，将RGB由 `[0,1]` 映射到 `[-1,1]`，在完整518画布上计算；越低越好。
- 指标对包括：真实8图渲染↔真实图、生成分支渲染↔送入VGGT的生成图、生成分支渲染↔对应真实图、MVGen生成图↔对应真实图。源视角不纳入“generated_views_only”平均。

## 时间

- MVGen扩散50步：平均 `{seconds('mvgen_diffusion_s'):.3f}s/场景`。
- MVGen含DepthPro、准备和保存的热场景时间：`{seconds('mvgen_warm_scene_s'):.3f}s`。
- VGGT生成分支模型推理：`{seconds('mv_vggt_inference_s'):.3f}s`；含预处理、保存重建和PLY导出的热分支时间 `{seconds('mv_vggt_warm_total_s'):.3f}s`。
- 两分支16张渲染+全部指标：`{seconds('render_and_metrics_both_branches_s'):.3f}s/场景`。
- 当前完整生成分支热流程（MVGen+VGGT+两分支评价）：`{seconds('mv_complete_warm_pipeline_s'):.3f}s/场景`。
- 冷启动估计：`{seconds('mv_complete_cold_pipeline_estimate_s'):.3f}s/场景`，其中MVGen/DepthPro加载 `{mvload:.3f}s`，VGGT加载 `{vgload:.3f}s`。
- 连续处理30场景、模型只加载一次时的摊销时间：`{seconds('mv_complete_amortized_30_s'):.3f}s/场景`。

## 场景差异

- MVGen→GT PSNR最佳5场景：{', '.join(best)}。
- MVGen→GT PSNR最低5场景：{', '.join(worst)}。
- chair平均最好，而bottle、car、couch出现明显失败；因此不能用训练集或单一chair案例判断泛化。详细类别表见 `metrics_by_category_generated_views.csv`。

## 文件说明

- `evaluation/*_comparison.jpg`：每场景8行，依次为测试GT、真实8图重建渲染、MVGen输入、MVGen重建渲染。
- `reconstruction/scenes/<scene>/<branch>/renders/*.png`：原始518×518渲染图。
- `point_cloud_filtered_no_cap.ply`：筛选后无点数上限点云。
- `metrics_per_view.csv`、`metrics_per_scene.csv`、`metric_summary.csv`：逐视角、逐场景和统计汇总。
- `timing_per_scene.csv`、`timing_summary.csv`：分阶段完整计时。
- `evaluation_protocol.json`：机器可读的尺寸和指标协议。
"""
(EVAL / "测试集30场景实验报告.md").write_text(report, encoding="utf-8")
print(EVAL / "测试集30场景实验报告.md")
