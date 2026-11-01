#!/usr/bin/env python3
"""Build final three-stage ablation tables, comparisons and Chinese report."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


SCENES = ["chair_620_101264_203062", "laptop_373_41558_82973", "bicycle_62_4318_10726",
          "car_336_34863_64261", "bottle_601_93006_187080"]
BASE = Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
ROOT = Path("/root/seva_vggt/outputs/joint_finetune_small")
OUT = ROOT / "delivery"
MANIFESTS = Path("/root/seva_vggt/config/joint_small_test5")
METHODS = {
    "A_原始MVGen+冻结VGGT": BASE / "evaluation_native_v2",
    "B_3D适配MVGen+冻结VGGT": ROOT / "adapted_render_test5",
    "C_3D适配MVGen+微调VGGT末端头": ROOT / "joint_tuned_render_test5",
}


def rows(path):
    with path.open(encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def write(path, data):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def mean(items, key): return float(np.mean([float(x[key]) for x in items]))


def filtered_xyz(path):
    r = torch.load(path, map_location="cpu"); g = r["gaussians"]
    xyz = g["xyz"][0].reshape(-1,3).float().numpy(); opa = g["opacity"][0].reshape(-1).float().numpy()
    scale = g["scale"][0].reshape(-1,3).float().numpy()
    finite = np.isfinite(xyz).all(1)&np.isfinite(opa)&np.isfinite(scale).all(1)
    pre = finite&(scale>0).all(1)&(opa>=.10); sm = scale.max(1); lim=float(np.percentile(sm[pre],99.9))
    return xyz[pre&(sm<=lim)]


def fitted(path, w, h):
    with Image.open(path) as im: cp=im.convert("RGB").copy()
    cp.thumbnail((w,h),Image.Resampling.LANCZOS); out=Image.new("RGB",(w,h),"white")
    out.paste(cp,((w-cp.width)//2,(h-cp.height)//2)); return out


def sheet(scene):
    m=json.loads((MANIFESTS/f"{scene}.json").read_text()); cw,ch,lw,hh=205,164,70,58
    titles=["GT","Original MVGen","3D-adapt MVGen","A render","B render","C joint render"]
    canvas=Image.new("RGB",(lw+cw*len(titles),hh+ch*7),"white"); d=ImageDraw.Draw(canvas)
    d.text((5,5),scene,fill="black")
    for j,t in enumerate(titles): d.text((lw+j*cw+5,31),t,fill="black")
    for r,s in enumerate(range(1,8)):
        y=hh+r*ch; d.text((5,y+8),f"view {s:02d}",fill="black")
        ps=[Path(m["views"][s]["image"]), BASE/"mvgen"/scene/"images"/f"view_{s:02d}_generated.png",
            ROOT/"adapted_mvgen_test5"/scene/"images"/f"view_{s:02d}_generated.png",
            BASE/"evaluation_native_v2"/"scenes"/scene/"mvgen_to_gt_native"/f"view_{s:02d}_render.png",
            ROOT/"adapted_render_test5"/"scenes"/scene/"mvgen_to_gt_native"/f"view_{s:02d}_render.png",
            ROOT/"joint_tuned_render_test5"/"scenes"/scene/"mvgen_to_gt_native"/f"view_{s:02d}_render.png"]
        for j,p in enumerate(ps): canvas.paste(fitted(p,cw,ch-18),(lw+j*cw,y))
    canvas.save(OUT/"对比图"/f"{scene}_三阶段对比.jpg",quality=95)


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"对比图").mkdir(exist_ok=True)
    sys.path.insert(0,"/root/seva_vggt/bridge"); import run_pipeline as rp
    metric_fields={
        "生成图_PSNR":"mv_generated_to_real_gt_psnr","生成图_SSIM":"mv_generated_to_real_gt_ssim",
        "生成图_LPIPS":"mv_generated_to_real_gt_lpips","渲染对输入_PSNR":"mv_render_to_generated_input_psnr",
        "渲染对输入_SSIM":"mv_render_to_generated_input_ssim","渲染对输入_LPIPS":"mv_render_to_generated_input_lpips",
        "渲染对GT_PSNR":"mv_render_to_real_gt_psnr","渲染对GT_SSIM":"mv_render_to_real_gt_ssim",
        "渲染对GT_LPIPS":"mv_render_to_real_gt_lpips"}
    loaded={n:[r for r in rows(p/"metrics_per_view_native.csv") if r["scene"] in SCENES and int(r["slot"])>0]
            for n,p in METHODS.items()}
    summary=[]
    for name,rr in loaded.items():
        row={"方法":name,"场景数":5,"生成视角数":35}
        row.update({cn:mean(rr,f) for cn,f in metric_fields.items()}); summary.append(row)
    write(OUT/"三阶段消融汇总.csv",summary)

    per_scene=[]
    for scene in SCENES:
        for name,rr in loaded.items():
            ss=[r for r in rr if r["scene"]==scene]
            row={"场景":scene,"方法":name,"视角数":len(ss)}
            row.update({cn:mean(ss,f) for cn,f in metric_fields.items()}); per_scene.append(row)
        sheet(scene)
    write(OUT/"逐场景消融.csv",per_scene)

    geo=[]
    for scene in SCENES:
        real=filtered_xyz(BASE/"reconstruction"/"scenes"/scene/"real8"/"vggt_reconstruction.pt")
        branches={
          "A_原始MVGen+冻结VGGT":BASE/"reconstruction"/"scenes"/scene/"mvgen_1real7gen"/"vggt_reconstruction.pt",
          "B_3D适配MVGen+冻结VGGT":ROOT/"adapted_vggt_test5"/"scenes"/scene/"mvgen_1real7gen"/"vggt_reconstruction.pt",
          "C_3D适配MVGen+微调VGGT末端头":ROOT/"joint_tuned_vggt_test5"/"scenes"/scene/"mvgen_1real7gen"/"vggt_reconstruction.pt"}
        for name,path in branches.items():
            pred=filtered_xyz(path); g=rp.compare_pointclouds(real,pred,50000,23,"real8",name)
            geo.append({"场景":scene,"方法":name,"点数":len(pred),
                "归一化Chamfer_L1":g["symmetric_chamfer_l1_normalized"],
                "归一化Chamfer_L2":g["symmetric_chamfer_l2_normalized"],
                "Hausdorff_P95":g["hausdorff_p95_normalized"],
                "Fscore_0.005":g["f_scores_at_normalized_threshold"]["0.005"]["f_score"],
                "Fscore_0.010":g["f_scores_at_normalized_threshold"]["0.010"]["f_score"],
                "Fscore_0.020":g["f_scores_at_normalized_threshold"]["0.020"]["f_score"]})
    write(OUT/"逐场景几何消融.csv",geo)
    geo_summary=[]
    for name in METHODS:
        rr=[r for r in geo if r["方法"]==name]
        geo_summary.append({"方法":name,"场景数":5,**{k:float(np.mean([float(x[k]) for x in rr]))
            for k in ("点数","归一化Chamfer_L1","归一化Chamfer_L2","Hausdorff_P95","Fscore_0.005","Fscore_0.010","Fscore_0.020")}})
    write(OUT/"几何消融汇总.csv",geo_summary)

    gen_train=json.loads((ROOT/"adapter_4train_2val"/"training_summary.json").read_text())
    vggt_train=json.loads((ROOT/"vggt_terminal_heads_24step"/"training_summary.json").read_text())
    protocol={"split":{"train":gen_train["train_scenes"],"validation":gen_train["validation_scenes"],"test":SCENES,
                       "all_disjoint":True},"phase_A":gen_train,"phase_B":vggt_train,
      "fixed":{"MVGen_steps":50,"CFG":2,"seed":23,"views":"1 real + 7 generated","VGGT_canvas":"518x518 pad",
               "evaluation":"remove virtual padding, remap intrinsics, render at paired native size"}}
    (OUT/"实验设置与完整结果.json").write_text(json.dumps({"protocol":protocol,"ablation":summary,"geometry":geo_summary},ensure_ascii=False,indent=2)+"\n")
    for src,name in [(ROOT/"adapter_4train_2val"/"training_summary.json","阶段A_训练摘要.json"),
                     (ROOT/"vggt_terminal_heads_24step"/"training_summary.json","阶段B_训练摘要.json")]: shutil.copy2(src,OUT/name)

    a,b,c=summary
    ga,gb,gc=geo_summary
    report=f"""# MVGenMaster–VGGT-G 小样本联合微调可行性报告

## 结论先行

本次实验验证了**训练链路和显存可行，但当前小样本朴素联合方案不具备泛化有效性**。不建议立即扩大为全模型联合微调。

- 阶段A只训练MVGenMaster的3D-prior `add_conv_in` 适配器（{gen_train['trainable_parameters']:,}参数，64步），验证扩散损失下降{abs(gen_train['validation_relative_change'])*100:.2f}%，但5个完全留出测试场景的生成图指标反而下降：PSNR {b['生成图_PSNR']-a['生成图_PSNR']:+.3f} dB、SSIM {b['生成图_SSIM']-a['生成图_SSIM']:+.4f}、LPIPS {b['生成图_LPIPS']-a['生成图_LPIPS']:+.4f}（LPIPS越低越好）。说明“扩散噪声损失下降”不能替代真实视角/几何一致性目标。
- 阶段A后保持VGGT冻结，渲染对GT有轻微但不显著的变化：PSNR {b['渲染对GT_PSNR']-a['渲染对GT_PSNR']:+.3f} dB、SSIM {b['渲染对GT_SSIM']-a['渲染对GT_SSIM']:+.4f}、LPIPS {b['渲染对GT_LPIPS']-a['渲染对GT_LPIPS']:+.4f}；不足以证明重影问题得到解决。
- 阶段B再只训练VGGT-G末端83,150个高斯参数头（24步）。在2个验证场景上PSNR +{vggt_train['validation_psnr_delta']:.3f} dB、SSIM +{vggt_train['validation_ssim_delta']:.4f}，但5个额外测试场景上相对阶段B冻结版本变为PSNR {c['渲染对GT_PSNR']-b['渲染对GT_PSNR']:+.3f} dB、SSIM {c['渲染对GT_SSIM']-b['渲染对GT_SSIM']:+.4f}、LPIPS {c['渲染对GT_LPIPS']-b['渲染对GT_LPIPS']:+.4f}，发生明显过拟合。

因此，联合微调方向不是完全不可行，而是**必须把优化目标从单纯扩散损失/重建头拟合，改成显式的跨视角深度、重投影、可见性和真实图回放约束**。

## 实验设置

- 数据严格分离：4训练场景、2验证场景、5额外测试场景，场景互不重叠。
- 输入：1张真实源图 + 7张生成图；MVGenMaster 50步、CFG=2、seed=23。
- 阶段A：冻结VAE与UNet骨干，只更新3D先验注入适配器；训练64步，学习率8e-6。
- 阶段B：冻结VGGT骨干、相机、深度和特征层，只更新xyz/rotation/scale/opacity/color末端头；训练24步，学习率5e-5。
- 所有测试渲染均移除VGGT的518×518虚拟padding，重新映射内参，并在对应GT原始宽高上直接光栅化；不是先渲染再拉伸后计算。
- 点云过滤：有限值、正尺度、opacity≥0.10、去除scale最大0.1%异常值；不设点数上限。

## 三阶段消融（5场景、35个生成视角）

|方法|生成PSNR↑|生成SSIM↑|生成LPIPS↓|渲染对GT PSNR↑|渲染对GT SSIM↑|渲染对GT LPIPS↓|
|---|---:|---:|---:|---:|---:|---:|
|A 原始MVGen + 冻结VGGT|{a['生成图_PSNR']:.3f}|{a['生成图_SSIM']:.4f}|{a['生成图_LPIPS']:.4f}|{a['渲染对GT_PSNR']:.3f}|{a['渲染对GT_SSIM']:.4f}|{a['渲染对GT_LPIPS']:.4f}|
|B 3D适配MVGen + 冻结VGGT|{b['生成图_PSNR']:.3f}|{b['生成图_SSIM']:.4f}|{b['生成图_LPIPS']:.4f}|{b['渲染对GT_PSNR']:.3f}|{b['渲染对GT_SSIM']:.4f}|{b['渲染对GT_LPIPS']:.4f}|
|C 3D适配MVGen + 微调VGGT头|{c['生成图_PSNR']:.3f}|{c['生成图_SSIM']:.4f}|{c['生成图_LPIPS']:.4f}|{c['渲染对GT_PSNR']:.3f}|{c['渲染对GT_SSIM']:.4f}|{c['渲染对GT_LPIPS']:.4f}|

## 几何结果

|方法|Chamfer-L1↓|F-score@0.010↑|点数均值|
|---|---:|---:|---:|
|A|{ga['归一化Chamfer_L1']:.5f}|{ga['Fscore_0.010']:.4f}|{ga['点数']:.0f}|
|B|{gb['归一化Chamfer_L1']:.5f}|{gb['Fscore_0.010']:.4f}|{gb['点数']:.0f}|
|C|{gc['归一化Chamfer_L1']:.5f}|{gc['Fscore_0.010']:.4f}|{gc['点数']:.0f}|

Chamfer与F-score给出不同方向时，意味着局部近邻覆盖可能增加，但远端错误结构/漂浮高斯同时增多；不能只看一个F-score就判定几何改善。

## 推荐方案

最合适的是“**冻结主干、分阶段、带VGGT一致性教师的生成器适配**”，而不是同时解冻两个大模型：

1. 首先冻结VGGT全部参数，只训练MVGenMaster的3D注入适配器或LoRA。损失必须同时包含扩散损失、生成目标图对GT的感知损失、VGGT深度重投影一致性、跨视角可见性/遮挡一致性；用VGGT作为固定几何教师。
2. 只有当生成图在类别留出验证集上同时满足“图像指标不下降、深度冲突率下降”后，才进入VGGT阶段。
3. VGGT阶段先只调opacity与scale头，并保持至少50%的8真图回放，限制漂浮高斯；不应像本轮一样同时调整xyz/rotation/color全部末端头。
4. 采用交替训练而非同一步端到端反传：G阶段10–20步，R阶段2–5步，每轮都在类别留出集早停。这样显存可控，也能明确定位退化来自生成器还是重建器。

工程概率估计（不是本实验直接测得的统计概率）：上述一致性感知交替方案在扩大到至少30训练场景、10类别留出验证场景后，获得稳定改善的概率约60%–75%；只做生成器3D适配约45%–60%；同时全量解冻两个模型约25%–40%，风险最高。

## 是否继续扩大训练

当前不应直接扩到全量联合训练。下一轮最短有效实验应为：30训练 + 10验证，生成器仅LoRA/3D适配器，加入VGGT深度重投影和遮挡损失；VGGT保持冻结。通过门槛建议为：验证集生成LPIPS不升、深度冲突率下降≥10%、渲染对GT PSNR不降。通过后再做opacity/scale两头的短阶段微调。
"""
    (OUT/"中文分析报告.md").write_text(report,encoding="utf-8")
    print(json.dumps({"summary":summary,"geometry":geo_summary,"output":str(OUT)},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
