#!/usr/bin/env python3
"""Paired four-way ablation: generator adapter x target-view soft filter."""

import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT=Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
OLD=ROOT/"evaluation_mvgen_target_depth_filter"/"metrics_per_view.csv"
NEW=ROOT/"mvgen_geometry_adapter_filter_test4"/"metrics_per_view.csv"
OUT=ROOT/"mvgen_geometry_adapter_filter_test4"
SCENES={"chair_620_101264_203062","laptop_373_41558_82973","plant_598_92218_184789","toybus_121_14153_28986"}

def read(path):
    with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))

def stats(a,b,higher,seed=23):
    a=np.asarray(a,float);b=np.asarray(b,float);d=b-a;rng=np.random.default_rng(seed)
    boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(10000)])
    try:p=float(wilcoxon(d).pvalue)
    except ValueError:p=1.0
    return {"from_mean":float(a.mean()),"to_mean":float(b.mean()),"delta_mean":float(d.mean()),
      "delta_median":float(np.median(d)),"ci95":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],
      "wilcoxon_p":p,"improved":int(((d>0) if higher else (d<0)).sum()),"views":len(d)}

def main():
    old={(r["scene"],int(r["slot"])):r for r in read(OLD) if r["scene"] in SCENES and r["source_type"]=="generated"}
    new={(r["scene"],int(r["slot"])):r for r in read(NEW)};keys=sorted(set(old)&set(new))
    methods={
      "original_pad_removed":(old,"baseline"),"original_pad_removed_filtered":(old,"processed"),
      "geometry_pad_removed":(new,"baseline"),"geometry_pad_removed_filtered":(new,"processed")}
    means={}
    for name,(table,prefix) in methods.items():
        means[name]={f"{ref}_{m}":float(np.mean([float(table[k][f"{prefix}_{ref}_{m}"]) for k in keys]))
          for ref in ("gt","input") for m in ("psnr","ssim","lpips")}
    comparisons={}
    pairs={
      "adapter_without_filter":("original_pad_removed","geometry_pad_removed"),
      "adapter_with_same_filter":("original_pad_removed_filtered","geometry_pad_removed_filtered"),
      "filter_on_geometry_adapter":("geometry_pad_removed","geometry_pad_removed_filtered")}
    for label,(left,right) in pairs.items():
      comparisons[label]={}
      lt,lp=methods[left];rt,rp=methods[right]
      for ref in ("gt","input"):
       for m in ("psnr","ssim","lpips"):
        comparisons[label][f"{ref}_{m}"]=stats([float(lt[k][f"{lp}_{ref}_{m}"]) for k in keys],
          [float(rt[k][f"{rp}_{ref}_{m}"]) for k in keys],m!="lpips")
    payload={"protocol":{"scenes":sorted(SCENES),"views":len(keys),"paired":True,"same_filter_parameters":True},
             "method_means":means,"paired_comparisons":comparisons}
    (OUT/"four_way_ablation.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    rows=[]
    for name,v in means.items():rows.append({"method":name,**v})
    with (OUT/"four_way_ablation.csv").open("w",newline="",encoding="utf-8-sig") as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    x=comparisons["adapter_with_same_filter"]
    report="# 几何适配器 × 软过滤四组消融\n\n同一4个未见类别、28个视角；所有组先移除padding Gaussian，过滤参数完全相同。\n\n"
    report+="|方法|GT PSNR↑|GT SSIM↑|GT LPIPS↓|输入 PSNR↑|\n|---|---:|---:|---:|---:|\n"
    for name,v in means.items():report+=f"|{name}|{v['gt_psnr']:.5f}|{v['gt_ssim']:.5f}|{v['gt_lpips']:.5f}|{v['input_psnr']:.5f}|\n"
    report+="\n## 在相同过滤下，几何适配器相对原模型\n\n"
    for m,label in (("gt_psnr","GT PSNR"),("gt_ssim","GT SSIM"),("gt_lpips","GT LPIPS"),("input_psnr","输入PSNR")):
      v=x[m];report+=f"- {label}: {v['delta_mean']:+.6f}，95% CI [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]，改善 {v['improved']}/{v['views']}，p={v['wilcoxon_p']:.4g}\n"
    (OUT/"四组消融报告.md").write_text(report,encoding="utf-8")
    print(json.dumps(payload,indent=2,ensure_ascii=False))

if __name__=="__main__":main()
