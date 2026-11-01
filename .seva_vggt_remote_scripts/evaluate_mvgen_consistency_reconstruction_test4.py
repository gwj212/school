#!/usr/bin/env python3
"""Native-size paired downstream evaluation of original/adapted MVGen through frozen VGGT-G."""

from __future__ import annotations

import csv
import argparse
import json
import math
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import wilcoxon

SCENES = [
    "chair_620_101264_203062", "laptop_373_41558_82973",
    "plant_598_92218_184789", "toybus_121_14153_28986",
]
ROOT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
MANIFESTS = Path("/root/seva_vggt/config/test30_mvgen")
ADAPTED_GEN = ROOT / "mvgen_consistency_adapter_test4"
ADAPTED_RECON = ROOT / "mvgen_consistency_adapter_reconstruction_test4" / "scenes"
OUTPUT = ROOT / "mvgen_consistency_adapter_downstream_test4"


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def fitted(image, width, height):
    canvas = Image.new("RGB", (width, height), "white")
    copy = image.copy(); copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(copy, ((width-copy.width)//2, (height-copy.height)//2)); return canvas


def sheet(scene, visual, path):
    cw, ch, lw, hh = 215, 170, 88, 55
    titles = ("GT", "original input", "original render", "adapted input", "adapted render")
    out = Image.new("RGB", (lw+5*cw, hh+7*ch), "white"); draw = ImageDraw.Draw(out)
    draw.text((5,5), scene, fill="black")
    for c,title in enumerate(titles): draw.text((lw+c*cw+5,30), title, fill="black")
    for i,item in enumerate(visual):
        y=hh+i*ch; draw.multiline_text((4,y+5), f"v{item['slot']:02d}\nGT ΔP\n{item['delta_gt_psnr']:+.2f}", fill="black")
        for c,image in enumerate(item["images"]): out.paste(fitted(image,cw,ch-5),(lw+c*cw,y))
    out.save(path, quality=95)


def filtered_xyz(reconstruction):
    g=reconstruction["gaussians"]
    xyz=g["xyz"][0].reshape(-1,3).float().numpy(); op=g["opacity"][0].reshape(-1).float().numpy()
    scale=g["scale"][0].reshape(-1,3).float().numpy()
    keep=np.isfinite(xyz).all(1)&np.isfinite(op)&np.isfinite(scale).all(1)&(scale>0).all(1)&(op>=.1)
    sm=scale.max(1); lim=float(np.percentile(sm[keep],99.9)); return xyz[keep&(sm<=lim)]


def bootstrap_ci(values, seed=23):
    rng=np.random.default_rng(seed); values=np.asarray(values,float)
    sims=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(10000)])
    return [float(np.percentile(sims,2.5)),float(np.percentile(sims,97.5))]


def paired_summary(rows, base_key, adapted_key, higher=True):
    a=np.array([r[base_key] for r in rows],float); b=np.array([r[adapted_key] for r in rows],float); d=b-a
    try: p=float(wilcoxon(d).pvalue)
    except ValueError: p=1.0
    return {"original_mean":float(a.mean()),"adapted_mean":float(b.mean()),"delta_mean":float(d.mean()),
            "delta_median":float(np.median(d)),"delta_ci95_bootstrap":bootstrap_ci(d),"wilcoxon_p":p,
            "improved_views":int(((d>0) if higher else (d<0)).sum()),"views":len(rows)}


def main():
    global ADAPTED_GEN, ADAPTED_RECON, OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--adapted-generation",type=Path,default=ADAPTED_GEN)
    parser.add_argument("--adapted-reconstruction",type=Path,default=ADAPTED_RECON)
    parser.add_argument("--adapted-branch",default="mvgen_consistency_1real7gen")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    ADAPTED_GEN=args.adapted_generation; ADAPTED_RECON=args.adapted_reconstruction; OUTPUT=args.output
    OUTPUT.mkdir(parents=True,exist_ok=True); (OUTPUT/"comparisons").mkdir(exist_ok=True); (OUTPUT/"renders").mkdir(exist_ok=True)
    sys.path[:0]=["/root/seva_vggt/bridge","/root/vggt"]
    import evaluate_render_test30_native as native
    import run_pipeline as rp
    from vggt.utils.gs_optimizer import render_gaussians
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    device=torch.device("cuda:0"); perceptual=lpips.LPIPS(net="alex").eval().to(device)
    rows=[]; geometry=[]
    for scene in SCENES:
        manifest=json.loads((MANIFESTS/f"{scene}.json").read_text()); gt_paths=[v["image"] for v in manifest["views"]]
        branches={
          "original": ROOT/"reconstruction"/"scenes"/scene/"mvgen_1real7gen"/"vggt_reconstruction.pt",
          "adapted": ADAPTED_RECON/scene/args.adapted_branch/"vggt_reconstruction.pt",
        }
        recs={k:torch.load(v,map_location="cpu") for k,v in branches.items()}
        cameras={}; gaussians={}
        for name,rec in recs.items():
            ex,ins=pose_encoding_to_extri_intri(rec["pose_enc"].float(),(native.MODEL_CANVAS,native.MODEL_CANVAS))
            cameras[name]=(ex[0].to(device),ins[0].to(device)); gaussians[name]=native.gaussian_dict(rec,device)
        real_rec=torch.load(ROOT/"reconstruction"/"scenes"/scene/"real8"/"vggt_reconstruction.pt",map_location="cpu")
        ref_xyz=filtered_xyz(real_rec)
        for name in ("original","adapted"):
            comp=rp.compare_pointclouds(ref_xyz,filtered_xyz(recs[name]),sample_count=50000,seed=23,
                                        reference_name="real8",prediction_name=name)
            geometry.append({"scene":scene,"method":name,
              "normalized_chamfer_l1":comp["symmetric_chamfer_l1_normalized"],
              "normalized_chamfer_l2":comp["symmetric_chamfer_l2_normalized"],
              "fscore_0p010":comp["f_scores_at_normalized_threshold"]["0.010"]["f_score"]})
        visual=[]
        for slot in range(1,8):
            gt_pil,gt=native.open_rgb(gt_paths[slot]); w,h=gt_pil.size; geom=native.preprocess_geometry(w,h)
            in_paths={"original":ROOT/"mvgen"/scene/"images"/f"view_{slot:02d}_generated.png",
                      "adapted":ADAPTED_GEN/scene/"images"/f"view_{slot:02d}_generated.png"}
            input_pil={}; input_t={}; render={}; values={}
            for name in ("original","adapted"):
                input_pil[name],it=native.open_rgb(in_paths[name]); input_t[name]=native.resize_tensor(it.to(device),h,w)
                ex,ins=cameras[name]
                render[name]=native.render_native(render_gaussians,gaussians[name],ex[slot],ins[slot],geom)
                outdir=OUTPUT/"renders"/scene/name; outdir.mkdir(parents=True,exist_ok=True)
                native.tensor_to_pil(render[name]).save(outdir/f"view_{slot:02d}_render.png")
                values[name]={}
                for domain,ref in (("gt",gt.to(device)),("input",input_t[name])):
                    met=native.image_metrics(ref,render[name],perceptual)
                    values[name][domain]=met
            row={"scene":scene,"category":scene.split("_")[0],"slot":slot,"width":w,"height":h,
                 "gt_path":gt_paths[slot],"original_input_path":str(in_paths["original"]),"adapted_input_path":str(in_paths["adapted"])}
            for name in ("original","adapted"):
                for domain in ("gt","input"):
                    for metric in ("psnr","ssim","lpips","mse"):
                        row[f"{name}_render_to_{domain}_{metric}"]=float(values[name][domain][metric])
            for domain in ("gt","input"):
                for metric in ("psnr","ssim","lpips","mse"):
                    row[f"delta_render_to_{domain}_{metric}"]=row[f"adapted_render_to_{domain}_{metric}"]-row[f"original_render_to_{domain}_{metric}"]
            rows.append(row)
            visual.append({"slot":slot,"delta_gt_psnr":row["delta_render_to_gt_psnr"],"images":(
                gt_pil,native.tensor_to_pil(input_t["original"]),native.tensor_to_pil(render["original"]),
                native.tensor_to_pil(input_t["adapted"]),native.tensor_to_pil(render["adapted"]))})
        sheet(scene,visual,OUTPUT/"comparisons"/f"{scene}.jpg")
        del recs,gaussians,cameras,real_rec; torch.cuda.empty_cache()
    write_csv(OUTPUT/"metrics_per_view.csv",rows); write_csv(OUTPUT/"geometry_vs_real8.csv",geometry)
    summary={"protocol":{"scenes":SCENES,"views":len(rows),"VGGT":"frozen; same checkpoint; raw Gaussian render; no post-filter",
                         "render":"native target size via inverse pad/crop camera remap"},"metrics":{},"geometry":{}}
    for domain in ("gt","input"):
        for metric in ("psnr","ssim","lpips"):
            summary["metrics"][f"render_to_{domain}_{metric}"]=paired_summary(
                rows,f"original_render_to_{domain}_{metric}",f"adapted_render_to_{domain}_{metric}",metric!="lpips")
    for metric in ("normalized_chamfer_l1","normalized_chamfer_l2","fscore_0p010"):
        base=[];adapt=[]
        for scene in SCENES:
            base.append(next(r[metric] for r in geometry if r["scene"]==scene and r["method"]=="original"))
            adapt.append(next(r[metric] for r in geometry if r["scene"]==scene and r["method"]=="adapted"))
        d=np.array(adapt)-np.array(base); summary["geometry"][metric]={"original_mean":float(np.mean(base)),"adapted_mean":float(np.mean(adapt)),"delta_mean":float(d.mean())}
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    m=summary["metrics"]
    report="# 一致性适配器下游 VGGT-G 验证\n\n冻结同一VGGT-G，分别重建原始MVGen与适配后MVGen输入；原生目标尺寸渲染，不使用后处理过滤。\n\n"
    report+="|域/指标|原始|适配后|变化|改善视角|95% CI|p|\n|---|---:|---:|---:|---:|---:|---:|\n"
    for key,label in (("render_to_gt_psnr","渲染-GT PSNR↑"),("render_to_gt_ssim","渲染-GT SSIM↑"),("render_to_gt_lpips","渲染-GT LPIPS↓"),("render_to_input_psnr","渲染-输入 PSNR↑")):
        x=m[key];ci=x["delta_ci95_bootstrap"];report+=f"|{label}|{x['original_mean']:.5f}|{x['adapted_mean']:.5f}|{x['delta_mean']:+.5f}|{x['improved_views']}/{x['views']}|[{ci[0]:+.5f},{ci[1]:+.5f}]|{x['wilcoxon_p']:.4g}|\n"
    report+="\n几何表为与real8重建点云对齐后的归一化Chamfer/F-score；详见CSV。\n"
    (OUTPUT/"report_zh.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
