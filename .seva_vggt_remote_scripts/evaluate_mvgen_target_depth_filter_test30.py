#!/usr/bin/env python3
"""Held-out validation of reversible target-view depth-conflict filtering for MVGen->VGGT-G."""

from __future__ import annotations

import csv, json, math, sys, time, traceback
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

MODEL_CANVAS = 518
PATCH = 14
ALPHA_THR = 0.75
FRONT_THR = 0.03
CONFLICT_WEIGHT = 0.25
ROOT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
MANIFESTS = Path("/root/seva_vggt/config/test30_mvgen")
OUTPUT = ROOT / "evaluation_mvgen_target_depth_filter"
CALIBRATION = {
    "chair_620_101264_203062", "laptop_373_41558_82973",
    "bicycle_62_4318_10726", "car_336_34863_64261",
    "bottle_601_93006_187080",
}


def geom(w, h):
    if w >= h: rw, rh = MODEL_CANVAS, round(h * MODEL_CANVAS / w / PATCH) * PATCH
    else: rh, rw = MODEL_CANVAS, round(w * MODEL_CANVAS / h / PATCH) * PATCH
    pw, ph = MODEL_CANVAS-rw, MODEL_CANVAS-rh
    return {"w":w,"h":h,"rw":int(rw),"rh":int(rh),"pl":int(pw//2),"pr":int(pw-pw//2),"pt":int(ph//2),"pb":int(ph-ph//2)}


def native_k(k, g):
    out=k.clone(); sx,sy=g["w"]/g["rw"],g["h"]/g["rh"]
    out[0,0],out[1,1]=k[0,0]*sx,k[1,1]*sy
    out[0,2],out[1,2]=(k[0,2]-g["pl"])*sx,(k[1,2]-g["pt"])*sy
    return out


def image_tensor(path, device):
    with Image.open(path) as im: arr=np.asarray(im.convert("RGB"),dtype=np.float32)/255
    return torch.from_numpy(arr).permute(2,0,1).contiguous().to(device)


def to_pil(x):
    arr=(x.detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()*255+0.5).astype(np.uint8)
    return Image.fromarray(arr)


def metric(a,b,perceptual):
    a,b=a.float().clamp(0,1),b.float().clamp(0,1); mse=float(((a-b)**2).mean())
    aa=a.permute(1,2,0).cpu().numpy(); bb=b.permute(1,2,0).cpu().numpy()
    with torch.inference_mode(): ld=float(perceptual(a[None]*2-1,b[None]*2-1).flatten()[0])
    return {"mse":mse,"psnr":-10*math.log10(max(mse,1e-12)),
            "ssim":float(structural_similarity(aa,bb,channel_axis=2,data_range=1.0)),"lpips":ld}


def viewmat(ext, device):
    out=torch.eye(4,device=device); out[:3,:3]=ext[:3,:3]; out[:3,3]=ext[:3,3]; return out


def raster(rasterization, flat, weights, view, K, h, w, depth=False):
    mode="RGB+ED" if depth else "RGB"
    out,alpha,_=rasterization(means=flat["xyz"],quats=flat["rotation"],scales=flat["scale"],
        opacities=flat["opacity"][:,0]*weights,colors=flat["color"],viewmats=view[None],Ks=K[None],
        width=w,height=h,near_plane=.01,far_plane=1e10,render_mode=mode)
    return out[0],alpha[0,...,0]


def target_depth(rasterization, flat, target, view, K, h, w):
    out,alpha,_=rasterization(means=flat["xyz"][target],quats=flat["rotation"][target],
        scales=flat["scale"][target],opacities=flat["opacity"][target,0],colors=flat["color"][target],
        viewmats=view[None],Ks=K[None],width=w,height=h,near_plane=.01,far_plane=1e10,render_mode="RGB+ED")
    return out[0,...,3],alpha[0,...,0]


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def fitted(image,w,h):
    canvas=Image.new("RGB",(w,h),"white"); cp=image.convert("RGB"); cp.thumbnail((w,h),Image.Resampling.LANCZOS)
    canvas.paste(cp,((w-cp.width)//2,(h-cp.height)//2)); return canvas


def make_sheet(scene, visuals, path):
    cw,ch,lw,hh=185,165,90,56
    titles=("Real GT","MVGen input","baseline->GT","filtered->GT","baseline->input","filtered->input")
    sheet=Image.new("RGB",(lw+len(titles)*cw,hh+7*ch),"white"); d=ImageDraw.Draw(sheet)
    d.text((4,4),scene,fill="black")
    for c,t in enumerate(titles): d.text((lw+c*cw+4,30),t,fill="black")
    for row in visuals:
        y=hh+(row["slot"]-1)*ch; d.text((4,y+5),f"view {row['slot']:02d}",fill="black")
        d.text((4,y+23),f"GT dP {row['gt_gain']:+.2f}",fill="black")
        d.text((4,y+41),f"IN dP {row['input_gain']:+.2f}",fill="black")
        for c,key in enumerate(("gt","input","base_gt","proc_gt","base_in","proc_in")):
            sheet.paste(fitted(row[key],cw,ch-8),(lw+c*cw,y))
    sheet.save(path,quality=92)


def aggregate(rows, split):
    rr=[r for r in rows if r["source_type"]=="generated" and (split=="all" or r["split"]==split)]
    result={"split":split,"scenes":len(set(r["scene"] for r in rr)),"views":len(rr)}
    for ref in ("input","gt"):
        for m in ("psnr","ssim","lpips","mse"):
            b=np.asarray([r[f"baseline_{ref}_{m}"] for r in rr]); p=np.asarray([r[f"processed_{ref}_{m}"] for r in rr])
            good=(p>b) if m in ("psnr","ssim") else (p<b)
            result[f"baseline_{ref}_{m}"]=float(b.mean()); result[f"processed_{ref}_{m}"]=float(p.mean())
            result[f"delta_{ref}_{m}"]=float((p-b).mean()); result[f"improved_{ref}_{m}"]=int(good.sum())
    result["front_conflict_fraction_non_target"]=float(np.mean([r["front_conflict_fraction_non_target"] for r in rr]))
    result["opacity_weight_mean"]=float(np.mean([r["opacity_weight_mean"] for r in rr]))
    return result


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); (OUTPUT/"scenes").mkdir(exist_ok=True); (OUTPUT/"comparisons").mkdir(exist_ok=True)
    sys.path.insert(0,"/root/vggt")
    from gsplat import rasterization
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    scenes=json.loads((MANIFESTS/"selection_protocol.json").read_text())["scenes"]
    device=torch.device("cuda:0"); perceptual=lpips.LPIPS(net="alex").eval().to(device)
    status_path=OUTPUT/"status.json"; status=json.loads(status_path.read_text()) if status_path.exists() else {"completed":[],"failed":{}}
    for scene in scenes:
        out_scene=OUTPUT/"scenes"/scene; metrics_path=out_scene/"metrics.json"
        if scene in status["completed"] and metrics_path.exists(): print("SKIP",scene,flush=True); continue
        started=time.perf_counter()
        try:
            out_scene.mkdir(parents=True,exist_ok=True)
            branch=ROOT/"reconstruction/scenes"/scene/"mvgen_1real7gen"
            rec=torch.load(branch/"vggt_reconstruction.pt",map_location="cpu")
            paths=json.loads((branch/"metrics.json").read_text())["input_paths"]
            gt_paths=[v["image"] for v in json.loads((MANIFESTS/f"{scene}.json").read_text())["views"]]
            geoms=[]; valid=torch.zeros((8,MODEL_CANVAS,MODEL_CANVAS),dtype=torch.bool,device=device)
            for i,p in enumerate(paths):
                with Image.open(p) as im: g=geom(*im.size)
                geoms.append(g); valid[i,g["pt"]:MODEL_CANVAS-g["pb"],g["pl"]:MODEL_CANVAS-g["pr"]]=True
            grid={k:v[0].float().reshape(8,MODEL_CANVAS,MODEL_CANVAS,v.shape[-1]).to(device)
                  for k,v in rec["gaussians"].items() if torch.is_tensor(v) and v.ndim>=2}
            flat={k:v[valid] for k,v in grid.items()}; source_ids=torch.arange(8,device=device)[:,None,None].expand_as(valid)[valid]
            ext,intr=pose_encoding_to_extri_intri(rec["pose_enc"].float(),(MODEL_CANVAS,MODEL_CANVAS)); ext,intr=ext[0].to(device),intr[0].to(device)
            scene_rows=[]; visual=[]
            for slot in range(8):
                inp=image_tensor(paths[slot],device); gt=image_tensor(gt_paths[slot],device); gi=geoms[slot]
                with Image.open(gt_paths[slot]) as im: gg=geom(*im.size)
                Ki=native_k(intr[slot],gi); Kg=native_k(intr[slot],gg); view=viewmat(ext[slot],device); target=source_ids==slot
                with torch.inference_mode():
                    rd,ra=target_depth(rasterization,flat,target,view,Ki,gi["h"],gi["w"])
                    cam=flat["xyz"]@view[:3,:3].T+view[:3,3]; z=cam[:,2]
                    u=Ki[0,0]*cam[:,0]/z.clamp_min(1e-8)+Ki[0,2]; v=Ki[1,1]*cam[:,1]/z.clamp_min(1e-8)+Ki[1,2]
                    inside=(z>.01)&(u>=0)&(u<gi["w"])&(v>=0)&(v<gi["h"])&(~target)
                    px=u.round().long().clamp(0,gi["w"]-1); py=v.round().long().clamp(0,gi["h"]-1)
                    sd,sa=rd[py,px],ra[py,px]; rel=(z-sd)/sd.clamp_min(1e-6)
                    conflict=inside&(sa>=ALPHA_THR)&torch.isfinite(sd)&(rel < -FRONT_THR)
                    one=torch.ones(len(source_ids),device=device); weights=one.clone(); weights[conflict]=CONFLICT_WEIGHT
                    bi,_=raster(rasterization,flat,one,view,Ki,gi["h"],gi["w"]); pi,_=raster(rasterization,flat,weights,view,Ki,gi["h"],gi["w"])
                    bg,_=raster(rasterization,flat,one,view,Kg,gg["h"],gg["w"]); pg,_=raster(rasterization,flat,weights,view,Kg,gg["h"],gg["w"])
                b_in,p_in=bi[...,:3].permute(2,0,1).clamp(0,1),pi[...,:3].permute(2,0,1).clamp(0,1)
                b_gt,p_gt=bg[...,:3].permute(2,0,1).clamp(0,1),pg[...,:3].permute(2,0,1).clamp(0,1)
                mbi,mpi,mgb,mgp=metric(inp,b_in,perceptual),metric(inp,p_in,perceptual),metric(gt,b_gt,perceptual),metric(gt,p_gt,perceptual)
                row={"scene":scene,"category":scene.split("_")[0],"split":"calibration" if scene in CALIBRATION else "holdout",
                     "slot":slot,"source_type":"real" if slot==0 else "generated","input_width":gi["w"],"input_height":gi["h"],
                     "gt_width":gg["w"],"gt_height":gg["h"],"gaussians":len(source_ids),"front_conflict_gaussians":int(conflict.sum()),
                     "front_conflict_fraction_non_target":float(conflict.sum()/(~target).sum()),"opacity_weight_mean":float(weights.mean())}
                for prefix,data in (("baseline_input",mbi),("processed_input",mpi),("baseline_gt",mgb),("processed_gt",mgp)):
                    row.update({f"{prefix}_{k}":x for k,x in data.items()})
                scene_rows.append(row)
                if slot>0:
                    visual.append({"slot":slot,"gt":to_pil(gt),"input":to_pil(inp),"base_gt":to_pil(b_gt),"proc_gt":to_pil(p_gt),
                                   "base_in":to_pil(b_in),"proc_in":to_pil(p_in),
                                   "gt_gain":mgp["psnr"]-mgb["psnr"],"input_gain":mpi["psnr"]-mbi["psnr"]})
                del inp,gt,rd,ra,cam,z,u,v,px,py,sd,sa,rel,bi,pi,bg,pg,b_in,p_in,b_gt,p_gt
                torch.cuda.empty_cache()
            metrics_path.write_text(json.dumps(scene_rows,ensure_ascii=False,indent=2),encoding="utf-8")
            make_sheet(scene,visual,OUTPUT/"comparisons"/f"{scene}.jpg")
            status["completed"].append(scene); status["failed"].pop(scene,None)
            print(f"COMPLETE {scene} {time.perf_counter()-started:.1f}s",flush=True)
            del rec,grid,flat; torch.cuda.empty_cache()
        except Exception as e:
            status["failed"][scene]={"error":repr(e),"traceback":traceback.format_exc()}; print("FAILED",scene,repr(e),flush=True)
        status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[]
    for scene in scenes:
        p=OUTPUT/"scenes"/scene/"metrics.json"
        if p.exists(): rows.extend(json.loads(p.read_text()))
    write_csv(OUTPUT/"metrics_per_view.csv",rows)
    scene_rows=[]
    for scene in scenes:
        rr=[r for r in rows if r["scene"]==scene and r["source_type"]=="generated"]
        if not rr: continue
        item={"scene":scene,"category":scene.split("_")[0],"split":rr[0]["split"],"views":len(rr)}
        for ref in ("input","gt"):
            for m in ("psnr","ssim","lpips","mse"):
                for p in ("baseline","processed"): item[f"{p}_{ref}_{m}"]=float(np.mean([r[f"{p}_{ref}_{m}"] for r in rr]))
                item[f"delta_{ref}_{m}"]=item[f"processed_{ref}_{m}"]-item[f"baseline_{ref}_{m}"]
        item["front_conflict_fraction_non_target"]=float(np.mean([r["front_conflict_fraction_non_target"] for r in rr])); scene_rows.append(item)
    write_csv(OUTPUT/"metrics_per_scene.csv",scene_rows)
    summary={s:aggregate(rows,s) for s in ("calibration","holdout","all")}
    summary["protocol"]={"uses_gt_for_filter":False,"padding_gaussians":"removed before filtering","hard_delete":False,
        "reference":"target source-view expected depth","alpha_threshold":ALPHA_THR,"front_relative_threshold":FRONT_THR,
        "conflict_opacity_weight":CONFLICT_WEIGHT,"calibration_scenes":sorted(CALIBRATION)}
    summary["failed"]=status["failed"]
    (OUTPUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    h=summary["holdout"]
    report=f"""# MVGen→VGGT-G 一致性与过滤可行性验证\n\n## 方案\n\n先硬移除预处理padding位置对应的Gaussian；不按scale硬删真实内容。对每个目标视角，仅用该目标来源Gaussian渲染可信深度。在alpha≥{ALPHA_THR}处，若其他来源Gaussian位于可信表面前方超过{FRONT_THR*100:.0f}%，只在该目标视角将其opacity乘{CONFLICT_WEIGHT}。过滤不使用GT，GT仅用于评测，原重建文件不修改。\n\n参数仅由5个校准场景确定；以下核心结论取剩余{h['scenes']}个独立场景、{h['views']}个生成视角。\n\n| 对比域 | 指标 | 基线 | 后处理 | 变化 | 改善视角 |\n|---|---|---:|---:|---:|---:|\n| 生成输入 | PSNR↑ | {h['baseline_input_psnr']:.4f} | {h['processed_input_psnr']:.4f} | {h['delta_input_psnr']:+.4f} | {h['improved_input_psnr']}/{h['views']} |\n| 生成输入 | SSIM↑ | {h['baseline_input_ssim']:.4f} | {h['processed_input_ssim']:.4f} | {h['delta_input_ssim']:+.4f} | {h['improved_input_ssim']}/{h['views']} |\n| 生成输入 | LPIPS↓ | {h['baseline_input_lpips']:.4f} | {h['processed_input_lpips']:.4f} | {h['delta_input_lpips']:+.4f} | {h['improved_input_lpips']}/{h['views']} |\n| 真实GT | PSNR↑ | {h['baseline_gt_psnr']:.4f} | {h['processed_gt_psnr']:.4f} | {h['delta_gt_psnr']:+.4f} | {h['improved_gt_psnr']}/{h['views']} |\n| 真实GT | SSIM↑ | {h['baseline_gt_ssim']:.4f} | {h['processed_gt_ssim']:.4f} | {h['delta_gt_ssim']:+.4f} | {h['improved_gt_ssim']}/{h['views']} |\n| 真实GT | LPIPS↓ | {h['baseline_gt_lpips']:.4f} | {h['processed_gt_lpips']:.4f} | {h['delta_gt_lpips']:+.4f} | {h['improved_gt_lpips']}/{h['views']} |\n\n非目标Gaussian被判定为当前视角前景冲突的平均比例为{h['front_conflict_fraction_non_target']*100:.2f}%。逐场景结果见CSV和30张对比图。\n"""
    (OUTPUT/"一致性过滤验证报告.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if status["failed"] or len(rows)!=240: raise SystemExit(1)


if __name__=="__main__": main()
