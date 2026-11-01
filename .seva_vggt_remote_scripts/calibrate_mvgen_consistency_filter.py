#!/usr/bin/env python3
"""Calibrate GT-free multi-view Gaussian consistency filtering on MVGen inputs."""

from __future__ import annotations

import csv, json, math, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

MODEL_CANVAS, PATCH = 518, 14
ROOT = Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
MANIFESTS = Path("/root/seva_vggt/config/test30_mvgen")
OUTPUT = ROOT / "mvgen_consistency_calibration"
SCENES = ["chair_620_101264_203062", "laptop_373_41558_82973",
          "bicycle_62_4318_10726", "car_336_34863_64261", "bottle_601_93006_187080"]
MODES = [
    "unpad",
    "scale_p999",
    "target_front_noscale025",
    "target_front_soft050",
    "target_front_soft025",
    "gaussian_consensus",
    "view_gaussian_consensus",
    "hybrid_target_consensus",
]


def geom(w, h):
    if w >= h: rw, rh = MODEL_CANVAS, round(h*MODEL_CANVAS/w/PATCH)*PATCH
    else: rh, rw = MODEL_CANVAS, round(w*MODEL_CANVAS/h/PATCH)*PATCH
    pw, ph = MODEL_CANVAS-rw, MODEL_CANVAS-rh
    return {"w":w,"h":h,"rw":int(rw),"rh":int(rh),"pl":int(pw//2),"pr":int(pw-pw//2),"pt":int(ph//2),"pb":int(ph-ph//2)}


def native_k(k,g):
    out=k.clone(); sx,sy=g["w"]/g["rw"],g["h"]/g["rh"]
    out[0,0],out[1,1]=k[0,0]*sx,k[1,1]*sy
    out[0,2],out[1,2]=(k[0,2]-g["pl"])*sx,(k[1,2]-g["pt"])*sy
    return out


def tensor(path,device):
    with Image.open(path) as im: a=np.asarray(im.convert("RGB"),dtype=np.float32)/255
    return torch.from_numpy(a).permute(2,0,1).contiguous().to(device)


def resize(x,h,w):
    return torch.nn.functional.interpolate(x[None],(h,w),mode="bicubic",align_corners=False,antialias=True)[0].clamp(0,1)


def metrics(a,b):
    a,b=a.float().clamp(0,1),b.float().clamp(0,1); mse=float(((a-b)**2).mean())
    aa=a.permute(1,2,0).cpu().numpy(); bb=b.permute(1,2,0).cpu().numpy()
    return {"psnr":-10*math.log10(max(mse,1e-12)),"ssim":float(structural_similarity(aa,bb,channel_axis=2,data_range=1.0)),"mse":mse}


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def viewmat(ext,device):
    v=torch.eye(4,device=device); v[:3,:3]=ext[:3,:3]; v[:3,3]=ext[:3,3]; return v


def raster(rasterization,flat,keep,weights,view,K,h,w,depth=False):
    mode="RGB+ED" if depth else "RGB"
    out,alpha,_=rasterization(means=flat["xyz"][keep],quats=flat["rotation"][keep],scales=flat["scale"][keep],
        opacities=flat["opacity"][keep,0]*weights[keep],colors=flat["color"][keep],viewmats=view[None],Ks=K[None],
        width=w,height=h,near_plane=.01,far_plane=1e10,render_mode=mode)
    return out[0],alpha[0,...,0]


def build_consensus(rasterization,flat,source_ids,ext,intr):
    n=len(source_ids); support=torch.zeros(n,dtype=torch.int16,device=source_ids.device)
    conflict=torch.zeros_like(support); tested=torch.zeros_like(support)
    for slot in range(8):
        target=source_ids==slot; view=viewmat(ext[slot],source_ids.device)
        ones=torch.ones(n,device=source_ids.device)
        out,alpha=raster(rasterization,flat,target,ones,view,intr[slot],MODEL_CANVAS,MODEL_CANVAS,True)
        depth=out[...,3]; cam=flat["xyz"]@view[:3,:3].T+view[:3,3]; z=cam[:,2]
        u=intr[slot,0,0]*cam[:,0]/z.clamp_min(1e-8)+intr[slot,0,2]
        v=intr[slot,1,1]*cam[:,1]/z.clamp_min(1e-8)+intr[slot,1,2]
        inside=(z>.01)&(u>=0)&(u<MODEL_CANVAS)&(v>=0)&(v<MODEL_CANVAS)&(~target)
        px=u.round().long().clamp(0,MODEL_CANVAS-1); py=v.round().long().clamp(0,MODEL_CANVAS-1)
        rd=depth[py,px]; ra=alpha[py,px]; ref=inside&(ra>=.75)&torch.isfinite(rd)&(rd>0)
        rel=(z-rd)/rd.clamp_min(1e-6); agree=ref&(rel.abs()<=.03); front=ref&(rel<-.03)
        tested += (agree|front).to(torch.int16); support += agree.to(torch.int16); conflict += front.to(torch.int16)
        del out,alpha,depth,cam,z,u,v,px,py,rd,ra,rel
    tf=tested.float(); sf=support.float(); cf=conflict.float(); rel=(sf+1)/(tf+2)
    gw=torch.ones(n,device=source_ids.device); bad=(tested>=2)&(conflict>support)
    gw[bad]=(0.15+0.85*rel[bad]).clamp(.15,.70)
    vw=torch.ones(8,device=source_ids.device); view_stats=[]
    for s in range(8):
        m=source_ids==s; t=float(tf[m].sum()); sp=float(sf[m].sum()); cfv=float(cf[m].sum())
        q=sp/max(sp+cfv,1.0); weight=float(np.clip((q-.30)/.50,.25,1.0));
        if s==0: weight=1.0
        vw[s]=weight; view_stats.append({"slot":s,"tested":t,"support":sp,"conflict":cfv,"quality":q,"weight":weight})
    return support,conflict,tested,gw,vw,view_stats


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); sys.path.insert(0,"/root/vggt")
    from gsplat import rasterization
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    device=torch.device("cuda:0"); rows=[]; scene_info=[]
    for scene in SCENES:
        started=time.perf_counter()
        try:
            branch=ROOT/"reconstruction/scenes"/scene/"mvgen_1real7gen"
            rec=torch.load(branch/"vggt_reconstruction.pt",map_location="cpu")
            paths=json.loads((branch/"metrics.json").read_text())["input_paths"]
            manifest=json.loads((MANIFESTS/f"{scene}.json").read_text()); gt_paths=[v["image"] for v in manifest["views"]]
            sizes=[]; valid=torch.zeros((8,MODEL_CANVAS,MODEL_CANVAS),dtype=torch.bool,device=device)
            for i,p in enumerate(paths):
                with Image.open(p) as im: g=geom(*im.size)
                sizes.append(g); valid[i,g["pt"]:MODEL_CANVAS-g["pb"],g["pl"]:MODEL_CANVAS-g["pr"]]=True
            grid={k:v[0].float().reshape(8,MODEL_CANVAS,MODEL_CANVAS,v.shape[-1]).to(device)
                  for k,v in rec["gaussians"].items() if torch.is_tensor(v) and v.ndim>=2}
            flat={k:v[valid] for k,v in grid.items()}; sg=torch.arange(8,device=device)[:,None,None].expand_as(valid); source_ids=sg[valid]
            ext,intr=pose_encoding_to_extri_intri(rec["pose_enc"].float(),(MODEL_CANVAS,MODEL_CANVAS)); ext,intr=ext[0].to(device),intr[0].to(device)
            support,conflict,tested,gw,vw,vstats=build_consensus(rasterization,flat,source_ids,ext,intr)
            scale=flat["scale"].amax(-1); limit=torch.quantile(scale,.999); scale_keep=scale<=limit
            scene_info.append({"scene":scene,"gaussians":len(source_ids),"scale_p999":float(limit),"support_ge2_fraction":float((support>=2).float().mean()),
                "conflict_ge2_fraction":float((conflict>=2).float().mean()),"penalized_fraction":float((gw<1).float().mean()),"view_stats":vstats,
                "seconds_consensus":time.perf_counter()-started})
            for slot in range(8):
                inp=tensor(paths[slot],device); gt=tensor(gt_paths[slot],device)
                gi=sizes[slot]; Kg=native_k(intr[slot],gi); view=viewmat(ext[slot],device)
                cam=flat["xyz"]@view[:3,:3].T+view[:3,3]; z=cam[:,2]
                u=intr[slot,0,0]*cam[:,0]/z.clamp_min(1e-8)+intr[slot,0,2]; v=intr[slot,1,1]*cam[:,1]/z.clamp_min(1e-8)+intr[slot,1,2]
                inside=(z>.01)&(u>=0)&(u<MODEL_CANVAS)&(v>=0)&(v<MODEL_CANVAS)&(source_ids!=slot)
                # Re-render target-source depth on the model canvas for a reversible target-specific conflict weight.
                ones=torch.ones(len(source_ids),device=device); target=source_ids==slot
                dout,dalpha=raster(rasterization,flat,target,ones,view,intr[slot],MODEL_CANVAS,MODEL_CANVAS,True)
                px=u.round().long().clamp(0,MODEL_CANVAS-1); py=v.round().long().clamp(0,MODEL_CANVAS-1)
                rd=dout[...,3][py,px]; ra=dalpha[py,px]; rel=(z-rd)/rd.clamp_min(1e-6)
                target_front=inside&(ra>=.75)&torch.isfinite(rd)&(rel<-.03)
                for mode in MODES:
                    keep=torch.ones(len(source_ids),dtype=torch.bool,device=device); w=torch.ones(len(source_ids),device=device)
                    if mode not in ("unpad", "target_front_noscale025"): keep &= scale_keep
                    if mode in ("gaussian_consensus","view_gaussian_consensus","hybrid_target_consensus"): w*=gw
                    if mode in ("view_gaussian_consensus","hybrid_target_consensus"): w*=vw[source_ids]
                    if mode=="target_front_soft050": w[target_front]*=.50
                    if mode=="target_front_soft025": w[target_front]*=.25
                    if mode=="target_front_noscale025": w[target_front]*=.25
                    if mode=="hybrid_target_consensus": w[target_front]*=.25
                    with torch.inference_mode(): out,alpha=raster(rasterization,flat,keep,w,view,Kg,gi["h"],gi["w"],False)
                    pred=out[...,:3].permute(2,0,1).clamp(0,1); mi=metrics(inp,pred)
                    # Render on the exact GT plane with the same Gaussian weights.
                    with Image.open(gt_paths[slot]) as im: gg=geom(*im.size)
                    Kgt=native_k(intr[slot],gg)
                    with torch.inference_mode(): outgt,_=raster(rasterization,flat,keep,w,view,Kgt,gg["h"],gg["w"],False)
                    predgt=outgt[...,:3].permute(2,0,1).clamp(0,1); mg=metrics(gt,predgt)
                    rows.append({"scene":scene,"slot":slot,"source_type":"real" if slot==0 else "generated","mode":mode,
                        "gaussians_kept":int(keep.sum()),"opacity_weight_mean":float(w[keep].mean()),"target_front_fraction":float(target_front.float().mean()),
                        **{f"render_to_input_{k}":x for k,x in mi.items()},**{f"render_to_gt_{k}":x for k,x in mg.items()}})
                    del out,alpha,pred,outgt,predgt
                del inp,gt,cam,z,u,v,dout,dalpha,px,py,rd,ra,rel
                torch.cuda.empty_cache()
            print(f"COMPLETE {scene} seconds={time.perf_counter()-started:.1f}",flush=True)
            del rec,grid,flat; torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED {scene}: {e!r}\n{traceback.format_exc()}",flush=True)
            raise
    write_csv(OUTPUT/"metrics_per_view.csv",rows)
    summary=[]
    for mode in MODES:
        rr=[r for r in rows if r["mode"]==mode and r["source_type"]=="generated"]
        item={"mode":mode,"scenes":len(SCENES),"generated_views":len(rr)}
        for key in ("render_to_input_psnr","render_to_input_ssim","render_to_input_mse","render_to_gt_psnr","render_to_gt_ssim","render_to_gt_mse","gaussians_kept","opacity_weight_mean"):
            item[key]=float(np.mean([r[key] for r in rr]))
        summary.append(item)
    write_csv(OUTPUT/"summary.csv",summary)
    (OUTPUT/"scene_consistency.json").write_text(json.dumps(scene_info,ensure_ascii=False,indent=2))
    (OUTPUT/"protocol.json").write_text(json.dumps({"scenes":SCENES,"modes":MODES,"filter_uses_gt":False,
        "support":"project each Gaussian into seven other source-specific expected-depth maps; agreement abs relative depth <=3%",
        "conflict":"Gaussian lies >3% in front of a reliable source-specific surface","scale":"remove only top 0.1% max-scale outliers",
        "view_weight":"source0=1; generated-source weights from cross-view support/conflict quality, clipped [0.25,1]"},ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
