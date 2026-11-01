#!/usr/bin/env python3
"""Apply the calibrated target-view soft filter to geometry-adapted reconstructions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image

SCENES=["chair_620_101264_203062","laptop_373_41558_82973","plant_598_92218_184789","toybus_121_14153_28986"]
ROOT=Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
RECON=ROOT/"mvgen_geometry_adapter_reconstruction_test4"/"scenes"
OUT=ROOT/"mvgen_geometry_adapter_filter_test4"
MANIFESTS=Path("/root/seva_vggt/config/test30_mvgen")
BRANCH="mvgen_geometry_1real7gen"


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"comparisons").mkdir(exist_ok=True)
    sys.path[:0]=["/root/seva_vggt/bridge","/root/vggt"]
    import evaluate_mvgen_target_depth_filter_test30 as f
    from gsplat import rasterization
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    device=torch.device("cuda:0");perceptual=lpips.LPIPS(net="alex").eval().to(device);rows=[]
    for scene in SCENES:
        branch=RECON/scene/BRANCH;rec=torch.load(branch/"vggt_reconstruction.pt",map_location="cpu")
        paths=json.loads((branch/"metrics.json").read_text())["input_paths"]
        gt_paths=[v["image"] for v in json.loads((MANIFESTS/f"{scene}.json").read_text())["views"]]
        geoms=[];valid=torch.zeros((8,f.MODEL_CANVAS,f.MODEL_CANVAS),dtype=torch.bool,device=device)
        for i,p in enumerate(paths):
            with Image.open(p) as im:g=f.geom(*im.size)
            geoms.append(g);valid[i,g["pt"]:f.MODEL_CANVAS-g["pb"],g["pl"]:f.MODEL_CANVAS-g["pr"]]=True
        grid={k:v[0].float().reshape(8,f.MODEL_CANVAS,f.MODEL_CANVAS,v.shape[-1]).to(device)
              for k,v in rec["gaussians"].items() if torch.is_tensor(v) and v.ndim>=2}
        flat={k:v[valid] for k,v in grid.items()};source_ids=torch.arange(8,device=device)[:,None,None].expand_as(valid)[valid]
        ext,intr=pose_encoding_to_extri_intri(rec["pose_enc"].float(),(f.MODEL_CANVAS,f.MODEL_CANVAS));ext,intr=ext[0].to(device),intr[0].to(device)
        visual=[]
        for slot in range(1,8):
            inp=f.image_tensor(paths[slot],device);gt=f.image_tensor(gt_paths[slot],device);gi=geoms[slot]
            with Image.open(gt_paths[slot]) as im:gg=f.geom(*im.size)
            Ki=f.native_k(intr[slot],gi);Kg=f.native_k(intr[slot],gg);view=f.viewmat(ext[slot],device);target=source_ids==slot
            with torch.inference_mode():
                rd,ra=f.target_depth(rasterization,flat,target,view,Ki,gi["h"],gi["w"])
                cam=flat["xyz"]@view[:3,:3].T+view[:3,3];z=cam[:,2]
                u=Ki[0,0]*cam[:,0]/z.clamp_min(1e-8)+Ki[0,2];v=Ki[1,1]*cam[:,1]/z.clamp_min(1e-8)+Ki[1,2]
                inside=(z>.01)&(u>=0)&(u<gi["w"])&(v>=0)&(v<gi["h"])&(~target)
                px=u.round().long().clamp(0,gi["w"]-1);py=v.round().long().clamp(0,gi["h"]-1)
                sd,sa=rd[py,px],ra[py,px];rel=(z-sd)/sd.clamp_min(1e-6)
                conflict=inside&(sa>=f.ALPHA_THR)&torch.isfinite(sd)&(rel < -f.FRONT_THR)
                one=torch.ones(len(source_ids),device=device);weights=one.clone();weights[conflict]=f.CONFLICT_WEIGHT
                bi,_=f.raster(rasterization,flat,one,view,Ki,gi["h"],gi["w"]);pi,_=f.raster(rasterization,flat,weights,view,Ki,gi["h"],gi["w"])
                bg,_=f.raster(rasterization,flat,one,view,Kg,gg["h"],gg["w"]);pg,_=f.raster(rasterization,flat,weights,view,Kg,gg["h"],gg["w"])
            b_in,p_in=bi[...,:3].permute(2,0,1).clamp(0,1),pi[...,:3].permute(2,0,1).clamp(0,1)
            b_gt,p_gt=bg[...,:3].permute(2,0,1).clamp(0,1),pg[...,:3].permute(2,0,1).clamp(0,1)
            mbi,mpi=f.metric(inp,b_in,perceptual),f.metric(inp,p_in,perceptual);mgb,mgp=f.metric(gt,b_gt,perceptual),f.metric(gt,p_gt,perceptual)
            row={"scene":scene,"category":scene.split("_")[0],"split":"all","slot":slot,"source_type":"generated",
                 "front_conflict_fraction_non_target":float(conflict.sum()/(~target).sum()),"opacity_weight_mean":float(weights.mean())}
            for prefix,data in (("baseline_input",mbi),("processed_input",mpi),("baseline_gt",mgb),("processed_gt",mgp)):
                row.update({f"{prefix}_{k}":x for k,x in data.items()})
            rows.append(row);visual.append({"slot":slot,"gt":f.to_pil(gt),"input":f.to_pil(inp),"base_gt":f.to_pil(b_gt),"proc_gt":f.to_pil(p_gt),
                "base_in":f.to_pil(b_in),"proc_in":f.to_pil(p_in),"gt_gain":mgp["psnr"]-mgb["psnr"],"input_gain":mpi["psnr"]-mbi["psnr"]})
        f.make_sheet(scene,visual,OUT/"comparisons"/f"{scene}.jpg");del rec,grid,flat;torch.cuda.empty_cache()
    f.write_csv(OUT/"metrics_per_view.csv",rows);summary=f.aggregate(rows,"all")
    summary["protocol"]={"filter":"target-source expected depth; alpha>=0.75; non-target front conflict >3%; opacity x0.25",
                         "uses_gt_for_filter":False,"hard_delete":False,"base":"geometry-aware MVGen adapter + frozen VGGT-G"}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
