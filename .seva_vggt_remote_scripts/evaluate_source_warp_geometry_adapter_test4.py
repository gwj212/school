#!/usr/bin/env python3
"""Check whether the geometry adapter improves source-view reprojection agreement."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

SCENES=["chair_620_101264_203062","laptop_373_41558_82973","plant_598_92218_184789","toybus_121_14153_28986"]
M=Path("/root/seva_vggt/config/test30_mvgen")
BASE=Path("/root/seva_vggt/outputs/test30_mvgen_vggt/mvgen")
ADAPT=Path("/root/seva_vggt/outputs/test30_mvgen_vggt/mvgen_geometry_adapter_test4")
OUT=ADAPT/"source_warp_evaluation"


def load_normalized(path,w,h):
    image=Image.open(path).convert("RGB").resize((w,h),Image.Resampling.BICUBIC)
    return np.asarray(image,dtype=np.float32).transpose(2,0,1)/127.5-1.0


def metrics(pred,reference,mask):
    valid=np.broadcast_to(mask.astype(bool),pred.shape)
    diff=pred[valid]-reference[valid]
    mse=float(np.mean(diff**2)); mae=float(np.mean(np.abs(diff)))
    return {"mse":mse,"mae":mae,"psnr":float(10*math.log10(4.0/max(mse,1e-12))),"valid_pixels":int(mask.sum())}


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"comparisons").mkdir(exist_ok=True)
    repo=Path("/root/seva_vggt/candidates/MVGenMaster"); os.chdir(repo); sys.path[:0]=["/root/seva_vggt/bridge",str(repo)]
    from depth_pro.depth_pro import create_model_and_transforms
    from train_mvgen_geometry_adapter import prepare_raw
    model,transform=create_model_and_transforms(device=torch.device("cuda")); model.eval()
    rows=[]
    for scene in SCENES:
        raw=prepare_raw(M/f"{scene}.json",repo/"check_points"/"pretrained_model",model,transform)
        h,w=raw["hw"]; warp=raw["source_warps"].numpy(); masks=raw["source_warp_masks"].numpy()
        sheet=Image.new("RGB",(3*w,7*h),"white")
        for slot in range(1,8):
            base=load_normalized(BASE/scene/"images"/f"view_{slot:02d}_generated.png",w,h)
            adapt=load_normalized(ADAPT/scene/"images"/f"view_{slot:02d}_generated.png",w,h)
            bm=metrics(base,warp[slot],masks[slot]); am=metrics(adapt,warp[slot],masks[slot])
            row={"scene":scene,"slot":slot,**{f"original_{k}":v for k,v in bm.items()},**{f"adapted_{k}":v for k,v in am.items()}}
            for k in ("mse","mae","psnr"): row[f"delta_{k}"]=row[f"adapted_{k}"]-row[f"original_{k}"]
            rows.append(row)
            mask=np.broadcast_to(masks[slot].transpose(1,2,0)>0,(h,w,3))
            warp_rgb=((warp[slot].transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)
            warp_rgb=np.where(mask,warp_rgb,255).astype(np.uint8)
            sheet.paste(Image.fromarray(warp_rgb),(0,(slot-1)*h))
            sheet.paste(Image.fromarray(((base.transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)),(w,(slot-1)*h))
            sheet.paste(Image.fromarray(((adapt.transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)),(2*w,(slot-1)*h))
        sheet.thumbnail((1200,5000),Image.Resampling.LANCZOS); sheet.save(OUT/"comparisons"/f"{scene}.jpg",quality=92)
    fields=list(rows[0]);
    with (OUT/"metrics_per_view.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
    summary={}
    for key in ("psnr","mae","mse"):
        a=np.array([r[f"original_{key}"] for r in rows]);b=np.array([r[f"adapted_{key}"] for r in rows]);d=b-a
        summary[key]={"original_mean":float(a.mean()),"adapted_mean":float(b.mean()),"delta_mean":float(d.mean()),
                      "improved_views":int(((d>0) if key=="psnr" else (d<0)).sum()),"views":len(rows)}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
