#!/usr/bin/env python3
"""Train/evaluate a tiny target-view Gaussian visibility gate with frozen MVGen/VGGT outputs."""

from __future__ import annotations

import csv, json, math, random, sys, time, traceback
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

MODEL_CANVAS=518; PATCH=14; TRAIN_LONG=224
ALPHA_THR=.75; FRONT_THR=.03; FIXED_WEIGHT=.25
ROOT=Path("/root/seva_vggt/outputs/test30_mvgen_vggt")
MANIFESTS=Path("/root/seva_vggt/config/test30_mvgen")
OUTPUT=ROOT/"visibility_gate_trainval"
TRAIN_CATS={"apple","ball","bicycle","car","couch","cup","teddybear","vase"}
VAL_CATS={"bottle","toaster"}
TEST_CATS={"chair","laptop","plant","toybus"}


class VisibilityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(9,24),nn.SiLU(),nn.Linear(24,12),nn.SiLU(),nn.Linear(12,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(FIXED_WEIGHT/(1-FIXED_WEIGHT)))
    def forward(self,x):
        return .05+.95*torch.sigmoid(self.net(x)).squeeze(-1)


def geom(w,h):
    if w>=h: rw,rh=MODEL_CANVAS,round(h*MODEL_CANVAS/w/PATCH)*PATCH
    else: rh,rw=MODEL_CANVAS,round(w*MODEL_CANVAS/h/PATCH)*PATCH
    pw,ph=MODEL_CANVAS-rw,MODEL_CANVAS-rh
    return {"w":w,"h":h,"rw":int(rw),"rh":int(rh),"pl":int(pw//2),"pr":int(pw-pw//2),"pt":int(ph//2),"pb":int(ph-ph//2)}


def native_k(k,g):
    o=k.clone(); sx,sy=g["w"]/g["rw"],g["h"]/g["rh"]
    o[0,0],o[1,1]=k[0,0]*sx,k[1,1]*sy; o[0,2],o[1,2]=(k[0,2]-g["pl"])*sx,(k[1,2]-g["pt"])*sy
    return o


def scaled_k(k,w0,h0,w,h):
    o=k.clone(); o[0]*=w/w0; o[1]*=h/h0; return o


def fit_size(w,h,long_side):
    s=min(1.0,long_side/max(w,h)); return max(1,round(w*s)),max(1,round(h*s))


def viewmat(ext,device):
    o=torch.eye(4,device=device); o[:3,:3]=ext[:3,:3]; o[:3,3]=ext[:3,3]; return o


def img_tensor(path,device,size=None,mask=False):
    with Image.open(path) as im:
        im=im.convert("L" if mask else "RGB")
        if size: im=im.resize(size,Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
        a=np.asarray(im,dtype=np.float32)/255
    if mask: return torch.from_numpy(a).to(device)
    return torch.from_numpy(a).permute(2,0,1).contiguous().to(device)


def to_pil(x):
    a=(x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255+0.5).astype(np.uint8); return Image.fromarray(a)


def metric(a,b,perceptual):
    mse=float(((a-b)**2).mean()); aa=a.permute(1,2,0).cpu().numpy(); bb=b.permute(1,2,0).cpu().numpy()
    with torch.inference_mode(): l=float(perceptual(a[None]*2-1,b[None]*2-1).flatten()[0])
    return {"psnr":-10*math.log10(max(mse,1e-12)),"ssim":float(structural_similarity(aa,bb,channel_axis=2,data_range=1.0)),"lpips":l,"mse":mse}


def robust_z(x):
    med=x.median(); mad=(x-med).abs().median().clamp_min(1e-5); return ((x-med)/(1.4826*mad)).clamp(-5,5)/5


def raster(rasterization,flat,weights,view,K,h,w,depth=False):
    out,alpha,_=rasterization(means=flat["xyz"],quats=flat["rotation"],scales=flat["scale"],
        opacities=flat["opacity"][:,0]*weights,colors=flat["color"],viewmats=view[None],Ks=K[None],width=w,height=h,
        near_plane=.01,far_plane=1e10,render_mode="RGB+ED" if depth else "RGB")
    return out[0],alpha[0,...,0]


def target_depth(rasterization,flat,target,view,K,h,w):
    out,alpha,_=rasterization(means=flat["xyz"][target],quats=flat["rotation"][target],scales=flat["scale"][target],
        opacities=flat["opacity"][target,0],colors=flat["color"][target],viewmats=view[None],Ks=K[None],width=w,height=h,
        near_plane=.01,far_plane=1e10,render_mode="RGB+ED")
    return out[0,...,3],alpha[0,...,0]


def load_scene(scene,device,pose_decode):
    branch=ROOT/"reconstruction/scenes"/scene/"mvgen_1real7gen"; rec=torch.load(branch/"vggt_reconstruction.pt",map_location="cpu")
    paths=json.loads((branch/"metrics.json").read_text())["input_paths"]
    manifest=json.loads((MANIFESTS/f"{scene}.json").read_text()); views=manifest["views"]
    geoms=[]; valid=torch.zeros((8,MODEL_CANVAS,MODEL_CANVAS),dtype=torch.bool,device=device)
    for i,p in enumerate(paths):
        with Image.open(p) as im:g=geom(*im.size)
        geoms.append(g); valid[i,g["pt"]:MODEL_CANVAS-g["pb"],g["pl"]:MODEL_CANVAS-g["pr"]]=True
    grid={k:v[0].float().reshape(8,MODEL_CANVAS,MODEL_CANVAS,v.shape[-1]).to(device)
          for k,v in rec["gaussians"].items() if torch.is_tensor(v) and v.ndim>=2}
    flat={k:v[valid] for k,v in grid.items()}; source_ids=torch.arange(8,device=device)[:,None,None].expand_as(valid)[valid]
    depth_conf=rec["depth_conf"][0].float().to(device)[valid]
    ext,intr=pose_decode(rec["pose_enc"].float(),(MODEL_CANVAS,MODEL_CANVAS)); ext,intr=ext[0].to(device),intr[0].to(device)
    static={"opacity":flat["opacity"][:,0].clamp(0,1),
            "scale":robust_z(flat["scale"].amax(-1).clamp_min(1e-8).log()),
            "depth_conf":robust_z(torch.log1p(depth_conf.clamp_min(0))),
            "color_delta":robust_z(torch.log1p(flat.get("color_delta_norm",torch.zeros_like(depth_conf)).reshape(-1).abs()))}
    return {"scene":scene,"rec":rec,"paths":paths,"views":views,"geoms":geoms,"flat":flat,"source_ids":source_ids,
            "ext":ext,"intr":intr,"static":static}


def conflict_features(data,slot,rasterization):
    flat=data["flat"]; sid=data["source_ids"]; gi=data["geoms"][slot]; view=viewmat(data["ext"][slot],sid.device)
    Ki=native_k(data["intr"][slot],gi); target=sid==slot
    with torch.no_grad():
        rd,ra=target_depth(rasterization,flat,target,view,Ki,gi["h"],gi["w"])
        cam=flat["xyz"]@view[:3,:3].T+view[:3,3]; z=cam[:,2]
        u=Ki[0,0]*cam[:,0]/z.clamp_min(1e-8)+Ki[0,2]; v=Ki[1,1]*cam[:,1]/z.clamp_min(1e-8)+Ki[1,2]
        inside=(z>.01)&(u>=0)&(u<gi["w"])&(v>=0)&(v<gi["h"])&(~target)
        px=u.round().long().clamp(0,gi["w"]-1); py=v.round().long().clamp(0,gi["h"]-1)
        sd,sa=rd[py,px],ra[py,px]; rel=(z-sd)/sd.clamp_min(1e-6)
        conflict=inside&(sa>=ALPHA_THR)&torch.isfinite(sd)&(rel<-FRONT_THR); idx=torch.where(conflict)[0]
        rs=data["ext"][:, :3,:3]; rt=rs[slot]; tr=(rs@rt.T).diagonal(dim1=-2,dim2=-1).sum(-1)
        angles=torch.acos(((tr-1)/2).clamp(-1,1))/math.pi
        st=data["static"]
        feat=torch.stack([(-rel[idx]).clamp(0,.5)*2,sa[idx],st["opacity"][idx],st["scale"][idx],st["depth_conf"][idx],
                          st["color_delta"][idx],(sid[idx]==0).float(),angles[sid[idx]],(sid[idx].float()-slot).abs()/7],-1)
    return idx,feat,view,Ki,float(conflict.float().mean()),target


def weights_for(model,n,idx,feat,mode,device):
    w=torch.ones(n,device=device)
    if mode=="fixed": w[idx]=FIXED_WEIGHT
    elif mode=="learned" and len(idx): w[idx]=model(feat)
    return w


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def fitted(im,w,h):
    c=Image.new("RGB",(w,h),"white"); x=im.convert("RGB");x.thumbnail((w,h),Image.Resampling.LANCZOS);c.paste(x,((w-x.width)//2,(h-x.height)//2));return c


def sheet(scene,vis,path):
    titles=("GT","MVGen","base GT","fixed GT","learned GT","base input","fixed input","learned input");cw,ch,lw,hh=145,150,84,54
    out=Image.new("RGB",(lw+len(titles)*cw,hh+7*ch),"white");d=ImageDraw.Draw(out);d.text((4,4),scene,fill="black")
    for c,t in enumerate(titles):d.text((lw+c*cw+3,28),t,fill="black")
    for r in vis:
        y=hh+(r["slot"]-1)*ch;d.text((3,y+5),f"view {r['slot']:02d}",fill="black");d.text((3,y+23),f"dP {r['gain']:+.2f}",fill="black")
        for c,k in enumerate(("gt","inp","bgt","fgt","lgt","bin","fin","lin")):out.paste(fitted(r[k],cw,ch-5),(lw+c*cw,y))
    out.save(path,quality=92)


def train_epoch(model,opt,scenes,device,rasterization,pose_decode,epoch):
    model.train(); random.Random(20261020+epoch).shuffle(scenes); logs=[]
    for si,scene in enumerate(scenes):
        data=load_scene(scene,device,pose_decode)
        slots=list(range(1,8));random.Random(epoch*100+si).shuffle(slots)
        for slot in slots:
            idx,feat,view,Ki,frac,target=conflict_features(data,slot,rasterization)
            if not len(idx):continue
            v=data["views"][slot]; gt_path=v["image"]; mask_path=v["mask"]
            with Image.open(gt_path) as im:w0,h0=im.size
            w,h=fit_size(w0,h0,TRAIN_LONG); Kg=native_k(data["intr"][slot],geom(w0,h0));Kt=scaled_k(Kg,w0,h0,w,h)
            gt=img_tensor(gt_path,device,(w,h)); mask=img_tensor(mask_path,device,(w,h),mask=True)
            weights=weights_for(model,len(data["source_ids"]),idx,feat,"learned",device)
            out,_=raster(rasterization,data["flat"],weights,view,Kt,h,w); pred=out[...,:3].permute(2,0,1).clamp(0,1)
            pix=(.5+1.5*mask)[None]; l1=((pred-gt).abs()*pix).mean(); mse=(((pred-gt)**2)*pix).mean()
            gate=weights[idx]; reg=(gate-FIXED_WEIGHT).abs().mean(); loss=l1+.5*mse+.002*reg
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            logs.append({"epoch":epoch,"scene":scene,"slot":slot,"loss":float(loss),"l1":float(l1),"mse":float(mse),
                         "gate_mean":float(gate.mean()),"gate_min":float(gate.min()),"gate_max":float(gate.max()),"conflict_fraction":frac})
        del data;torch.cuda.empty_cache();print(f"TRAIN e{epoch} {si+1}/{len(scenes)} {scene}",flush=True)
    return logs


def evaluate(model,scenes,split,device,rasterization,pose_decode,perceptual,save_visual=False):
    model.eval();rows=[]
    for si,scene in enumerate(scenes):
        data=load_scene(scene,device,pose_decode);vis=[]
        for slot in range(1,8):
            idx,feat,view,Ki,frac,target=conflict_features(data,slot,rasterization);v=data["views"][slot]
            inp=img_tensor(data["paths"][slot],device);gt=img_tensor(v["image"],device)
            gi=data["geoms"][slot]
            with Image.open(v["image"]) as gt_im: gt_w,gt_h=gt_im.size
            gg=geom(gt_w,gt_h);Kg=native_k(data["intr"][slot],gg)
            images={};ms={}
            with torch.inference_mode():
                for mode in ("baseline","fixed","learned"):
                    w=weights_for(model,len(data["source_ids"]),idx,feat,mode,device)
                    oi,_=raster(rasterization,data["flat"],w,view,Ki,gi["h"],gi["w"]);og,_=raster(rasterization,data["flat"],w,view,Kg,gg["h"],gg["w"])
                    pi=oi[...,:3].permute(2,0,1).clamp(0,1);pg=og[...,:3].permute(2,0,1).clamp(0,1)
                    images[mode]=(pi,pg);ms[(mode,"input")]=metric(inp,pi,perceptual);ms[(mode,"gt")]=metric(gt,pg,perceptual)
            row={"split":split,"scene":scene,"category":scene.split("_")[0],"slot":slot,"conflict_gaussians":len(idx),"conflict_fraction":frac,
                 "learned_gate_mean":float(model(feat).mean()) if len(idx) else 1.0}
            for mode in ("baseline","fixed","learned"):
                for ref in ("input","gt"):row.update({f"{mode}_{ref}_{k}":x for k,x in ms[(mode,ref)].items()})
            rows.append(row)
            if save_visual:
                vis.append({"slot":slot,"gt":to_pil(gt),"inp":to_pil(inp),"bgt":to_pil(images["baseline"][1]),"fgt":to_pil(images["fixed"][1]),
                            "lgt":to_pil(images["learned"][1]),"bin":to_pil(images["baseline"][0]),"fin":to_pil(images["fixed"][0]),
                            "lin":to_pil(images["learned"][0]),"gain":row["learned_gt_psnr"]-row["baseline_gt_psnr"]})
            del inp,gt,images;torch.cuda.empty_cache()
        if save_visual:sheet(scene,vis,OUTPUT/"comparisons"/f"{scene}.jpg")
        del data;torch.cuda.empty_cache();print(f"EVAL {split} {si+1}/{len(scenes)} {scene}",flush=True)
    return rows


def summarize(rows):
    out={"scenes":len(set(r["scene"] for r in rows)),"views":len(rows)}
    for mode in ("baseline","fixed","learned"):
        for ref in ("input","gt"):
            for m in ("psnr","ssim","lpips","mse"):out[f"{mode}_{ref}_{m}"]=float(np.mean([r[f"{mode}_{ref}_{m}"] for r in rows]))
    for mode in ("fixed","learned"):
        for ref in ("input","gt"):
            for m in ("psnr","ssim","lpips","mse"):
                d=np.asarray([r[f"{mode}_{ref}_{m}"]-r[f"baseline_{ref}_{m}"] for r in rows]);better=d>0 if m in ("psnr","ssim") else d<0
                out[f"{mode}_delta_{ref}_{m}"]=float(d.mean());out[f"{mode}_improved_{ref}_{m}"]=int(better.sum())
    out["learned_gate_mean"]=float(np.mean([r["learned_gate_mean"] for r in rows]));return out


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/"comparisons").mkdir(exist_ok=True);sys.path.insert(0,"/root/vggt")
    from gsplat import rasterization
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    all_scenes=json.loads((MANIFESTS/"selection_protocol.json").read_text())["scenes"]
    train=[s for s in all_scenes if s.split("_")[0] in TRAIN_CATS];val=[s for s in all_scenes if s.split("_")[0] in VAL_CATS];test=[s for s in all_scenes if s.split("_")[0] in TEST_CATS]
    protocol={"train_categories":sorted(TRAIN_CATS),"val_categories":sorted(VAL_CATS),"test_categories":sorted(TEST_CATS),
              "train_scenes":train,"val_scenes":val,"test_scenes":test,"frozen_models":True,"trainable_parameters":None,
              "features":["front_depth_amount","reference_alpha","opacity","robust_log_scale","robust_log_depth_conf","color_delta_norm","source_is_real","camera_rotation_angle","slot_distance"],
              "training_target":"real GT RGB and object mask; GT is never an inference feature"}
    device=torch.device("cuda:0");torch.manual_seed(20261020);model=VisibilityGate().to(device);protocol["trainable_parameters"]=sum(p.numel() for p in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);perceptual=lpips.LPIPS(net="alex").eval().to(device)
    logs=[];best=-1e9;best_epoch=-1;history=[]
    for epoch in range(1,4):
        logs.extend(train_epoch(model,opt,train,device,rasterization,pose_encoding_to_extri_intri,epoch))
        val_rows=evaluate(model,val,"validation",device,rasterization,pose_encoding_to_extri_intri,perceptual,False);s=summarize(val_rows)
        score=s["learned_gt_psnr"]-s["fixed_gt_psnr"];history.append({"epoch":epoch,"score":score,"summary":s})
        torch.save({"model":model.state_dict(),"epoch":epoch,"protocol":protocol},OUTPUT/f"epoch_{epoch}.pt")
        if score>best:best=score;best_epoch=epoch;torch.save({"model":model.state_dict(),"epoch":epoch,"protocol":protocol},OUTPUT/"best.pt")
        print("VAL",epoch,json.dumps(s),flush=True)
    ck=torch.load(OUTPUT/"best.pt",map_location=device);model.load_state_dict(ck["model"])
    val_rows=evaluate(model,val,"validation_best",device,rasterization,pose_encoding_to_extri_intri,perceptual,True)
    test_rows=evaluate(model,test,"category_holdout_test",device,rasterization,pose_encoding_to_extri_intri,perceptual,True)
    write_csv(OUTPUT/"training_log.csv",logs);write_csv(OUTPUT/"metrics_per_view.csv",val_rows+test_rows)
    summary={"protocol":protocol,"best_epoch":best_epoch,"history":history,"validation":summarize(val_rows),"category_holdout_test":summarize(test_rows)}
    (OUTPUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    t=summary["category_holdout_test"]
    report=f"""# 轻量Gaussian可见性门控训练验证\n\n- MVGen与VGGT完全冻结；仅训练{protocol['trainable_parameters']}参数的MLP。\n- 训练类别：{', '.join(sorted(TRAIN_CATS))}；验证类别：{', '.join(sorted(VAL_CATS))}；测试为完全未见类别：{', '.join(sorted(TEST_CATS))}。\n- 最佳epoch：{best_epoch}。\n\n|方法|GT PSNR↑|GT SSIM↑|GT LPIPS↓|输入PSNR↑|\n|---|---:|---:|---:|---:|\n|无过滤|{t['baseline_gt_psnr']:.4f}|{t['baseline_gt_ssim']:.4f}|{t['baseline_gt_lpips']:.4f}|{t['baseline_input_psnr']:.4f}|\n|固定0.25软过滤|{t['fixed_gt_psnr']:.4f}|{t['fixed_gt_ssim']:.4f}|{t['fixed_gt_lpips']:.4f}|{t['fixed_input_psnr']:.4f}|\n|学习门控|{t['learned_gt_psnr']:.4f}|{t['learned_gt_ssim']:.4f}|{t['learned_gt_lpips']:.4f}|{t['learned_input_psnr']:.4f}|\n\n学习门控相对无过滤：GT PSNR {t['learned_delta_gt_psnr']:+.4f} dB，SSIM {t['learned_delta_gt_ssim']:+.5f}，LPIPS {t['learned_delta_gt_lpips']:+.5f}；输入PSNR {t['learned_delta_input_psnr']:+.4f} dB。\n"""
    (OUTPUT/"可见性门控训练报告.md").write_text(report,encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    try:main()
    except Exception:
        OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/"failure.txt").write_text(traceback.format_exc());raise
