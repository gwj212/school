#!/usr/bin/env python3
"""Geometry-first tight fusion pilot.

MVGen local/source/scene latent features are injected into the four VGGT token
scales used by the depth head and into the camera token.  Final render loss is
allowed to flow through camera/depth into the bridge; a scale-invariant real8
depth-teacher loss anchors geometry.  Both pretrained backbones remain frozen.
"""

from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mvgen_vggt_feature_fusion import TRAIN, VAL, load_scene, render_loss


class GeometryTokenFusion(nn.Module):
    layers = (4, 11, 17, 23)
    def __init__(self, token_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 256, 1), nn.GELU(),
        )
        self.to_token = nn.Conv2d(256, token_dim, 1)
        nn.init.normal_(self.to_token.weight, std=1e-5); nn.init.zeros_(self.to_token.bias)
        self.layer_gate = nn.Parameter(torch.full((4,), -4.0))
        self.camera_gate = nn.Parameter(torch.tensor(-4.0))
        self.current_latents = None
        self.last_residual_rms = None

    def token_residual(self, patch_count, dtype):
        lat = self.current_latents
        b, s, _, h, w = lat.shape
        src = lat[:, :1].expand(-1, s, -1, -1, -1)
        mean = lat.mean(1, keepdim=True).expand(-1, s, -1, -1, -1)
        x = torch.cat([lat, src, mean], 2).reshape(b*s, 12, h, w).float()
        side = int(round(patch_count ** .5))
        x = self.to_token(self.encoder(x))
        x = F.interpolate(x, (side, side), mode="bilinear", align_corners=False)
        x = x.flatten(2).transpose(1, 2).reshape(b, s, patch_count, -1)
        self.last_residual_rms = x.float().square().mean().sqrt()
        return x.to(dtype)

    def hook(self, _module, _inputs, output):
        tokens, patch_start = output
        residual = self.token_residual(tokens[-1].shape[2] - patch_start, tokens[-1].dtype)
        result = list(tokens)
        for gate_idx, layer in enumerate(self.layers):
            value = result[layer].clone()
            value[:, :, patch_start:] = value[:, :, patch_start:] + torch.sigmoid(self.layer_gate[gate_idx]) * residual
            if layer == self.layers[-1]:
                value[:, :, 0] = value[:, :, 0] + torch.sigmoid(self.camera_gate) * residual.mean(2)
            result[layer] = value
        return result, patch_start


def normalized_log_depth(depth):
    depth = depth.float().clamp_min(1e-4)
    logd = depth.log()
    center = logd.flatten(-2).median(-1).values[..., None, None]
    return logd - center


def evaluate(model, fusion, cache, device, dtype, pose_fn, render_fn, ssim_fn, size, enabled):
    rows=[]; model.eval(); fusion.eval()
    with torch.inference_mode():
        for sample in cache:
            fusion.current_latents=sample["latents"][None].to(device)
            handle=model.aggregator.register_forward_hook(fusion.hook) if enabled else None
            with torch.autocast("cuda",dtype=dtype): pred=model(sample["input"][None].to(device))
            if handle: handle.remove()
            _, vals=render_loss(pred,sample["gt"].to(device),list(range(1,8)),pose_fn,render_fn,ssim_fn,size)
            rows += [{"scene":sample["scene"],**v} for v in vals]
            del pred; torch.cuda.empty_cache()
    return {"rows":rows,"mean_psnr":float(np.mean([r["psnr"] for r in rows])),
            "mean_ssim":float(np.mean([r["ssim"] for r in rows])),"mean_l1":float(np.mean([r["l1"] for r in rows]))}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifests",type=Path,required=True); p.add_argument("--generated",type=Path,required=True)
    p.add_argument("--latents",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=20); p.add_argument("--lr",type=float,default=8e-5); p.add_argument("--render-size",type=int,default=192); p.add_argument("--seed",type=int,default=23)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    sys.path[:0]=["/root/vggt","/root/seva_vggt/bridge"]
    import run_pipeline as rp
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.gs_optimizer import render_gaussians,ssim
    device=torch.device("cuda:0"); dtype=torch.bfloat16
    cache=[load_scene(s,a.manifests,a.generated,a.latents,load_and_preprocess_images) for s in TRAIN+VAL]
    model=VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL),enable_gaussian=True)
    if hasattr(model.gaussian_head,"enable_color_head_after_init"): model.gaussian_head.enable_color_head_after_init()
    ckpt=rp.load_reconstruction_checkpoint(model,Path("/root/vggt/ckpts/ckpt.pth")); model.point_head=None; model.track_head=None
    model.requires_grad_(False).to(device).eval(); model._release_depth=True
    fusion=GeometryTokenFusion().to(device)
    # Cache real8 depth teachers once; these are never inputs to the fused inference graph.
    with torch.inference_mode():
        for sample in cache:
            with torch.autocast("cuda",dtype=dtype): teacher=model(sample["gt"][None].to(device))
            sample["teacher_depth"]=normalized_log_depth(teacher["depth"][0]).cpu().half(); del teacher; torch.cuda.empty_cache()
    train,val=cache[:len(TRAIN)],cache[len(TRAIN):]
    before=evaluate(model,fusion,val,device,dtype,pose_encoding_to_extri_intri,render_gaussians,ssim,a.render_size,False)
    opt=torch.optim.AdamW(fusion.parameters(),lr=a.lr,weight_decay=.01); model.gaussian_head.train(); fusion.train()
    trace=[]; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    for step in range(a.steps):
        sample=train[step%len(train)]; slot=1+((step*3+step//len(train))%7); opt.zero_grad(set_to_none=True)
        fusion.current_latents=sample["latents"][None].to(device); handle=model.aggregator.register_forward_hook(fusion.hook)
        with torch.autocast("cuda",dtype=dtype): pred=model(sample["input"][None].to(device))
        handle.remove(); rloss,metrics=render_loss(pred,sample["gt"].to(device),[0,slot],pose_encoding_to_extri_intri,render_gaussians,ssim,a.render_size)
        depth_pred=normalized_log_depth(pred["depth"][0,[0,slot]])
        depth_target=sample["teacher_depth"][[0,slot]].to(device).float()
        depth_loss=F.smooth_l1_loss(depth_pred,depth_target,beta=.1)
        loss=rloss+.05*depth_loss+1e-4*fusion.last_residual_rms
        loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(fusion.parameters(),1).item()); opt.step()
        row={"step":step+1,"scene":sample["scene"],"slot":slot,"loss":float(loss.item()),"render_loss":float(rloss.item()),"depth_loss":float(depth_loss.item()),"grad_norm":grad,
             "layer_gates":[float(x) for x in torch.sigmoid(fusion.layer_gate).detach().cpu()],"camera_gate":float(torch.sigmoid(fusion.camera_gate).item()),"metrics":metrics}
        trace.append(row); print(json.dumps(row),flush=True); del pred,loss,rloss; torch.cuda.empty_cache()
    after=evaluate(model,fusion,val,device,dtype,pose_encoding_to_extri_intri,render_gaussians,ssim,a.render_size,True)
    path=a.output/"mvgen_vggt_geometry_token_fusion.pt"; torch.save({"state_dict":fusion.state_dict(),"steps":a.steps,"lr":a.lr,"seed":a.seed},path)
    summary={"method":"MVGen multi-view latent -> VGGT multi-scale depth/camera tokens -> depth/camera -> Gaussian render",
             "trainable_parameters":sum(x.numel() for x in fusion.parameters()),"train_scenes":TRAIN,"validation_scenes":VAL,"steps":a.steps,"lr":a.lr,"render_size":a.render_size,
             "validation_before":before,"validation_after":after,"validation_psnr_delta":after["mean_psnr"]-before["mean_psnr"],"validation_ssim_delta":after["mean_ssim"]-before["mean_ssim"],"validation_l1_delta":after["mean_l1"]-before["mean_l1"],
             "elapsed_seconds":time.perf_counter()-started,"peak_allocated_gib":torch.cuda.max_memory_allocated(device)/1024**3,"checkpoint":ckpt,"adapter":str(path)}
    (a.output/"training_trace.json").write_text(json.dumps(trace,indent=2,ensure_ascii=False)+"\n"); (a.output/"training_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n"); print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
