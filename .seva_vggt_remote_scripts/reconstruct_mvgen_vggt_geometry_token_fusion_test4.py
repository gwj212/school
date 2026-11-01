#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time,traceback
from pathlib import Path
import torch
from reconstruct_mvgen_vggt_feature_fusion_test4 import SCENES,cpu_tree

def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifests",type=Path,required=True); p.add_argument("--generated",type=Path,required=True); p.add_argument("--latents",type=Path,required=True); p.add_argument("--adapter",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 sys.path[:0]=["/root/seva_vggt/bridge","/root/vggt"]
 import run_pipeline as rp
 from run_vggt_head_adapter_test5 import export_filtered
 from train_mvgen_vggt_geometry_token_fusion import GeometryTokenFusion
 from vggt.models.vggt import VGGT
 from vggt.utils.load_fn import load_and_preprocess_images
 d=torch.device("cuda:0"); model=VGGT.from_pretrained(str(rp.DEFAULT_VGGT_MODEL),enable_gaussian=True)
 if hasattr(model.gaussian_head,"enable_color_head_after_init"): model.gaussian_head.enable_color_head_after_init()
 model.point_head=None; model.track_head=None; ck=rp.load_reconstruction_checkpoint(model,Path("/root/vggt/ckpts/ckpt.pth")); model.requires_grad_(False).to(d).eval()
 fusion=GeometryTokenFusion().to(d).eval(); fusion.load_state_dict(torch.load(a.adapter,map_location=d)["state_dict"]); status={"completed":[],"failed":{},"checkpoint":ck,"adapter":str(a.adapter)}
 for scene in SCENES:
  b=a.output/"scenes"/scene/"mvgen_geometry_token_fusion_1real7gen"; b.mkdir(parents=True,exist_ok=True)
  try:
   m=json.loads((a.manifests/f"{scene}.json").read_text()); paths=[m["views"][0]["image"]]+[str(a.generated/scene/"images"/f"view_{i:02d}_generated.png") for i in range(1,8)]
   inp=load_and_preprocess_images(paths,mode="pad").to(d); fusion.current_latents=torch.load(a.latents/f"{scene}.pt",map_location=d)["latents"][None]; h=model.aggregator.register_forward_hook(fusion.hook)
   torch.cuda.synchronize(); t=time.perf_counter()
   with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16): rec=model(inp[None])
   torch.cuda.synchronize(); elapsed=time.perf_counter()-t; h.remove(); torch.save(cpu_tree(rec),b/"vggt_reconstruction.pt"); pc=export_filtered(rec,b/"point_cloud_filtered_no_cap.ply")
   (b/"metrics.json").write_text(json.dumps({"scene":scene,"model_inference_seconds":elapsed,"point_cloud":pc},indent=2)+"\n"); status["completed"].append(scene); print("COMPLETE",scene,elapsed,flush=True); del rec,inp; torch.cuda.empty_cache()
  except Exception as e: status["failed"][scene]={"error":repr(e),"traceback":traceback.format_exc()}
  (a.output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
 if status["failed"]: raise SystemExit(1)
if __name__=="__main__": main()
