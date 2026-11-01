#!/usr/bin/env python3
from pathlib import Path
import os
import sys

import torch
from easydict import EasyDict
from omegaconf import OmegaConf

repo = Path("/root/seva_vggt/candidates/MVGenMaster")
os.chdir(repo)
sys.path.insert(0, str(repo))

from my_diffusers.models import UNet2DConditionModel

model_dir = repo / "check_points" / "pretrained_model"
config = EasyDict(OmegaConf.load(model_dir / "config.yaml"))
base = Path("/root/seva_vggt/candidates/MVGenMaster/check_points/sd2-community-stable-diffusion-2-1")
if not base.exists():
    base = Path(config.pretrained_model_name_or_path)

unet = UNet2DConditionModel.from_pretrained(
    str(base), subfolder="unet", rank=0, model_cfg=config.model_cfg,
    low_cpu_mem_usage=False, ignore_mismatched_sizes=True, local_files_only=True,
)
weights = torch.load(model_dir / "ema_unet.pt", map_location="cpu")
unet.load_state_dict(weights)

groups = {}
for name, param in unet.named_parameters():
    prefix = name.split(".")[0]
    groups[prefix] = groups.get(prefix, 0) + param.numel()

print("TOP_LEVEL_GROUPS")
for key, value in sorted(groups.items(), key=lambda item: -item[1]):
    print(f"{key}\t{value}")
print("CANDIDATE_NAMES")
keywords = ("coord", "camera", "conv_in", "class_embedding", "attn1", "attn2")
for name, param in unet.named_parameters():
    if any(key in name for key in keywords):
        print(f"{name}\t{param.numel()}")
