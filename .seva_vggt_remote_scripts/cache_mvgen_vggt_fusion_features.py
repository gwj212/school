#!/usr/bin/env python3
"""Cache MVGen VAE latents for the exact eight VGGT input views.

The cache is only an optimization for the pilot.  In a deployed fused model the
same frozen VAE encoder is part of the forward graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def pad518(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width >= height:
        new_width = 518
        new_height = round(height * (518 / width) / 14) * 14
    else:
        new_height = 518
        new_width = round(width * (518 / height) / 14) * 14
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    tensor = pil_to_tensor(image).float() / 255.0
    dh, dw = 518 - new_height, 518 - new_width
    return F.pad(tensor, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), value=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(
        args.base_model, subfolder="vae", local_files_only=True
    ).requires_grad_(False).to(device, dtype=torch.float16).eval()
    index = {}
    for scene in args.scenes:
        manifest = json.loads((args.manifests / f"{scene}.json").read_text())
        paths = [manifest["views"][0]["image"]] + [
            str(args.generated / scene / "images" / f"view_{slot:02d}_generated.png")
            for slot in range(1, 8)
        ]
        images = torch.stack([pad518(path) for path in paths]).to(device, dtype=torch.float16)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            latents = vae.encode(images * 2 - 1).latent_dist.mode()
            latents = latents * vae.config.scaling_factor
        output = args.output / f"{scene}.pt"
        torch.save({"scene": scene, "paths": paths, "latents": latents.cpu()}, output)
        index[scene] = {"path": str(output), "shape": list(latents.shape), "inputs": paths}
        print(f"CACHED {scene} {tuple(latents.shape)}", flush=True)
    (args.output / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
