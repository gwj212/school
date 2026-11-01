#!/usr/bin/env python3
"""Download the public SEVA dependencies from a Hugging Face endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


JOBS = (
    ("stabilityai/sd-vae-ft-mse", "config.json", "sd-vae-ft-mse"),
    (
        "stabilityai/sd-vae-ft-mse",
        "diffusion_pytorch_model.safetensors",
        "sd-vae-ft-mse",
    ),
    (
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "open_clip_config.json",
        "CLIP-ViT-H-14-laion2B-s32B-b79K",
    ),
    (
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "open_clip_pytorch_model.bin",
        "CLIP-ViT-H-14-laion2B-s32B-b79K",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("/root/seva_vggt/models"))
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    for repo, filename, subdirectory in JOBS:
        destination = args.model_root / subdirectory
        destination.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=destination,
            endpoint=args.endpoint,
        )
        print(path, Path(path).stat().st_size)
    print(
        "The stabilityai/stable-virtual-camera repository is gated. "
        "Download it with an authorized Hugging Face account or restore the "
        "verified modelv1.1.safetensors from the source container."
    )


if __name__ == "__main__":
    main()
