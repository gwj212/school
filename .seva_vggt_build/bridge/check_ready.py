#!/usr/bin/env python3
"""Report whether the isolated SEVA/VGGT workspace is ready to run."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path("/root/seva_vggt")
CHECKS = {
    "seva_source": ROOT / "seva/seva/eval.py",
    "seva_weight": ROOT / "models/stable-virtual-camera/modelv1.1.safetensors",
    "seva_config": ROOT / "models/stable-virtual-camera/config.yaml",
    "vae_weight": ROOT / "models/sd-vae-ft-mse/diffusion_pytorch_model.safetensors",
    "clip_weight": ROOT
    / "models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
    "vggt_source": Path("/root/vggt/vggt/models/vggt.py"),
    "vggt_checkpoint": Path("/root/vggt/ckpts/ckpt.pth"),
}
EXPECTED_SIZES = {
    "seva_weight": 5_056_009_640,
    "vae_weight": 334_643_276,
    "clip_weight": 3_944_692_325,
}


def main() -> None:
    paths = {}
    for name, path in CHECKS.items():
        size = path.stat().st_size if path.is_file() else None
        expected = EXPECTED_SIZES.get(name)
        paths[name] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": size,
            "expected_bytes": expected,
            "complete": path.exists() and (expected is None or size == expected),
        }
    sys.path.insert(0, str(ROOT / "seva"))
    sys.path.insert(0, "/root/vggt")
    imports = {}
    for module in ("seva", "vggt", "gsplat", "diffusers", "open_clip"):
        try:
            importlib.import_module(module)
            imports[module] = "ok"
        except Exception as error:  # readiness report should show every failure
            imports[module] = f"{type(error).__name__}: {error}"
    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "paths": paths,
        "imports": imports,
        "files_ready": all(item["complete"] for item in paths.values()),
        "imports_ready": all(value == "ok" for value in imports.values()),
    }
    result["run_ready"] = (
        result["files_ready"] and result["imports_ready"] and result["cuda_available"]
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
