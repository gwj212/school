#!/usr/bin/env python3
"""Generate SEVA 1-real + 7-target views for the sealed validation30 only.

The SEVA model is loaded once and scenes are resumable.  No VGGT inference and
no blind-test manifest is touched by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image


ROOT = Path("/root/seva_vggt")


def complete(scene_dir: Path) -> bool:
    generated = scene_dir / "generated_only"
    return (scene_dir / "metadata.json").is_file() and all(
        (generated / f"view_{index:02d}.png").is_file() for index in range(1, 8)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--single-cfg", type=float, default=4.0)
    parser.add_argument("--short-side", type=int, default=576)
    parser.add_argument("--context", type=int, default=21)
    parser.add_argument("--chunk-strategy", default="nearest-gt")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--seva-root", type=Path, default=ROOT / "seva")
    parser.add_argument("--seva-model", type=Path, default=ROOT / "models/stable-virtual-camera")
    parser.add_argument("--seva-version", type=float, default=1.1)
    parser.add_argument("--vae-model", type=Path, default=ROOT / "models/sd-vae-ft-mse")
    parser.add_argument(
        "--clip-weights",
        type=Path,
        default=ROOT / "models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
    )
    args = parser.parse_args()
    if args.short_side % 64:
        raise ValueError("--short-side must be divisible by 64")
    protocol = json.loads(args.protocol.read_text())
    scenes = list(protocol["validation_scenes"])
    blind = set(protocol["blind_test_scenes"])
    overlap = sorted(set(scenes) & blind)
    if overlap:
        raise RuntimeError(f"validation/blind overlap: {overlap}")
    if len(scenes) != 30:
        raise RuntimeError(f"expected 30 validation scenes, got {len(scenes)}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    sys.path[:0] = [str(ROOT / "bridge"), str(args.seva_root)]
    import run_pipeline as rp
    from seva.eval import run_one_scene, set_lowvram_mode
    from seva.model import SGMWrapper
    from seva.modules.autoencoder import AutoEncoder
    from seva.modules.conditioner import CLIPConditioner
    from seva.sampling import DiscreteDenoiser
    from seva.utils import load_model as load_seva_model

    # Validate all heavyweight local assets without reading a blind manifest.
    first_manifest = args.manifest_root / f"{scenes[0]}.json"
    validation_args = argparse.Namespace(
        manifest=first_manifest,
        seva_root=args.seva_root,
        seva_model=args.seva_model,
        vae_model=args.vae_model,
        clip_weights=args.clip_weights,
        vggt_root=Path("/root/vggt"),
        vggt_model=rp.DEFAULT_VGGT_MODEL,
        vggt_checkpoint=Path("/root/vggt/ckpts/ckpt.pth"),
    )
    rp.validate_paths(validation_args)

    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
    os.environ["SEVA_WRITE_VIDEO"] = "0"
    os.environ["SEVA_VAE_MODEL"] = str(args.vae_model)
    os.environ["SEVA_CLIP_MODEL"] = str(args.clip_weights)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    set_lowvram_mode(True)
    load_started = time.perf_counter()
    seva_model = SGMWrapper(
        load_seva_model(
            model_version=args.seva_version,
            pretrained_model_name_or_path=str(args.seva_model),
            device="cpu",
            dtype=dtype,
        ).eval()
    )
    autoencoder = AutoEncoder(chunk_size=1).eval()
    conditioner = CLIPConditioner().eval()
    denoiser = DiscreteDenoiser(num_idx=1000, device=str(device))
    model_load_seconds = time.perf_counter() - load_started

    status = {
        "protocol": str(args.protocol),
        "split": "validation_scenes",
        "blind_accessed": False,
        "scene_count": len(scenes),
        "settings": {
            "steps": args.steps,
            "single_cfg": args.single_cfg,
            "short_side": args.short_side,
            "context": args.context,
            "chunk_strategy": args.chunk_strategy,
            "seed": args.seed,
            "dtype": str(dtype),
        },
        "model_load_seconds": model_load_seconds,
        "completed": [],
        "failed": [],
    }
    status_path = args.output_root / "status.json"
    for scene_index, scene in enumerate(scenes, 1):
        scene_dir = args.output_root / scene
        if complete(scene_dir):
            status["completed"].append(scene)
            print(f"SEVA SKIP {scene_index}/{len(scenes)} {scene}", flush=True)
            continue
        manifest_path = args.manifest_root / f"{scene}.json"
        payload, _, _ = rp.load_manifest(manifest_path)
        baseline, inputs, targets = rp.select_fixed_total_split(payload, 1, 8)
        if [int(view["index"]) for view in baseline] != list(range(8)):
            raise RuntimeError(f"{scene}: expected fixed slots 0..7")
        if [int(view["index"]) for view in inputs] != [0]:
            raise RuntimeError(f"{scene}: expected real input slot 0")
        ordered_views = inputs + targets
        c2ws, intrinsics = rp.tensor_cameras(ordered_views)
        version = {
            "H": args.short_side,
            "W": args.short_side,
            "T": args.context,
            "C": 4,
            "f": 8,
            "options": {
                "chunk_strategy": args.chunk_strategy,
                "beta_linear_start": 5e-6,
                "log_snr_shift": 2.4,
                "guider_types": 1,
                "cfg": args.single_cfg,
                "camera_scale": 2.0,
                "num_steps": args.steps,
                "cfg_min": 1.2,
                "encoding_t": 1,
                "decoding_t": 1,
                "save_input": False,
                "save_output": False,
                "write_video": False,
                "return_tensors": True,
                "L_short": args.short_side,
                "transform_input": "crop",
                "transform_target": "crop",
            },
        }
        image_cond = {
            "img": [view["image"] for view in inputs] + [None] * len(targets),
            "input_indices": [0],
            "prior_indices": None,
        }
        camera_cond = {
            "c2w": c2ws,
            "K": intrinsics,
            "input_indices": list(range(8)),
        }
        started = time.perf_counter()
        try:
            result = next(
                run_one_scene(
                    "img2img",
                    version,
                    model=seva_model,
                    ae=autoencoder,
                    conditioner=conditioner,
                    denoiser=denoiser,
                    image_cond=image_cond,
                    camera_cond=camera_cond,
                    save_path=str(scene_dir / "seva_internal"),
                    use_traj_prior=False,
                    traj_prior_Ks=None,
                    traj_prior_c2ws=None,
                    seed=args.seed,
                )
            )
            generated = next(
                value for key, value in result.items() if key.startswith("samples-rgb")
            ).detach().cpu()
            paths = rp.save_generated(generated, targets, scene_dir / "generated_only")
            metadata = {
                "scene": scene,
                "manifest": str(manifest_path),
                "split": "validation",
                "input_indices": [0],
                "target_indices": list(range(1, 8)),
                "generated_paths": paths,
                "output_canvas": list(Image.open(paths[0]).size),
                "elapsed_seconds": time.perf_counter() - started,
                **status["settings"],
            }
            (scene_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            status["completed"].append(scene)
            del result, generated
            torch.cuda.empty_cache()
            print(f"SEVA DONE {scene_index}/{len(scenes)} {scene}", flush=True)
        except Exception as error:
            status["failed"].append({"scene": scene, "error": repr(error)})
            print(f"SEVA FAILED {scene_index}/{len(scenes)} {scene}: {error!r}", flush=True)
        status["updated_at_unix"] = time.time()
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    status["finished_at_unix"] = time.time()
    status["complete_scene_count"] = sum(complete(args.output_root / scene) for scene in scenes)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
