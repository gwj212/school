#!/usr/bin/env python3
"""Render saved MVGenMaster->VGGT-G reconstructions at their input cameras."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


ROOT = Path("/root/seva_vggt/outputs/model_strategy_test5")
MANIFEST_ROOT = Path("/root/seva_vggt/config/strategy_test5")
OUT = ROOT / "vggt_render_comparison"
SCENES = [
    "chair_386_46082_92005",
    "car_599_92296_183295",
    "couch_349_36574_67726",
    "toybus_517_74123_144462",
    "ball_113_13373_24790",
]


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def metrics(reference: torch.Tensor, prediction: torch.Tensor, model: torch.nn.Module) -> dict[str, float | None]:
    reference = reference.float().clamp(0, 1)
    prediction = prediction.float().clamp(0, 1)
    mse = torch.mean((reference - prediction) ** 2).item()
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    ref_np = reference.permute(1, 2, 0).detach().cpu().numpy()
    pred_np = prediction.permute(1, 2, 0).detach().cpu().numpy()
    ssim = structural_similarity(ref_np, pred_np, channel_axis=2, data_range=1.0)
    with torch.inference_mode():
        distance = model(reference[None] * 2 - 1, prediction[None] * 2 - 1)
    return {
        "psnr": finite(psnr),
        "ssim": finite(float(ssim)),
        "lpips": finite(float(distance.flatten()[0].detach().cpu())),
    }


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def contact_sheet(scene: str, view_indices: list[int], real: torch.Tensor, generated: torch.Tensor,
                  rendered: list[torch.Tensor], output: Path) -> None:
    thumb = 260
    label_width = 135
    header = 46
    sheet = Image.new("RGB", (label_width + 3 * thumb, header + len(view_indices) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    titles = ("Real GT", "MVGenMaster", "VGGT-G render")
    for col, title in enumerate(titles):
        draw.text((label_width + col * thumb + 55, 15), title, fill="black")
    for slot, view_index in enumerate(view_indices):
        y = header + slot * thumb
        source_type = "real source" if slot == 0 else "generated"
        draw.multiline_text((8, y + 12), f"view {view_index:02d}\n{source_type}", fill="black", spacing=5)
        for col, tensor in enumerate((real[slot], generated[slot], rendered[slot])):
            image = tensor_to_pil(tensor).resize((thumb, thumb), Image.Resampling.LANCZOS)
            sheet.paste(image, (label_width + col * thumb, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], source_type: str | None = None) -> dict[str, float | int | None]:
    selected = rows if source_type is None else [r for r in rows if r["source_type"] == source_type]
    result: dict[str, float | int | None] = {"count": len(selected)}
    for pair in ("render_to_mvgen", "render_to_real", "mvgen_to_real"):
        for metric in ("psnr", "ssim", "lpips"):
            values = [r[f"{pair}_{metric}"] for r in selected if r[f"{pair}_{metric}"] is not None]
            result[f"{pair}_{metric}"] = float(np.mean(values)) if values else None
    return result


def main() -> None:
    sys.path.insert(0, "/root/vggt")
    from vggt.utils.gs_optimizer import render_gaussians
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for scene in SCENES:
        scene_out = OUT / scene
        render_dir = scene_out / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        generation = json.loads((ROOT / "mvgenmaster" / scene / "generation_metadata.json").read_text())
        view_indices = [int(generation["source_manifest_index"])] + [int(v) for v in generation["target_manifest_indices"]]
        reconstruction = torch.load(ROOT / "mvgenmaster_vggt" / scene / "vggt_reconstruction.pt", map_location="cpu")
        real_reconstruction = torch.load(ROOT / "seva" / scene / "real8" / "vggt_reconstruction.pt", map_location="cpu")
        mixed = reconstruction["images"][0].float().to(device)
        real = real_reconstruction["images"][0].float().to(device)
        if mixed.shape != real.shape or mixed.shape[0] != len(view_indices):
            raise RuntimeError(f"{scene}: image/order mismatch {mixed.shape} {real.shape} {view_indices}")
        height, width = mixed.shape[-2:]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            reconstruction["pose_enc"].float(), image_size_hw=(height, width)
        )
        extrinsics = extrinsics[0].to(device)
        intrinsics = intrinsics[0].to(device)
        gaussians = {
            key: value[0].reshape(-1, value.shape[-1]).float().to(device)
            for key, value in reconstruction["gaussians"].items()
            if torch.is_tensor(value) and value.ndim >= 2
        }
        rendered_views: list[torch.Tensor] = []
        scene_rows: list[dict] = []
        for slot, view_index in enumerate(view_indices):
            with torch.inference_mode():
                rendered = render_gaussians(
                    gaussians["xyz"], gaussians["rotation"], gaussians["scale"],
                    gaussians["opacity"], gaussians["color"], extrinsics[slot],
                    intrinsics[slot], height, width,
                )
            tensor_to_pil(rendered).save(render_dir / f"view_{view_index:02d}.png")
            rendered_views.append(rendered.detach().cpu())
            r_to_m = metrics(mixed[slot], rendered, lpips_model)
            r_to_g = metrics(real[slot], rendered, lpips_model)
            m_to_g = metrics(real[slot], mixed[slot], lpips_model)
            row = {
                "scene": scene,
                "slot": slot,
                "view_index": view_index,
                "source_type": "real" if slot == 0 else "generated",
                **{f"render_to_mvgen_{k}": v for k, v in r_to_m.items()},
                **{f"render_to_real_{k}": v for k, v in r_to_g.items()},
                **{f"mvgen_to_real_{k}": v for k, v in m_to_g.items()},
            }
            scene_rows.append(row)
            all_rows.append(row)
            del rendered
        contact_sheet(scene, view_indices, real.cpu(), mixed.cpu(), rendered_views,
                      scene_out / f"{scene}_real_mvgen_vggt_render.jpg")
        summary = {"scene": scene, **aggregate(scene_rows),
                   **{f"generated_only_{k}": v for k, v in aggregate(scene_rows, "generated").items()}}
        summaries.append(summary)
        (scene_out / "metrics.json").write_text(json.dumps({
            "protocol": "saved_mvgen_vggt_reconstruction_render_v1",
            "scene": scene,
            "view_indices": view_indices,
            "render_camera": "VGGT predicted pose_enc",
            "real_reference": "exact tensor from paired real8 VGGT input",
            "summary": summary,
            "views": scene_rows,
        }, indent=2) + "\n", encoding="utf-8")
        del reconstruction, real_reconstruction, mixed, real, gaussians, extrinsics, intrinsics
        torch.cuda.empty_cache()
        print(f"COMPLETE {scene}", flush=True)
    write_csv(OUT / "metrics_per_view.csv", all_rows)
    write_csv(OUT / "metrics_per_scene.csv", summaries)
    overall = {
        "protocol": "saved_mvgen_vggt_reconstruction_render_v1",
        "scene_count": len(SCENES),
        "view_count": len(all_rows),
        "all_views": aggregate(all_rows),
        "generated_views_only": aggregate(all_rows, "generated"),
    }
    (OUT / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
