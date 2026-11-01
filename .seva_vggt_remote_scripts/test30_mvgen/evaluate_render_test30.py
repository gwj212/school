#!/usr/bin/env python3
"""Render both saved branches on the exact 518 canvas and compute paired image metrics."""

from __future__ import annotations

import argparse, csv, json, math, sys, time
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


def finite(x): return float(x) if math.isfinite(float(x)) else None


def image_metrics(reference: torch.Tensor, prediction: torch.Tensor, perceptual) -> dict:
    ref = reference.float().clamp(0, 1); pred = prediction.float().clamp(0, 1)
    if ref.shape != pred.shape: raise RuntimeError(f"metric size mismatch: {tuple(ref.shape)} vs {tuple(pred.shape)}")
    mse = torch.mean((ref - pred) ** 2).item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    ref_np = ref.permute(1, 2, 0).detach().cpu().numpy()
    pred_np = pred.permute(1, 2, 0).detach().cpu().numpy()
    ssim = structural_similarity(ref_np, pred_np, channel_axis=2, data_range=1.0)
    with torch.inference_mode(): distance = perceptual(ref[None] * 2 - 1, pred[None] * 2 - 1)
    return {"psnr": finite(psnr), "ssim": finite(ssim), "lpips": finite(distance.flatten()[0].item()), "mse": mse}


def pil(t: torch.Tensor):
    return Image.fromarray((t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8))


def write_csv(path: Path, rows: list[dict]):
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def aggregate(rows: list[dict], output: Path):
    ids = {"scene", "category", "slot", "source_type", "canvas_height", "canvas_width", "same_size_validated"}
    summaries = []
    scopes = [("all_views", rows), ("generated_views_only", [r for r in rows if r["source_type"] == "generated"])]
    for scope, subset in scopes:
        for key in sorted(set().union(*(r.keys() for r in subset)) - ids):
            vals = [float(r[key]) for r in subset if r.get(key) is not None and isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))]
            if not vals: continue
            a = np.asarray(vals)
            summaries.append({"scope": scope, "metric": key, "count": len(a), "mean": a.mean(), "min": a.min(),
                              "max": a.max(), "sum": a.sum(), "median": np.median(a),
                              "std": a.std(ddof=1) if len(a) > 1 else 0.0})
    write_csv(output / "metric_summary.csv", summaries)
    return summaries


def render_branch(reconstruction, output: Path, perceptual, references: dict[str, torch.Tensor], device):
    from vggt.utils.gs_optimizer import render_gaussians
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = reconstruction["images"][0].float()
    h, w = images.shape[-2:]
    ext, intr = pose_encoding_to_extri_intri(reconstruction["pose_enc"].float(), (h, w))
    ext, intr = ext[0].to(device), intr[0].to(device)
    g = {k: v[0].reshape(-1, v.shape[-1]).float().to(device) for k, v in reconstruction["gaussians"].items()
         if torch.is_tensor(v) and v.ndim >= 2}
    output.mkdir(parents=True, exist_ok=True)
    rendered, metric_rows = [], []
    for slot in range(8):
        with torch.inference_mode():
            rd = render_gaussians(g["xyz"], g["rotation"], g["scale"], g["opacity"], g["color"], ext[slot], intr[slot], h, w)
        pil(rd).save(output / f"view_{slot:02d}_render.png")
        rendered.append(rd.detach().cpu())
        metrics = {}
        for name, tensor in references.items():
            m = image_metrics(tensor[slot].to(device), rd, perceptual)
            metrics.update({f"{name}_{k}": v for k, v in m.items()})
        metric_rows.append(metrics)
        del rd
    del g, ext, intr; torch.cuda.empty_cache()
    return images, rendered, metric_rows


def make_sheet(scene: str, real: torch.Tensor, real_render: list[torch.Tensor], mv: torch.Tensor,
               mv_render: list[torch.Tensor], path: Path):
    thumb, label, header = 180, 90, 45
    titles = ("Test GT (real8)", "real8 render", "MVGen input", "MVGen render")
    sheet = Image.new("RGB", (label + 4 * thumb, header + 8 * thumb), "white"); draw = ImageDraw.Draw(sheet)
    for c, title in enumerate(titles): draw.text((label + c * thumb + 10, 15), title, fill="black")
    for i in range(8):
        y = header + i * thumb; draw.multiline_text((5, y + 8), f"view {i:02d}\n{'real' if i == 0 else 'generated'}", fill="black")
        for c, tensor in enumerate((real[i], real_render[i], mv[i], mv_render[i])):
            sheet.paste(pil(tensor).resize((thumb, thumb), Image.Resampling.LANCZOS), (label + c * thumb, y))
    sheet.save(path, quality=94)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--mvgen-root", type=Path, required=True)
    ap.add_argument("--reconstruction-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/root/vggt")
    device = torch.device("cuda:0"); perceptual = lpips.LPIPS(net="alex").eval().to(device)
    scenes = json.loads((args.manifests / "selection_protocol.json").read_text())["scenes"]
    all_rows, scene_rows, timing_rows = [], [], []
    for scene in scenes:
        scene_started = time.perf_counter(); category = scene.split("_")[0]
        root = args.reconstruction_root / "scenes" / scene
        real_rec = torch.load(root / "real8" / "vggt_reconstruction.pt", map_location="cpu")
        real_ref = real_rec["images"][0].float()
        real_inputs, real_render, real_metrics = render_branch(real_rec, root / "real8" / "renders", perceptual,
                                                                {"real8_render_to_real": real_ref}, device)
        del real_rec
        mv_rec = torch.load(root / "mvgen_1real7gen" / "vggt_reconstruction.pt", map_location="cpu")
        mv_ref = mv_rec["images"][0].float()
        if real_ref.shape != mv_ref.shape or tuple(real_ref.shape[-2:]) != (518, 518):
            raise RuntimeError(f"{scene}: expected identical [8,3,518,518], got {real_ref.shape} and {mv_ref.shape}")
        mv_inputs, mv_render, mv_metrics = render_branch(mv_rec, root / "mvgen_1real7gen" / "renders", perceptual,
            {"mv_render_to_generated_input": mv_ref, "mv_render_to_real_gt": real_ref}, device)
        del mv_rec
        rows = []
        for i in range(8):
            direct = image_metrics(real_ref[i].to(device), mv_ref[i].to(device), perceptual)
            row = {"scene": scene, "category": category, "slot": i, "source_type": "real" if i == 0 else "generated",
                   "canvas_height": 518, "canvas_width": 518, "same_size_validated": 1,
                   **real_metrics[i], **mv_metrics[i], **{f"mv_generated_to_real_gt_{k}": v for k, v in direct.items()}}
            rows.append(row); all_rows.append(row)
        write_csv(args.output / f"{scene}_metrics_per_view.csv", rows)
        make_sheet(scene, real_ref, real_render, mv_ref, mv_render, args.output / f"{scene}_comparison.jpg")
        numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in {"slot", "canvas_height", "canvas_width", "same_size_validated"}]
        scene_row = {"scene": scene, "category": category}
        for key in numeric:
            vals = [float(r[key]) for r in rows if r[key] is not None and math.isfinite(float(r[key]))]
            scene_row[key] = float(np.mean(vals)) if vals else None
            gen = [float(r[key]) for r in rows if r["source_type"] == "generated" and r[key] is not None and math.isfinite(float(r[key]))]
            scene_row[f"generated_only.{key}"] = float(np.mean(gen)) if gen else None
        scene_rows.append(scene_row)
        gen_meta = json.loads((args.mvgen_root / scene / "generation_metadata.json").read_text())
        real_meta = json.loads((root / "real8" / "metrics.json").read_text())
        mv_meta = json.loads((root / "mvgen_1real7gen" / "metrics.json").read_text())
        eval_seconds = time.perf_counter() - scene_started
        warm_mv = gen_meta["timing_seconds"]["warm_scene_total"] + mv_meta["timing_seconds"]["warm_branch_total"] + eval_seconds
        timing_rows.append({"scene": scene, "mvgen_depth_s": gen_meta["timing_seconds"]["depth_estimation"],
            "mvgen_diffusion_s": gen_meta["timing_seconds"]["diffusion_generation"],
            "mvgen_warm_scene_s": gen_meta["timing_seconds"]["warm_scene_total"],
            "real8_vggt_warm_s": real_meta["timing_seconds"]["warm_branch_total"],
            "mv_vggt_preprocess_s": mv_meta["timing_seconds"]["input_preprocess"],
            "mv_vggt_inference_s": mv_meta["timing_seconds"]["model_inference"],
            "mv_vggt_serialize_s": mv_meta["timing_seconds"]["serialize_reconstruction"],
            "mv_pointcloud_export_s": mv_meta["timing_seconds"]["point_cloud_export"],
            "mv_vggt_warm_total_s": mv_meta["timing_seconds"]["warm_branch_total"],
            "render_and_metrics_both_branches_s": eval_seconds,
            "mv_complete_warm_pipeline_s": warm_mv,
            "mv_complete_cold_pipeline_estimate_s": warm_mv + gen_meta["shared_model_load_seconds"] + mv_meta["shared_model_load_seconds"],
            "mv_complete_amortized_30_s": warm_mv + (gen_meta["shared_model_load_seconds"] + mv_meta["shared_model_load_seconds"]) / len(scenes)})
        (args.output / f"{scene}_protocol.json").write_text(json.dumps({"scene": scene, "canvas": [518, 518],
            "size_validation": "Both references are tensors saved by the two VGGT preprocessing passes; every render is generated directly at the same 518x518 camera canvas.",
            "rows": rows}, indent=2, ensure_ascii=False) + "\n")
        print(f"COMPLETE {scene} eval={eval_seconds:.2f}s", flush=True)
        del real_ref, mv_ref, real_render, mv_render, real_inputs, mv_inputs; torch.cuda.empty_cache()
    write_csv(args.output / "metrics_per_view.csv", all_rows); write_csv(args.output / "metrics_per_scene.csv", scene_rows); write_csv(args.output / "timing_per_scene.csv", timing_rows)
    summaries = aggregate(all_rows, args.output)
    timing_summary = []
    for key in timing_rows[0]:
        if key == "scene": continue
        a = np.asarray([r[key] for r in timing_rows], dtype=float)
        timing_summary.append({"metric": key, "count": len(a), "mean": a.mean(), "min": a.min(), "max": a.max(),
                               "sum": a.sum(), "median": np.median(a), "std": a.std(ddof=1)})
    write_csv(args.output / "timing_summary.csv", timing_summary)
    protocol = {"experiment": "held-out my_test_multiview test30: real8 versus MVGenMaster 1real+7gen -> unchanged VGGT-G",
        "scene_count": 30, "input_canvas": [518, 518], "same_size_guarantee": "references and renders are all exact 518x518 tensors; no post-hoc stretch is used for metrics",
        "metric_definitions": {"PSNR": "RGB [0,1], all pixels and channels, -10*log10(mean squared error)",
            "SSIM": "skimage structural_similarity, channel_axis=2, data_range=1.0, full 518x518 RGB",
            "LPIPS": "LPIPS AlexNet, RGB mapped from [0,1] to [-1,1], full 518x518 image",
            "MSE": "mean squared RGB error on [0,1]"},
        "metric_pairs": {"real8_render_to_real": "real8 reconstruction render versus its true test image",
            "mv_render_to_generated_input": "MVGen branch render versus the exact real/generated image given to VGGT",
            "mv_render_to_real_gt": "MVGen branch render versus the paired true test image",
            "mv_generated_to_real_gt": "MVGen generated input itself versus the paired true test image"},
        "timing": {"warm": "per-scene work after models are resident", "cold_estimate": "warm plus one MVGen/DepthPro load and one VGGT load",
            "amortized_30": "warm plus both shared model loads divided by 30"}}
    (args.output / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"metric_summary_rows": len(summaries), "scene_rows": len(scene_rows), "view_rows": len(all_rows)}, indent=2))


if __name__ == "__main__": main()
