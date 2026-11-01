#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/root/seva_vggt/outputs/geodiff_vggt_300")
STAGE = ROOT / "stage1"
PROTOCOL = Path("/root/seva_vggt/config/geodiff_vggt_300/selection_protocol.json")
METRICS = ("psnr", "ssim", "l1", "depth_absrel", "depth_bad20", "depth_silog")


def load(path: Path):
    return json.loads(path.read_text())


def by_scene(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene"]].append(row)
    return {
        scene: {key: float(np.mean([r[key] for r in values])) for key in METRICS}
        for scene, values in grouped.items()
    }


def paired_report(baseline, candidate, seed=20261030, draws=20000):
    base = by_scene(baseline)
    cand = by_scene(candidate)
    scenes = sorted(set(base) & set(cand))
    rng = np.random.default_rng(seed)
    report = {}
    for key in METRICS:
        delta = np.asarray([cand[s][key] - base[s][key] for s in scenes], dtype=np.float64)
        samples = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(1)
        report[key] = {
            "baseline_mean": float(np.mean([base[s][key] for s in scenes])),
            "candidate_mean": float(np.mean([cand[s][key] for s in scenes])),
            "delta": float(delta.mean()),
            "relative_percent": float(100.0 * delta.mean() / max(abs(np.mean([base[s][key] for s in scenes])), 1e-12)),
            "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "improved_scenes": int(np.sum(delta > 0 if key in ("psnr", "ssim") else delta < 0)),
            "scene_count": len(scenes),
        }
    return report


def main():
    summary = load(STAGE / "summary.json")
    baseline = load(STAGE / "validation_baseline.json")["rows"]
    epochs = {epoch: load(STAGE / f"validation_epoch_{epoch}.json") for epoch in (1, 2, 3)}
    checkpoint = torch.load(STAGE / "best.pt", map_location="cpu")
    best_epoch = int(checkpoint["epoch"])
    best_rows = epochs[best_epoch]["rows"]

    state = checkpoint["bridge"]
    trace = load(STAGE / "training_trace.json")
    grad = np.asarray([row["grad_norm"] for row in trace], dtype=np.float64)
    visibility = np.asarray([row["visibility_mean"] for row in trace], dtype=np.float64)
    nonfinite_rows = [row for row in trace if row.get("nonfinite_gradients")]

    per_category = {}
    categories = sorted({row["scene"].split("_", 1)[0] for row in baseline})
    for category in categories:
        b = [row for row in baseline if row["scene"].startswith(category + "_")]
        c = [row for row in best_rows if row["scene"].startswith(category + "_")]
        per_category[category] = {
            key: float(np.mean([r[key] for r in c]) - np.mean([r[key] for r in b]))
            for key in METRICS
        }

    protocol = load(PROTOCOL)
    blind_status = []
    for scene in protocol["blind_test_scenes"]:
        path = ROOT / "cache" / scene / "mvgen_multiscale_features.pt"
        blind_status.append({"scene": scene, "cache_exists": path.is_file()})

    layer_logits = state["layer_gates"].float()
    camera_logit = state["camera_gate"].float()
    result = {
        "summary": summary,
        "best_epoch": best_epoch,
        "epochs": {str(e): epochs[e]["metrics"] for e in epochs},
        "paired_scene_bootstrap": paired_report(baseline, best_rows),
        "per_category_delta": per_category,
        "checkpoint_gates": {
            "layer_logits": layer_logits.tolist(),
            "layer_sigmoid": torch.sigmoid(layer_logits).tolist(),
            "camera_logit": float(camera_logit),
            "camera_sigmoid": float(torch.sigmoid(camera_logit)),
            "visibility_final_bias_mean": float(state["visibility.3.bias"].float().mean()),
            "visibility_final_weight_norm": float(state["visibility.3.weight"].float().norm()),
        },
        "training_trace": {
            "rows": len(trace),
            "nonfinite_rows": len(nonfinite_rows),
            "grad_norm_min_median_p95_max": [
                float(grad.min()), float(np.median(grad)), float(np.quantile(grad, 0.95)), float(grad.max())
            ],
            "preclip_grad_over_1": int(np.sum(grad > 1.0)),
            "visibility_first_last": [float(visibility[0]), float(visibility[-1])],
            "visibility_epoch_mean": {
                str(epoch): float(np.mean([r["visibility_mean"] for r in trace if r["epoch"] == epoch]))
                for epoch in (1, 2, 3)
            },
        },
        "blind_isolation": {
            "declared_scenes": len(blind_status),
            "cached_scenes": sum(item["cache_exists"] for item in blind_status),
            "details": blind_status,
            "blind_output_exists": (ROOT / "blind30").exists() and any((ROOT / "blind30").iterdir()),
        },
    }
    out = STAGE / "audit_stage1.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
