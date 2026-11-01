#!/usr/bin/env python3
"""Aggregate the five-scene MVGenMaster timing rerun."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/root/seva_vggt/outputs/model_strategy_test5/mvgenmaster_timing_rerun")


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(a)), "mean": float(a.mean()), "min": float(a.min()),
        "max": float(a.max()), "sum": float(a.sum()), "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


rows = []
for path in sorted(ROOT.glob("*/generation_metadata.json")):
    d = json.loads(path.read_text())
    rows.append({
        "scene": d["scene_id"], "generation_seconds": d["generation_seconds"],
        "cold_process_work_seconds": d["cold_process_work_seconds"],
        "peak_allocated_gib": d["peak_allocated_gib"],
        "steps": d["steps"], "cfg": d["cfg"], "seed": d["seed"],
        "height": d["resolution_hw"][0], "width": d["resolution_hw"][1],
        "output_views": len(d["target_manifest_indices"]),
    })
with (ROOT / "timing_per_scene.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
summary = {
    "protocol": "mvgenmaster_timing_rerun_test5_v1",
    "scene_count": len(rows),
    "settings": {"steps": 50, "cfg": 2.0, "seed": 23, "source_views": 1, "generated_views": 7},
    "timing_scope": {
        "generation_seconds": "Only diffusion pipeline call with CUDA synchronization; excludes model/depth loading and file save.",
        "cold_process_work_seconds": "Fresh process from start through generation and output serialization.",
    },
    "generation_seconds": stats([r["generation_seconds"] for r in rows]),
    "cold_process_work_seconds": stats([r["cold_process_work_seconds"] for r in rows]),
    "peak_allocated_gib": stats([r["peak_allocated_gib"] for r in rows]),
}
(ROOT / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
