#!/usr/bin/env python3
"""Paired MVGen-vs-SEVA comparison against real-image VGGT depth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("absrel", "rmse_norm", "silog", "bad20")
METHODS = ("mvgen_baseline", "seva")


def bootstrap(scene_deltas: np.ndarray, seed: int = 20261030, draws: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    sampled = scene_deltas[rng.integers(0, len(scene_deltas), size=(draws, len(scene_deltas)))].mean(1)
    return {
        "mean_delta_mvgen_minus_seva": float(scene_deltas.mean()),
        "ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
        "mvgen_better_scenes": int((scene_deltas < 0).sum()),
        "seva_better_scenes": int((scene_deltas > 0).sum()),
        "ties": int((scene_deltas == 0).sum()),
        "scene_count": int(len(scene_deltas)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8-sig")))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] in METHODS:
            grouped[(row["alignment"], row["region"])].append(row)

    result = {
        "comparison": "MVGen-generated RGB -> frozen VGGT depth versus SEVA-generated RGB -> the same frozen VGGT depth; reference is the same scene's eight real GT RGB views -> VGGT depth",
        "warning": "GT-image-derived VGGT depth proxy, not sensor ground-truth depth",
        "views_per_method": 210,
        "results": {},
    }
    for (alignment, region), subset in sorted(grouped.items()):
        key = f"{alignment}/{region}"
        method_means = {}
        by_scene_method: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in subset:
            by_scene_method[(row["scene"], row["method"])].append(row)
        for method in METHODS:
            method_rows = [row for row in subset if row["method"] == method]
            method_means[method] = {
                metric: float(np.nanmean([float(row[metric]) for row in method_rows]))
                for metric in METRICS
            }
        block = {"means": method_means, "mvgen_relative_percent_vs_seva": {}, "paired_scene_bootstrap": {}}
        scenes = sorted({row["scene"] for row in subset})
        for metric in METRICS:
            seva = method_means["seva"][metric]
            mvgen = method_means["mvgen_baseline"][metric]
            block["mvgen_relative_percent_vs_seva"][metric] = 100.0 * (mvgen - seva) / seva
            deltas = []
            for scene in scenes:
                mv = np.nanmean([float(row[metric]) for row in by_scene_method[(scene, "mvgen_baseline")]])
                sv = np.nanmean([float(row[metric]) for row in by_scene_method[(scene, "seva")]])
                if np.isfinite(mv) and np.isfinite(sv):
                    deltas.append(mv - sv)
            block["paired_scene_bootstrap"][metric] = bootstrap(np.asarray(deltas))
        result["results"][key] = block

    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
