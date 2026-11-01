#!/usr/bin/env python3
"""Summarize a fixed-total SEVA/VGGT ablation and render one shared-view montage."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BRANCH_RE = re.compile(r"seva_(\d+)real_(\d+)gen$")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_number(data: dict, *paths: tuple[str, ...]):
    for path in paths:
        value = data
        try:
            for key in path:
                value = value[key]
        except (KeyError, TypeError):
            continue
        if isinstance(value, (int, float)):
            return value
    return None


def collect(root: Path) -> list[dict]:
    rows = []
    for branch in root.iterdir():
        match = BRANCH_RE.fullmatch(branch.name) if branch.is_dir() else None
        if not match:
            continue
        image_metrics = read_json(branch / "image_metrics.json")
        metrics = read_json(branch / "metrics.json")
        metadata_path = root / f"run_metadata_{branch.name}.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        comparison_path = root / f"comparison_real8_vs_{branch.name}.json"
        comparison = read_json(comparison_path) if comparison_path.exists() else {}
        rows.append(
            {
                "real_count": int(match.group(1)),
                "generated_count": int(match.group(2)),
                "branch": branch.name,
                "real_indices": metadata.get("input_indices"),
                "generated_indices": metadata.get("target_indices"),
                "image_psnr": first_number(image_metrics, ("mean_psnr",)),
                "image_ssim": first_number(image_metrics, ("mean_ssim",)),
                "object_psnr": first_number(image_metrics, ("mean_object_psnr",)),
                "background_psnr": first_number(image_metrics, ("mean_background_psnr",)),
                "normalized_chamfer_vs_real8": first_number(
                    comparison, ("symmetric_chamfer_l1_normalized",)
                ),
                "raw_chamfer_vs_real8": first_number(
                    comparison, ("symmetric_chamfer_l1",)
                ),
                "opacity_mean": first_number(metrics, ("reconstruction", "opacity", "mean")),
                "opacity_p50": first_number(metrics, ("reconstruction", "opacity", "p50")),
                "bbox_diagonal": first_number(metrics, ("reconstruction", "bbox_diagonal")),
            }
        )
    return sorted(rows, key=lambda row: row["real_count"])


def save_tables(root: Path, rows: list[dict]) -> None:
    payload = {
        "experiment": "fixed total of 8 camera slots; vary real/SEVA ratio",
        "reference": "real8",
        "warning": "Image metrics average different generated target views; normalized Chamfer is directly comparable across rows.",
        "rows": rows,
    }
    (root / "fixed8_ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(rows[0].keys()) if rows else []
    with (root / "fixed8_ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_view_image(branch: Path, view_index: int) -> Path | None:
    candidates = [
        branch / "generated_only" / f"view_{view_index:02d}_generated.jpg",
        branch / "generated_only" / f"view_{view_index}_generated.jpg",
        branch / "generated_only" / f"view_{view_index:02d}.png",
        branch / "generated_only" / f"view_{view_index}.png",
        branch / "images" / f"view_{view_index:02d}_generated.jpg",
        branch / "images" / f"view_{view_index}_generated.jpg",
        branch / "images" / f"view_{view_index:02d}_generated.png",
        branch / "images" / f"view_{view_index}_generated.png",
        branch / "images" / f"view_{view_index:02d}_real.jpg",
        branch / "images" / f"view_{view_index}_real.jpg",
    ]
    return next((path for path in candidates if path.exists()), None)


def make_montage(root: Path, rows: list[dict], view_index: int) -> Path | None:
    tiles: list[tuple[str, Path]] = []
    reference = find_view_image(root / "real8", view_index)
    if reference:
        tiles.append(("Real target", reference))
    for row in rows:
        image_path = find_view_image(root / row["branch"], view_index)
        if image_path:
            tiles.append((f"{row['real_count']} real + {row['generated_count']} SEVA", image_path))
    if not tiles:
        return None

    thumb_size = (360, 360)
    label_height = 42
    columns = 4
    rows_count = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_size[0], rows_count * (thumb_size[1] + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    for index, (label, path) in enumerate(tiles):
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * (thumb_size[1] + label_height)
        canvas.paste(image, (x + (thumb_size[0] - image.width) // 2, y + (thumb_size[1] - image.height) // 2))
        draw.text((x + 8, y + thumb_size[1] + 8), label, fill="black", font=font)
    output = root / f"view_{view_index:02d}_across_real_seva_ratios.jpg"
    canvas.save(output, quality=95)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--view", type=int, default=17)
    args = parser.parse_args()
    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"No seva_*real_*gen branches found under {args.root}")
    save_tables(args.root, rows)
    montage = make_montage(args.root, rows, args.view)
    print(json.dumps({"rows": len(rows), "montage": str(montage) if montage else None}, indent=2))


if __name__ == "__main__":
    main()
