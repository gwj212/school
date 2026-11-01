#!/usr/bin/env python3
"""Create a labeled thumbnail sheet for reconstruction input images."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def build(directory: Path, output: Path, columns: int = 5) -> None:
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    tile_w, tile_h, label_h = 190, 190, 24
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (tile_w - 8, tile_h - 8), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_w
        y = (index // columns) * (tile_h + label_h)
        canvas.paste(image, (x + (tile_w - image.width) // 2, y + (tile_h - image.height) // 2))
        draw.text((x + 5, y + tile_h + 4), path.stem, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.directory, args.output)


if __name__ == "__main__":
    main()
