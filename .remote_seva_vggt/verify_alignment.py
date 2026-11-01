from pathlib import Path

import numpy as np
import torch

from seva.eval import load_img_and_K, transform_img_and_K
from metric_alignment import align_path_to_canvas


path = Path("/root/seva_vggt/datasets/co3d/cup/46_2479_7421/images/frame000000.jpg")
if not path.exists():
    import json

    manifest = json.load(open("/root/seva_vggt/config/benchmark30/cup_46_2479_7421.json"))
    path = Path(manifest["views"][0]["ground_truth"])

image, _ = load_img_and_K(str(path), None, K=None, device="cpu")
image = (image + 1.0) * 0.5
expected, _ = transform_img_and_K(image, (384, 384), mode="crop")
actual, geometry = align_path_to_canvas(path, (384, 384), "crop")
actual_tensor = torch.from_numpy(actual).permute(2, 0, 1).unsqueeze(0)
print(
    {
        "path": str(path),
        "geometry": geometry,
        "shape_equal": tuple(expected.shape) == tuple(actual_tensor.shape),
        "max_abs_error": float((expected - actual_tensor).abs().max()),
        "mean_abs_error": float((expected - actual_tensor).abs().mean()),
    }
)
