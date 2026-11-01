#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/seva_vggt/outputs/test30_mvgen_vggt
BASE="$ROOT/mvgen_vggt_feature_fusion"
OUT="$ROOT/mvgen_vggt_geometry_token_fusion"
M=/root/seva_vggt/config/test30_mvgen
PY=/root/seva_vggt/env/bin/python
mkdir -p "$OUT"
$PY /root/seva_vggt/bridge/train_mvgen_vggt_geometry_token_fusion.py --manifests "$M" --generated "$ROOT/mvgen" --latents "$BASE/vae_latents" --output "$OUT/train" --steps 20 --lr 8e-5 --render-size 192 --seed 23
$PY /root/seva_vggt/bridge/reconstruct_mvgen_vggt_geometry_token_fusion_test4.py --manifests "$M" --generated "$ROOT/mvgen" --latents "$BASE/vae_latents" --adapter "$OUT/train/mvgen_vggt_geometry_token_fusion.pt" --output "$OUT/reconstruction_test4"
$PY /root/seva_vggt/bridge/evaluate_mvgen_consistency_reconstruction_test4.py --adapted-generation "$ROOT/mvgen" --adapted-reconstruction "$OUT/reconstruction_test4/scenes" --adapted-branch mvgen_geometry_token_fusion_1real7gen --output "$OUT/evaluation_test4"
