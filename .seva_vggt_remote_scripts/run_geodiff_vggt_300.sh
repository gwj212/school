#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/seva_vggt
PY="$ROOT/scripts/python.sh"
MVGEN_PY="$ROOT/candidates/env_mvgen/bin/python"
BRIDGE="$ROOT/bridge"
CONFIG="$ROOT/config/geodiff_vggt_300"
CACHE="$ROOT/outputs/geodiff_vggt_300/cache"
TRAIN_OUT="$ROOT/outputs/geodiff_vggt_300/stage1"
SMOKE_OUT="$ROOT/outputs/geodiff_vggt_300/stage1_smoke"
BLIND_OUT="$ROOT/outputs/geodiff_vggt_300/blind30"
BASE_MODEL="$ROOT/candidates/MVGenMaster/check_points/sd2-community-stable-diffusion-2-1"

mkdir -p "$CACHE" "$SMOKE_OUT" "$TRAIN_OUT" "$BLIND_OUT"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "BLOCKED: NVIDIA device is not available; split and CPU depth reports are safe, GPU cache/training is not." >&2
  exit 42
fi

"$MVGEN_PY" "$BRIDGE/cache_mvgen_multiscale_features_batch.py" \
  --protocol "$CONFIG/selection_protocol.json" \
  --manifest-root "$CONFIG" \
  --blind-manifest-root /root/seva_vggt/config/test30_mvgen \
  --cache-root "$CACHE" \
  --base-model "$BASE_MODEL" \
  --steps 12 --cfg 2.0 \
  --splits train validation

# Exercise one complete forward/backward/validation pass before committing to
# the 300-scene run.  Teacher depth is cached and reused by the full training.
"$PY" "$BRIDGE/train_geodiff_vggt_stage1.py" \
  --protocol "$CONFIG/selection_protocol.json" \
  --manifest-root "$CONFIG" \
  --cache-root "$CACHE" \
  --output "$SMOKE_OUT" \
  --epochs 1 --lr 8e-5 --render-size 192 \
  --max-train-scenes 1 --max-val-scenes 1

"$PY" "$BRIDGE/train_geodiff_vggt_stage1.py" \
  --protocol "$CONFIG/selection_protocol.json" \
  --manifest-root "$CONFIG" \
  --cache-root "$CACHE" \
  --output "$TRAIN_OUT" \
  --epochs 3 --lr 8e-5 --render-size 192

# Blind data is first accessed only after validation-based model selection.
"$MVGEN_PY" "$BRIDGE/cache_mvgen_multiscale_features_batch.py" \
  --protocol "$CONFIG/selection_protocol.json" \
  --manifest-root "$CONFIG" \
  --blind-manifest-root /root/seva_vggt/config/test30_mvgen \
  --cache-root "$CACHE" \
  --base-model "$BASE_MODEL" \
  --steps 12 --cfg 2.0 \
  --splits blind_test

"$PY" "$BRIDGE/evaluate_geodiff_vggt_blind30.py" \
  --checkpoint "$TRAIN_OUT/best.pt" \
  --protocol "$CONFIG/selection_protocol.json" \
  --manifests /root/seva_vggt/config/test30_mvgen \
  --cache-root "$CACHE" \
  --output "$BLIND_OUT" \
  --render-size 192

