#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/seva_vggt/outputs/joint_finetune_small
OUT="$ROOT/adapted_mvgen_trainval6"
MANIFESTS=/root/seva_vggt/config/joint_small_train
ADAPTER="$ROOT/adapter_4train_2val/mvgen_add_conv_in_adapter.pt"
BASE=/root/seva_vggt/candidates/MVGenMaster/check_points/sd2-community-stable-diffusion-2-1
PY=/root/seva_vggt/candidates/env_mvgen/bin/python
SCRIPT=/root/seva_vggt/bridge/run_mvgen_adapter.py

mkdir -p "$OUT"
for scene in \
  cup_46_2479_7421 \
  laptop_386_46021_91916 \
  plant_529_76885_148624 \
  bicycle_404_53644_104645 \
  ball_394_48737_96400 \
  apple_484_69426_135397
do
  if [[ -f "$OUT/$scene/generation_metadata.json" ]]; then
    echo "SKIP $scene"
    continue
  fi
  "$PY" "$SCRIPT" \
    --repo /root/seva_vggt/candidates/MVGenMaster \
    --manifest "$MANIFESTS/$scene.json" \
    --output "$OUT/$scene" \
    --base-model "$BASE" \
    --target-indices 1,2,3,4,5,6,7 \
    --seed 23 --steps 50 --cfg 2 \
    --adapter "$ADAPTER"
done
