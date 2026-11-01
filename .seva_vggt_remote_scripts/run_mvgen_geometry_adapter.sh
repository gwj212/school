#!/usr/bin/env bash
set -euo pipefail

PY=/root/seva_vggt/candidates/env_mvgen/bin/python
SCRIPT=/root/seva_vggt/bridge/train_mvgen_geometry_adapter.py
M=/root/seva_vggt/config/test30_mvgen
OUT=/root/seva_vggt/outputs/test30_mvgen_vggt/mvgen_geometry_adapter

"$PY" "$SCRIPT" \
  --train-manifests \
    "$M/apple_350_36665_68011.json" \
    "$M/ball_378_44141_87986.json" \
    "$M/bicycle_608_95757_192662.json" \
    "$M/car_336_34863_64261.json" \
    "$M/couch_503_72319_140597.json" \
    "$M/cup_20_722_1367.json" \
    "$M/teddybear_34_1423_4310.json" \
    "$M/vase_392_47724_94792.json" \
  --val-manifests \
    "$M/bottle_601_93006_187080.json" \
    "$M/bottle_618_100690_201667.json" \
    "$M/toaster_375_42448_85012.json" \
    "$M/toaster_407_54986_106438.json" \
  --output "$OUT" \
  --steps 48 --lr 4e-6 --seed 23 \
  --pixel-weight 0.10 --edge-weight 0.02 --warp-weight 0.20 --max-timestep 350
