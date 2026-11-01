#!/usr/bin/env bash
set -euo pipefail

OUT=/root/seva_vggt/outputs/test30_mvgen_vggt/mvgen_geometry_adapter_test4
PY=/root/seva_vggt/candidates/env_mvgen/bin/python
RUN=/root/seva_vggt/bridge/run_mvgen_adapter.py
REPO=/root/seva_vggt/candidates/MVGenMaster
BASE=$REPO/check_points/sd2-community-stable-diffusion-2-1
ADAPTER=/root/seva_vggt/outputs/test30_mvgen_vggt/mvgen_geometry_adapter/mvgen_add_conv_in_adapter.pt
mkdir -p "$OUT"

for scene in \
  chair_620_101264_203062 \
  laptop_373_41558_82973 \
  plant_598_92218_184789 \
  toybus_121_14153_28986
do
  "$PY" "$RUN" \
    --repo "$REPO" \
    --manifest "/root/seva_vggt/config/test30_mvgen/$scene.json" \
    --output "$OUT/$scene" \
    --base-model "$BASE" \
    --target-indices 1,2,3,4,5,6,7 \
    --seed 23 --steps 50 --cfg 2 \
    --adapter "$ADAPTER"
done
