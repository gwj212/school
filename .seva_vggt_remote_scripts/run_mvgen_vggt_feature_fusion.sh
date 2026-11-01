#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/seva_vggt/outputs/test30_mvgen_vggt
M=/root/seva_vggt/config/test30_mvgen
GEN="$ROOT/mvgen"
OUT="$ROOT/mvgen_vggt_feature_fusion"
LAT="$OUT/vae_latents"
BASE=/root/seva_vggt/candidates/MVGenMaster/check_points/sd2-community-stable-diffusion-2-1
SCENES=(
  apple_350_36665_68011 ball_378_44141_87986 bicycle_608_95757_192662 car_336_34863_64261
  couch_503_72319_140597 cup_20_722_1367 teddybear_34_1423_4310 vase_392_47724_94792
  bottle_601_93006_187080 bottle_618_100690_201667 toaster_375_42448_85012 toaster_407_54986_106438
  chair_620_101264_203062 laptop_373_41558_82973 plant_598_92218_184789 toybus_121_14153_28986
)
mkdir -p "$OUT"
/root/seva_vggt/candidates/env_mvgen/bin/python /root/seva_vggt/bridge/cache_mvgen_vggt_fusion_features.py \
  --manifests "$M" --generated "$GEN" --output "$LAT" --base-model "$BASE" --scenes "${SCENES[@]}"
/root/seva_vggt/env/bin/python /root/seva_vggt/bridge/train_mvgen_vggt_feature_fusion.py \
  --manifests "$M" --generated "$GEN" --latents "$LAT" --output "$OUT/train" \
  --steps 32 --lr 2e-4 --render-size 224 --seed 23
/root/seva_vggt/env/bin/python /root/seva_vggt/bridge/reconstruct_mvgen_vggt_feature_fusion_test4.py \
  --manifests "$M" --generated "$GEN" --latents "$LAT" \
  --adapter "$OUT/train/mvgen_vggt_feature_fusion_adapter.pt" \
  --output "$OUT/reconstruction_test4"
/root/seva_vggt/env/bin/python /root/seva_vggt/bridge/evaluate_mvgen_consistency_reconstruction_test4.py \
  --adapted-generation "$GEN" \
  --adapted-reconstruction "$OUT/reconstruction_test4/scenes" \
  --adapted-branch mvgen_feature_fusion_1real7gen \
  --output "$OUT/evaluation_test4"
