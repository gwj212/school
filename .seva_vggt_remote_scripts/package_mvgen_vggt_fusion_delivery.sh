#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/seva_vggt/outputs/test30_mvgen_vggt
DEL="$ROOT/mvgen_vggt_fusion_delivery"
rm -rf "$DEL"
mkdir -p "$DEL/shallow_feature_fusion/train" "$DEL/shallow_feature_fusion/evaluation" \
         "$DEL/geometry_token_fusion/train" "$DEL/geometry_token_fusion/evaluation" "$DEL/code"
cp "$ROOT/mvgen_vggt_feature_fusion/train/"{mvgen_vggt_feature_fusion_adapter.pt,training_summary.json,training_trace.json} "$DEL/shallow_feature_fusion/train/"
cp -a "$ROOT/mvgen_vggt_feature_fusion/evaluation_test4/"{comparisons,renders,metrics_per_view.csv,geometry_vs_real8.csv,summary.json,report_zh.md} "$DEL/shallow_feature_fusion/evaluation/"
cp "$ROOT/mvgen_vggt_geometry_token_fusion/train/"{mvgen_vggt_geometry_token_fusion.pt,training_summary.json,training_trace.json} "$DEL/geometry_token_fusion/train/"
cp -a "$ROOT/mvgen_vggt_geometry_token_fusion/evaluation_test4/"{comparisons,renders,metrics_per_view.csv,geometry_vs_real8.csv,summary.json,report_zh.md} "$DEL/geometry_token_fusion/evaluation/"
cp /root/seva_vggt/bridge/{cache_mvgen_vggt_fusion_features.py,train_mvgen_vggt_feature_fusion.py,reconstruct_mvgen_vggt_feature_fusion_test4.py,train_mvgen_vggt_geometry_token_fusion.py,reconstruct_mvgen_vggt_geometry_token_fusion_test4.py,run_mvgen_vggt_feature_fusion.sh,run_mvgen_vggt_geometry_token_fusion.sh} "$DEL/code/"
tar -C "$ROOT" --zstd -cf "$ROOT/mvgen_vggt_fusion_delivery.tar.zst" mvgen_vggt_fusion_delivery
sha256sum "$ROOT/mvgen_vggt_fusion_delivery.tar.zst"
du -sh "$DEL" "$ROOT/mvgen_vggt_fusion_delivery.tar.zst"
