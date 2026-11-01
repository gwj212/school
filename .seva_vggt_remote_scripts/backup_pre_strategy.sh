#!/usr/bin/env bash
set -euo pipefail

backup_dir=/root/seva_vggt/backups/pre_model_strategy_20261014_2055
mkdir -p "$backup_dir"

tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  -czf "$backup_dir/seva_vggt_code_config.tar.gz" -C /root \
  seva_vggt/bridge seva_vggt/source seva_vggt/scripts seva_vggt/config \
  seva_vggt/seva/seva seva_vggt/seva/demo.py seva_vggt/seva/demo_gr.py

tar --exclude='__pycache__' --exclude='*.pyc' --exclude='ckpts' \
  --exclude='data' --exclude='co3d' --exclude='output' \
  --exclude='co3d_completion_train' --exclude='co3d_completion_test' \
  --exclude='my_train_20v' --exclude='my_test_multiview' \
  --exclude='mvp_background_validation' \
  -czf "$backup_dir/vggt_code.tar.gz" -C /root \
  vggt/vggt vggt/tools vggt/eval.py vggt/eval_nvs.py \
  vggt/train_nogt.py vggt/build_co3d_completion.sh

git -C /root/vggt status --short > "$backup_dir/vggt_git_status.txt"
git -C /root/vggt diff --binary > "$backup_dir/vggt_worktree.patch"

find /root/seva_vggt/bridge /root/seva_vggt/source /root/seva_vggt/scripts \
  /root/seva_vggt/config /root/vggt/vggt /root/vggt/tools \
  -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
  | sort -z | xargs -0 sha256sum > "$backup_dir/file_manifest_sha256.txt"

{
  echo "created_utc=$(date -u +%FT%TZ)"
  echo "seva_vggt_root=/root/seva_vggt"
  echo "vggt_commit=$(git -C /root/vggt rev-parse HEAD)"
  echo "benchmark_manifest_count=$(find /root/seva_vggt/config/benchmark30 -maxdepth 1 -name '*.json' | wc -l)"
  echo "cfg_sweep_source=/root/seva_vggt/outputs/cfg_sweep30_t8_s12/cfg_2"
  echo "depth_diagnosis=/root/seva_vggt/outputs/depth_ghost30_cfg2_t8_s12"
} > "$backup_dir/README.txt"

sha256sum "$backup_dir"/*.tar.gz "$backup_dir/vggt_worktree.patch" \
  > "$backup_dir/BACKUP_SHA256SUMS.txt"
ls -lh "$backup_dir"
