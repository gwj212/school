#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/seva_vggt"
MANIFEST="${1:?usage: run_scene.sh MANIFEST [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-}"
GPU="${GPU:-0}"
TARGETS="${TARGETS:-8}"
INPUTS="${INPUTS:-}"
TOTAL="${TOTAL:-}"
VGGT_SIDE="${VGGT_SIDE:-518}"

ARGS=("${MANIFEST}" --device cuda:0 --target-count "${TARGETS}" --vggt-max-side "${VGGT_SIDE}")
if [[ -n "${INPUTS}" || -n "${TOTAL}" ]]; then
  if [[ -z "${INPUTS}" || -z "${TOTAL}" ]]; then
    echo "INPUTS and TOTAL must be set together" >&2
    exit 2
  fi
  ARGS+=(--input-count "${INPUTS}" --total-count "${TOTAL}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${ROOT}/scripts/python.sh" \
  "${ROOT}/bridge/run_pipeline.py" "${ARGS[@]}"
