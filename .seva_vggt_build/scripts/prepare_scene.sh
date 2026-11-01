#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/seva_vggt"
SCENE="${1:?usage: prepare_scene.sh SCENE [interpolated20|exact20_from_completion]}"
MODE="${2:-interpolated20}"

"${ROOT}/scripts/python.sh" "${ROOT}/bridge/dataset_adapter.py" \
  "${SCENE}" \
  --mode "${MODE}" \
  --output "${ROOT}/config/scenes/${SCENE}_${MODE}.json"
