#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/seva_vggt"
NVJIT="${ROOT}/env/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
export LD_LIBRARY_PATH="${NVJIT}:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
exec "${ROOT}/env/bin/python" "$@"
