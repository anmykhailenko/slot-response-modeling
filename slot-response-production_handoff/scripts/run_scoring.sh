#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_scoring.sh <PT:YYYYMMDD>" >&2
  exit 1
fi

PT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[RUN_SCORING] project_root=${PROJECT_ROOT}"
echo "[RUN_SCORING] pt=${PT}"
echo "[RUN_SCORING] config=${PROJECT_ROOT}/configs/response_scoring.yaml"

cd "${PROJECT_ROOT}"
python src/modeling/run_phase1_response_scoring.py --config configs/response_scoring.yaml --pt "${PT}"
