#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_monitoring.sh <PT:YYYYMMDD> [REFERENCE_PT:YYYYMMDD]" >&2
  exit 1
fi

PT="$1"
REFERENCE_PT="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[RUN_MONITORING] project_root=${PROJECT_ROOT}"
echo "[RUN_MONITORING] pt=${PT}"
if [[ -n "${REFERENCE_PT}" ]]; then
  echo "[RUN_MONITORING] reference_pt=${REFERENCE_PT}"
else
  echo "[RUN_MONITORING] reference_pt=auto_previous_available_partition"
fi
echo "[RUN_MONITORING] config=${PROJECT_ROOT}/configs/response_monitoring.yaml"

cd "${PROJECT_ROOT}"
if [[ -n "${REFERENCE_PT}" ]]; then
  python src/monitoring/run_response_monitoring.py --config configs/response_monitoring.yaml --pt "${PT}" --reference-pt "${REFERENCE_PT}"
else
  python src/monitoring/run_response_monitoring.py --config configs/response_monitoring.yaml --pt "${PT}"
fi
