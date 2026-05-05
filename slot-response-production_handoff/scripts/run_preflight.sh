#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[RUN_PREFLIGHT] project_root=${PROJECT_ROOT}"
echo "[RUN_PREFLIGHT] config=${PROJECT_ROOT}/configs/response_model.yaml"

cd "${PROJECT_ROOT}"
python checks/preflight_check.py --config configs/response_model.yaml "$@"
