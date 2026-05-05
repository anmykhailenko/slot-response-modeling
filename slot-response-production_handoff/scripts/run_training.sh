#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[RUN_TRAINING] project_root=${PROJECT_ROOT}"
echo "[RUN_TRAINING] config=${PROJECT_ROOT}/configs/response_model.yaml"

cd "${PROJECT_ROOT}"
python src/modeling/run_phase1_response_modeling.py --config configs/response_model.yaml
