#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[VALIDATE_ENV] project_root=${PROJECT_ROOT}"

required_env_vars=(
  ALIBABA_CLOUD_ACCESS_KEY_ID
  ALIBABA_CLOUD_ACCESS_KEY_SECRET
  ODPS_PROJECT
  ODPS_ENDPOINT
)

for name in "${required_env_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "[VALIDATE_ENV] missing_env_var=${name}" >&2
    exit 1
  fi
  echo "[VALIDATE_ENV] env_var_present=${name}"
done

cd "${PROJECT_ROOT}"
python - <<'PY'
import importlib

required_modules = [
    "joblib",
    "numpy",
    "pandas",
    "yaml",
    "sklearn",
    "lightgbm",
    "odps",
]

for module_name in required_modules:
    importlib.import_module(module_name)
    print(f"[VALIDATE_ENV] python_module_present={module_name}")
PY
