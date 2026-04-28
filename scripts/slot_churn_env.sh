if [ -z "${ALIBABA_CLOUD_ACCESS_KEY_ID:-}" ]; then
  echo "ALIBABA_CLOUD_ACCESS_KEY_ID must be set before sourcing scripts/slot_churn_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

if [ -z "${ALIBABA_CLOUD_ACCESS_KEY_SECRET:-}" ]; then
  echo "ALIBABA_CLOUD_ACCESS_KEY_SECRET must be set before sourcing scripts/slot_churn_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export ODPS_ENDPOINT="https://service.ap-southeast-1.maxcompute.aliyun.com/api"
export ODPS_PROJECT="pai_rec_prod"

if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  python() {
    python3 "$@"
  }
fi
