#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
echo "[batch] start"
if [ -f "$ROOT_DIR/configs/ops.env" ]; then
  . "$ROOT_DIR/configs/ops.env"
fi

if [ -z "${SUPABASE_URL:-}" ]; then
  echo "[ERROR] SUPABASE_URL 설정이 필요합니다. configs/ops.env를 확인하세요."
  exit 1
fi
case "${SUPABASE_URL}" in
  *YOUR_PROJECT*)
    echo "[ERROR] SUPABASE_URL 설정이 필요합니다. configs/ops.env를 확인하세요."
    exit 1
    ;;
esac
if [ -z "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
  echo "[ERROR] SUPABASE_SERVICE_ROLE_KEY 설정이 필요합니다. configs/ops.env를 확인하세요."
  exit 1
fi
if [ "${SUPABASE_SERVICE_ROLE_KEY}" = "YOUR_SUPABASE_SERVICE_ROLE_KEY" ]; then
  echo "[ERROR] SUPABASE_SERVICE_ROLE_KEY 설정이 필요합니다. configs/ops.env를 확인하세요."
  exit 1
fi
if [ -z "${APP_BASE_URL:-}" ]; then
  echo "[ERROR] APP_BASE_URL 설정이 필요합니다. (예: https://signalmaker-production.up.railway.app)"
  exit 1
fi

BASE_RAW="${APP_BASE_URL%/}"
case "$BASE_RAW" in
  http://*|https://*) ;;
  *) BASE_RAW="https://${BASE_RAW}" ;;
esac
BASE_URL="$BASE_RAW"
PY_BIN="./.venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python"
fi

"$PY_BIN" - <<'PY'
import os
import sys
import urllib.request

base = os.environ.get("APP_BASE_URL", "").rstrip("/")
url = f"{base}/api/health"
try:
    with urllib.request.urlopen(url, timeout=8) as r:
        status = int(getattr(r, "status", 0))
        body = r.read(128).decode("utf-8", "ignore")
except Exception as e:
    print(f"[ERROR] APP_BASE_URL({base}) 헬스체크 실패: {e}")
    sys.exit(1)

if status != 200:
    print(f"[ERROR] APP_BASE_URL({base}) 헬스체크 비정상 status={status} body={body}")
    sys.exit(1)
print(f"[OK] health check: {url}")
PY

"$PY_BIN" ./scripts/pass_check_batch_runner.py --base-url "$BASE_URL" --interval 5m --periods 24h,3d,7d --seed-days 7
echo "[batch] done"
