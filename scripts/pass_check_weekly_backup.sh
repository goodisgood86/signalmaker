#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
echo "[backup] start"
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

PY_BIN="./.venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python"
fi

"$PY_BIN" ./scripts/pass_check_backup.py --out-dir backups/pass_check
echo "[backup] done"
