#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
if [[ -f "$ROOT_DIR/configs/ops.env" ]]; then
  source "$ROOT_DIR/configs/ops.env"
fi

if [[ -z "${SUPABASE_URL:-}" || "${SUPABASE_URL}" == *"YOUR_PROJECT"* ]]; then
  echo "[ERROR] SUPABASE_URL 설정이 필요합니다. configs/ops.env를 확인하세요."
  exit 1
fi
if [[ -z "${SUPABASE_SERVICE_ROLE_KEY:-}" || "${SUPABASE_SERVICE_ROLE_KEY}" == "YOUR_SUPABASE_SERVICE_ROLE_KEY" ]]; then
  echo "[ERROR] SUPABASE_SERVICE_ROLE_KEY 설정이 필요합니다. configs/ops.env를 확인하세요."
  exit 1
fi

BASE_URL="${APP_BASE_URL:-http://127.0.0.1:8000}"
PY_BIN="./.venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python"
fi

if ! curl -sS -m 5 "$BASE_URL/api/health" >/dev/null 2>&1; then
  echo "[ERROR] APP_BASE_URL($BASE_URL) 헬스체크 실패"
  exit 1
fi

"$PY_BIN" ./scripts/pass_check_batch_runner.py --base-url "$BASE_URL" --interval 5m --periods 24h,3d,7d
