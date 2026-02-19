#!/bin/zsh
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

BASE_URL="${APP_BASE_URL:-http://127.0.0.1:8010}"
PORT="${BASE_URL##*:}"
LOG=/tmp/coin_passcheck_batch_uvicorn.log

./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
UV_PID=$!
cleanup() {
  kill "$UV_PID" >/dev/null 2>&1 || true
  wait "$UV_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for i in {1..30}; do
  if curl -sS -m 2 "$BASE_URL/api/pass_check_db?symbol=BTCUSDT&market=spot&interval=5m&period=3d" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

./scripts/pass_check_batch_runner.py --base-url "$BASE_URL" --interval 5m --periods 24h,3d,7d
