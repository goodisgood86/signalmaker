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

./scripts/pass_check_backup.py --out-dir backups/pass_check
