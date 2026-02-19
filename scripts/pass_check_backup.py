#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx


TABLES = ["pass_check_events", "pass_check_progress", "pass_check_summary"]


def _env(name: str) -> str:
    v = str(os.getenv(name, "")).strip()
    if not v:
        raise RuntimeError(f"missing env: {name}")
    return v


def _fetch_all(client: httpx.Client, base_url: str, key: str, table: str, page_size: int = 1000) -> List[Dict[str, Any]]:
    offset = 0
    out: List[Dict[str, Any]] = []
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    while True:
        params = {
            "select": "*",
            "order": "id.asc",
            "limit": str(page_size),
            "offset": str(offset),
        }
        r = client.get(f"{base_url}/rest/v1/{table}", headers=headers, params=params, timeout=60.0)
        r.raise_for_status()
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            break
        out.extend(rows)
        if len(rows) < page_size:
            break
        offset += len(rows)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Backup pass_check tables from Supabase")
    p.add_argument("--out-dir", default="backups/pass_check", help="backup base dir")
    p.add_argument("--page-size", type=int, default=1000, help="page size")
    args = p.parse_args()

    try:
        base_url = _env("SUPABASE_URL").rstrip("/")
        key = _env("SUPABASE_SERVICE_ROLE_KEY")
    except Exception as e:
        print(f"[ERR] {e}", file=sys.stderr)
        return 2

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / ts
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "supabase_url": base_url,
        "tables": {},
    }

    with httpx.Client() as client:
        for table in TABLES:
            try:
                rows = _fetch_all(client, base_url, key, table, page_size=max(100, args.page_size))
            except Exception as e:
                print(f"[ERR] backup failed: table={table} error={e}", file=sys.stderr)
                return 1
            path = out_root / f"{table}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["tables"][table] = {
                "rows": len(rows),
                "file": str(path),
            }
            print(f"[OK] {table}: {len(rows)} rows")

    mpath = out_root / "manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = Path(args.out_dir) / "LATEST"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(out_root), encoding="utf-8")

    print(f"[DONE] backup complete: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
