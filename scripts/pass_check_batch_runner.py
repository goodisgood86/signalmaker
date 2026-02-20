#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import httpx


def _parse_periods(raw: str) -> List[str]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    allow = {"24h", "3d", "7d"}
    out = [x for x in vals if x in allow]
    return out or ["3d"]


def _parse_symbols(raw: str) -> List[str]:
    vals = [x.strip().upper() for x in str(raw or "").split(",") if x.strip()]
    return vals


def _run_one(
    client: httpx.Client,
    base: str,
    market: str,
    interval: str,
    periods: List[str],
    symbols: List[str],
    seed_days: int,
    bootstrap_year: bool,
    bootstrap_signal_step: int,
) -> dict:
    payload = {
        "market": market,
        "interval": interval,
        "periods": periods,
        "seed_days": int(seed_days),
        "bootstrap_year": bool(bootstrap_year),
        "bootstrap_signal_step": int(bootstrap_signal_step),
    }
    if symbols:
        payload["symbols"] = symbols
    r = client.post(f"{base}/api/pass_check_batch", json=payload, timeout=600.0)
    r.raise_for_status()
    data = r.json()
    updated = data.get("updated") if isinstance(data, dict) else []
    return {
        "market": market,
        "interval": interval,
        "periods": periods,
        "symbols": symbols,
        "updated_count": len(updated) if isinstance(updated, list) else 0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Run incremental pass_check batch jobs sequentially")
    p.add_argument("--base-url", default=os.getenv("APP_BASE_URL", "http://127.0.0.1:8000"), help="API base URL")
    p.add_argument("--interval", default="5m", help="kline interval (default: 5m)")
    p.add_argument("--periods", default="24h,3d,7d", help="comma-separated: 24h,3d,7d")
    p.add_argument("--spot-symbols", default="BTCUSDT,ETHUSDT,XRPUSDT,DOGEUSDT,SUIUSDT,SOLUSDT", help="comma-separated")
    p.add_argument("--futures-symbols", default="CROSSUSDT", help="comma-separated")
    p.add_argument("--sleep-ms", type=int, default=250, help="sleep between requests")
    p.add_argument("--seed-days", type=int, default=90, help="initial seed lookback days when progress is empty")
    p.add_argument("--bootstrap-year", action="store_true", help="enable bootstrap mode for first run")
    p.add_argument("--bootstrap-signal-step", type=int, default=12, help="signal step in bootstrap mode")
    args = p.parse_args()

    base = str(args.base_url).rstrip("/")
    periods = _parse_periods(args.periods)
    spot_symbols = _parse_symbols(args.spot_symbols)
    futures_symbols = _parse_symbols(args.futures_symbols)

    if not periods:
        print("[ERR] periods is empty", file=sys.stderr)
        return 2

    jobs: List[tuple[str, str]] = []
    if spot_symbols:
        jobs.extend([("spot", s) for s in spot_symbols])
    if futures_symbols:
        jobs.extend([("futures", s) for s in futures_symbols])

    if not jobs:
        print("[ERR] no symbols to process", file=sys.stderr)
        return 2

    print(f"[INFO] start batch: base={base}, interval={args.interval}, periods={periods}, jobs={len(jobs)}")
    with httpx.Client() as client:
        for idx, (market, symbol) in enumerate(jobs, start=1):
            print(f"[INFO] ({idx}/{len(jobs)}) market={market}, symbol={symbol}")
            try:
                result = _run_one(
                    client,
                    base,
                    market,
                    args.interval,
                    periods,
                    [symbol],
                    args.seed_days,
                    args.bootstrap_year,
                    args.bootstrap_signal_step,
                )
            except Exception as e:
                print(f"[ERR] market={market} symbol={symbol} failed: {e}", file=sys.stderr)
                return 1
            print("[OK] " + json.dumps(result, ensure_ascii=False))
            if idx < len(jobs) and args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    print("[DONE] batch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
