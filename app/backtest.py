from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .binance import BINANCE_FUTURES_BASE_URL, BINANCE_SPOT_BASE_URL, BinanceClient
from .calibration import (
    apply_isotonic,
    apply_platt,
    fit_isotonic,
    fit_platt,
    save_isotonic_model,
    save_platt_model,
)
from .indicators import compute_indicators
from .levels import compute_avwap, find_pivots, select_avwap_anchors
from .regime import classify_regime
from .scoring import score_signal_trendy
from .volume_profile import compute_volume_profile


@dataclass(frozen=True)
class PredRow:
    open_time_ms: int
    regime: str
    p_raw: float
    label: int
    p_cal: float


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


async def _fetch_klines(symbol: str, interval: str, limit: int, market: str) -> List[Dict[str, Any]]:
    base_url = BINANCE_FUTURES_BASE_URL if str(market).lower() == "futures" else BINANCE_SPOT_BASE_URL
    client = BinanceClient(base_url=base_url)
    try:
        ks = await client.klines(symbol=symbol, interval=interval, limit=limit)
    finally:
        await client.aclose()
    out: List[Dict[str, Any]] = []
    for k in ks:
        out.append(
            {
                "open_time_ms": int(k.open_time_ms),
                "open": float(k.open),
                "high": float(k.high),
                "low": float(k.low),
                "close": float(k.close),
                "volume": float(k.volume),
            }
        )
    return out


def _label_barrier(
    df: pd.DataFrame,
    t: int,
    *,
    tp_pct: float,
    sl_pct: float,
    max_holding_bars: int,
    fee_bps: float,
    slippage_bps: float,
) -> int:
    # round-trip 비용을 단순 비율로 반영
    total_cost = ((fee_bps + slippage_bps) * 2.0) / 10000.0
    entry = float(df["close"].iloc[t])
    tp = entry * (1.0 + tp_pct + total_cost)
    sl = entry * (1.0 - sl_pct - total_cost)
    end = min(len(df) - 1, t + max_holding_bars)

    for i in range(t + 1, end + 1):
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])
        tp_hit = high >= tp
        sl_hit = low <= sl
        if tp_hit and sl_hit:
            return 0
        if tp_hit:
            return 1
        if sl_hit:
            return 0
    return 1 if float(df["close"].iloc[end]) > entry else 0


def _brier(ps: List[float], ys: List[int]) -> float:
    n = len(ps)
    if n == 0:
        return 0.0
    s = 0.0
    for p, y in zip(ps, ys):
        d = float(p) - float(y)
        s += d * d
    return s / n


def run_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(config.get("symbol", "BTCUSDT")).upper()
    interval = str(config.get("interval", "5m"))
    market = str(config.get("market", "spot")).lower()
    limit = int(config.get("limit", 1000))

    strategy = config.get("strategy", {})
    tp_pct = float(strategy.get("tp_pct", 0.01))
    sl_pct = float(strategy.get("sl_pct", 0.007))
    max_holding_bars = int(strategy.get("max_holding_bars", 24))
    calibration_method = str(config.get("calibration_method", "platt")).lower()
    cost = config.get("cost", {})
    fee_bps = float(cost.get("fee_bps", 4.0))
    slippage_bps = float(cost.get("slippage_bps", 2.0))

    data = asyncio.run(_fetch_klines(symbol=symbol, interval=interval, limit=limit, market=market))
    df = pd.DataFrame(data)
    if len(df) < 120:
        raise RuntimeError("not enough data for backtest: need at least 120 bars")

    rows: List[PredRow] = []
    start = 60
    end = len(df) - max_holding_bars - 1
    for t in range(start, end):
        view = df.iloc[: t + 1].copy()
        ind = compute_indicators(view)
        if ind is None:
            continue
        regime = classify_regime(view)
        pivots = find_pivots(view, left=3, right=3)
        anchors = select_avwap_anchors(view, pivots)
        avwap_levels: List[Dict[str, float]] = []
        for a in anchors:
            idx = int(a["index"])
            v = compute_avwap(view, anchor_index=idx)
            if v is None:
                continue
            avwap_levels.append({"anchor": a["kind"], "value": float(v)})
        vp = compute_volume_profile(view, lookback=min(len(view), 300), bins=48)
        close = float(view["close"].iloc[-1])
        prev_close = float(view["close"].iloc[-2])

        score = score_signal_trendy(
            close=close,
            prev_close=prev_close,
            ind=ind,
            regime=regime.regime,
            avwap_levels=avwap_levels,
            volume_profile=vp,
        )
        p_raw = score.buy_pct / 100.0
        y = _label_barrier(
            df,
            t,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            max_holding_bars=max_holding_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        rows.append(
            PredRow(
                open_time_ms=int(view["open_time_ms"].iloc[-1]),
                regime=regime.regime,
                p_raw=float(p_raw),
                label=int(y),
                p_cal=float(p_raw),
            )
        )

    if len(rows) < 50:
        raise RuntimeError("not enough prediction rows for calibration")

    split = int(len(rows) * 0.7)
    train = rows[:split]
    valid = rows[split:]
    if calibration_method == "isotonic":
        model = fit_isotonic([r.p_raw for r in train], [r.label for r in train])
    else:
        calibration_method = "platt"
        model = fit_platt([r.p_raw for r in train], [r.label for r in train], epochs=2500, lr=0.08)

    calibrated_rows: List[PredRow] = []
    for r in rows:
        if calibration_method == "isotonic":
            p_cal = apply_isotonic(model, r.p_raw)
        else:
            p_cal = apply_platt(model, r.p_raw)
        calibrated_rows.append(
            PredRow(
                open_time_ms=r.open_time_ms,
                regime=r.regime,
                p_raw=r.p_raw,
                label=r.label,
                p_cal=p_cal,
            )
        )

    valid_raw = [r.p_raw for r in calibrated_rows[split:]]
    valid_cal = [r.p_cal for r in calibrated_rows[split:]]
    valid_y = [r.label for r in calibrated_rows[split:]]

    brier_raw = _brier(valid_raw, valid_y)
    brier_cal = _brier(valid_cal, valid_y)
    acc_raw = sum((1 if p >= 0.5 else 0) == y for p, y in zip(valid_raw, valid_y)) / max(1, len(valid_y))
    acc_cal = sum((1 if p >= 0.5 else 0) == y for p, y in zip(valid_cal, valid_y)) / max(1, len(valid_y))

    run_id = _utc_now_text() + f"_{symbol}_{interval}"
    out_dir = Path("artifacts/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{run_id}_preds.csv"
    summary_path = out_dir / f"{run_id}_summary.json"
    if calibration_method == "isotonic":
        model_path = Path("artifacts/models/isotonic_calibrator.json")
    else:
        model_path = Path("artifacts/models/platt_calibrator.json")

    with preds_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["open_time_ms", "regime", "p_raw", "label", "p_cal"])
        w.writeheader()
        for r in calibrated_rows:
            w.writerow(asdict(r))

    if calibration_method == "isotonic":
        save_isotonic_model(model, model_path)
    else:
        save_platt_model(model, model_path)

    summary = {
        "run_id": run_id,
        "symbol": symbol,
        "interval": interval,
        "market": market,
        "calibration_method": calibration_method,
        "cost": {"fee_bps": fee_bps, "slippage_bps": slippage_bps},
        "rows": len(calibrated_rows),
        "split_train": len(train),
        "split_valid": len(valid),
        "brier_raw_valid": round(brier_raw, 6),
        "brier_cal_valid": round(brier_cal, 6),
        "acc_raw_valid": round(float(acc_raw), 6),
        "acc_cal_valid": round(float(acc_cal), 6),
        "model_path": str(model_path),
        "preds_path": str(preds_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Spot TA backtest + platt calibration")
    parser.add_argument("--config", required=True, help="path to json config")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = run_backtest(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
