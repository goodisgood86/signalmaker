from __future__ import annotations

import asyncio
import gc
import hashlib
import math
import os
from bisect import bisect_right
from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from urllib.parse import parse_qs
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .binance import (
    ALLOWED_INTERVALS,
    ALLOWED_MARKETS,
    BINANCE_FUTURES_BASE_URL,
    BINANCE_SPOT_BASE_URL,
    CachedBinanceClient,
    Kline,
)
from .calibration import apply_isotonic, apply_platt, load_isotonic_model, load_platt_model
from .explain_rules import build_mtf_explain, build_single_explain
from .indicators import compute_indicators
from .levels import compute_avwap, find_pivots, select_avwap_anchors
from .news import fetch_news_sentiment
from .regime import classify_regime
from .scoring import score_signal, score_signal_trendy
from .volume_profile import compute_volume_profile


def _klines_to_df(klines: List[Any]) -> pd.DataFrame:
    rows = []
    for k in klines:
        rows.append(
            {
                "open_time_ms": k.open_time_ms,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
                "close_time_ms": k.close_time_ms,
            }
        )
    df = pd.DataFrame(rows)
    return df


async def _fetch_klines_paged_for_pass_check(
    *, symbol: str, interval: str, market: str, total_limit: int
) -> List[Kline]:
    symbol_u = symbol.upper().strip()
    market_u = market.lower().strip()
    if interval not in ALLOWED_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
    if market_u not in ALLOWED_MARKETS:
        raise ValueError(f"market must be one of {sorted(ALLOWED_MARKETS)}")
    base_url = BINANCE_FUTURES_BASE_URL if market_u == "futures" else BINANCE_SPOT_BASE_URL
    path = "/fapi/v1/klines" if market_u == "futures" else "/api/v3/klines"
    max_batch = 1000
    remain = max(50, min(int(total_limit), 200000))
    end_time_ms: int | None = None
    rows_all: List[List[Any]] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=12.0) as c:
        while remain > 0:
            batch = min(max_batch, remain)
            params: Dict[str, Any] = {"symbol": symbol_u, "interval": interval, "limit": batch}
            if end_time_ms is not None:
                params["endTime"] = end_time_ms
            r = await c.get(path, params=params)
            r.raise_for_status()
            rows: List[List[Any]] = r.json()
            if not rows:
                break
            rows_all.extend(rows)
            oldest_open = int(rows[0][0])
            end_time_ms = oldest_open - 1
            remain -= len(rows)
            if len(rows) <= 0:
                break
    by_open: Dict[int, List[Any]] = {}
    for row in rows_all:
        by_open[int(row[0])] = row
    merged = [by_open[k] for k in sorted(by_open.keys())]
    if len(merged) > total_limit:
        merged = merged[-int(total_limit) :]
    out: List[Kline] = []
    for row in merged:
        out.append(
            Kline(
                open_time_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time_ms=int(row[6]),
            )
        )
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    cache_max = max(64, int(str(os.getenv("BINANCE_CACHE_MAX_ENTRIES", "192")) or "192"))
    app.state.binance = CachedBinanceClient(ttl_seconds=10.0, cache_max_entries=cache_max)
    try:
        yield
    finally:
        await app.state.binance.aclose()


app = FastAPI(title="Coin TA MVP", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

_PLATT_MODEL_PATH = Path("artifacts/models/platt_calibrator.json")
_ISOTONIC_MODEL_PATH = Path("artifacts/models/isotonic_calibrator.json")
_FX_CACHE: Dict[str, float] = {"ts": 0.0, "usdt_krw": 1350.0}
_NEWS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_PASS_TUNE_CACHE: Dict[str, Any] = {}
_PASS_TUNE_PROFILE_CACHE: Dict[str, Any] = {}
_PASS_CHECK_CACHE: Dict[str, Any] = {}
_WS_ANALYSIS_CACHE: Dict[str, Any] = {}
_WS_ANALYSIS_CACHE_TTL_S = 3.0
_WS_ANALYSIS_POLL_S = 3.0
_PASS_TUNE_CACHE_MAX = max(24, int(str(os.getenv("PASS_TUNE_CACHE_MAX", "80")) or "80"))
_PASS_TUNE_PROFILE_CACHE_MAX = max(24, int(str(os.getenv("PASS_TUNE_PROFILE_CACHE_MAX", "80")) or "80"))
_WS_ANALYSIS_CACHE_MAX = max(24, int(str(os.getenv("WS_ANALYSIS_CACHE_MAX", "120")) or "120"))


def _trim_ttl_cache(cache_obj: Dict[str, Any], *, now_ts: float, ttl_s: float) -> None:
    stale: List[str] = []
    for k, v in cache_obj.items():
        ts = float(v.get("ts", 0.0)) if isinstance(v, dict) else 0.0
        if now_ts - ts > ttl_s:
            stale.append(k)
    for k in stale:
        cache_obj.pop(k, None)


def _set_bounded_cache(cache_obj: Dict[str, Any], key: str, value: Any, *, max_entries: int) -> None:
    cache_obj[key] = value
    overflow = len(cache_obj) - max_entries
    if overflow <= 0:
        return
    for old_k in list(cache_obj.keys())[:overflow]:
        cache_obj.pop(old_k, None)


def _try_malloc_trim() -> None:
    # Linux/glibc 환경에서만 동작. 실패해도 무시.
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        trim_fn = getattr(libc, "malloc_trim", None)
        if callable(trim_fn):
            trim_fn(0)
    except Exception:
        return


def _compact_runtime_memory() -> None:
    gc.collect()
    _try_malloc_trim()


def _pc_window_start_ms(days: int = 90) -> int:
    return int(time() * 1000) - int(days) * 24 * 60 * 60 * 1000


def _sb_url() -> str:
    return str(os.getenv("SUPABASE_URL", "")).strip().rstrip("/")


def _sb_key() -> str:
    return str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or "").strip()


def _sim_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


async def _sb_request(method: str, path: str, *, params: Dict[str, Any] | None = None, json_body: Any = None) -> Any:
    url = _sb_url()
    key = _sb_key()
    if not url or not key:
        raise HTTPException(status_code=500, detail="supabase is not configured")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    if method.upper() in {"POST", "PATCH", "PUT"}:
        headers["Content-Type"] = "application/json"
        headers["Prefer"] = "return=representation"
    req_method = method.upper()
    timeout = httpx.Timeout(20.0, connect=8.0)
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.request(req_method, f"{url}{path}", headers=headers, params=params, json=json_body)
            break
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.NetworkError) as e:
            last_err = e
            if attempt >= 3:
                raise HTTPException(status_code=502, detail=f"supabase request timeout/network error: {type(e).__name__}")
            await asyncio.sleep(0.4 * (attempt + 1))
    else:
        raise HTTPException(status_code=502, detail=f"supabase request failed: {type(last_err).__name__ if last_err else 'unknown'}")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"supabase request failed: {r.status_code} {r.text[:200]}")
    if not r.text:
        return None
    try:
        return r.json()
    except Exception:
        return None


async def _sim_get_user_by_nickname(nickname: str) -> Optional[Dict[str, Any]]:
    rows = await _sb_request(
        "GET",
        "/rest/v1/sim_users",
        params={"select": "id,nickname,password_hash", "nickname": f"eq.{nickname}", "limit": "1"},
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


async def _sim_get_or_create_public_user() -> Dict[str, Any]:
    nickname = "__public__"
    user = await _sim_get_user_by_nickname(nickname)
    if user:
        return user
    rows = await _sb_request(
        "POST",
        "/rest/v1/sim_users",
        json_body=[{"nickname": nickname, "password_hash": _sim_password_hash("public"), "created_ms": int(time() * 1000)}],
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    user2 = await _sim_get_user_by_nickname(nickname)
    if user2:
        return user2
    raise HTTPException(status_code=500, detail="failed to init public sim user")


async def _pc_get_progress(symbol: str, market: str, interval: str, period: str) -> Optional[Dict[str, Any]]:
    rows = await _sb_request(
        "GET",
        "/rest/v1/pass_check_progress",
        params={
            "select": "id,symbol,market,interval,period,last_signal_time_ms,updated_ms",
            "symbol": f"eq.{symbol}",
            "market": f"eq.{market}",
            "interval": f"eq.{interval}",
            "period": f"eq.{period}",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


async def _pc_get_summary(symbol: str, market: str, interval: str, period: str) -> Optional[Dict[str, Any]]:
    rows = await _sb_request(
        "GET",
        "/rest/v1/pass_check_summary",
        params={
            "select": "id,symbol,market,interval,period,pass_count,executed_count,no_entry_count,tp1_hit_count,sl_hit_count,no_hit_count,resolved_count,latest_signal_time_ms,updated_ms",
            "symbol": f"eq.{symbol}",
            "market": f"eq.{market}",
            "interval": f"eq.{interval}",
            "period": f"eq.{period}",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


async def _pc_get_first_signal_time_ms(symbol: str, market: str, interval: str, period: str) -> int:
    start_ms = _pc_window_start_ms(90)
    rows = await _sb_request(
        "GET",
        "/rest/v1/pass_check_events",
        params={
            "select": "signal_time_ms",
            "symbol": f"eq.{symbol}",
            "market": f"eq.{market}",
            "interval": f"eq.{interval}",
            "period": f"eq.{period}",
            "signal_time_ms": f"gte.{start_ms}",
            "order": "signal_time_ms.asc",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return int(rows[0].get("signal_time_ms", 0) or 0)
    return 0


async def _pc_get_entry_baseline_counts(symbol: str, market: str, interval: str) -> Dict[str, int]:
    start_ms = _pc_window_start_ms(90)
    rows = await _sb_request(
        "GET",
        "/rest/v1/pass_check_events",
        params={
            "select": "signal_time_ms,executed",
            "symbol": f"eq.{symbol}",
            "market": f"eq.{market}",
            "interval": f"eq.{interval}",
            "period": "in.(24h,3d,7d)",
            "signal_time_ms": f"gte.{start_ms}",
            "order": "signal_time_ms.asc",
            "limit": "100000",
        },
    )
    events = rows if isinstance(rows, list) else []
    by_signal: Dict[int, bool] = {}
    for e in events:
        stm = int(e.get("signal_time_ms", 0) or 0)
        if stm <= 0:
            continue
        prev = bool(by_signal.get(stm, False))
        by_signal[stm] = prev or bool(e.get("executed"))
    if not by_signal:
        return {
            "pass_count": 0,
            "executed_count": 0,
            "no_entry_count": 0,
            "first_signal_time_ms": 0,
            "latest_signal_time_ms": 0,
        }
    keys = sorted(by_signal.keys())
    pass_cnt = len(keys)
    exec_cnt = sum(1 for k in keys if by_signal[k])
    return {
        "pass_count": int(pass_cnt),
        "executed_count": int(exec_cnt),
        "no_entry_count": int(max(0, pass_cnt - exec_cnt)),
        "first_signal_time_ms": int(keys[0]),
        "latest_signal_time_ms": int(keys[-1]),
    }


async def _pc_refresh_summary_from_events(symbol: str, market: str, interval: str, period: str) -> None:
    start_ms = _pc_window_start_ms(90)
    rows = await _sb_request(
        "GET",
        "/rest/v1/pass_check_events",
        params={
            "select": "signal_time_ms,executed,result",
            "symbol": f"eq.{symbol}",
            "market": f"eq.{market}",
            "interval": f"eq.{interval}",
            "period": f"eq.{period}",
            "signal_time_ms": f"gte.{start_ms}",
            "order": "signal_time_ms.desc",
            "limit": "100000",
        },
    )
    events = rows if isinstance(rows, list) else []
    pass_cnt = len(events)
    executed = [e for e in events if bool(e.get("executed"))]
    executed_cnt = len(executed)
    tp_hits = sum(1 for e in executed if str(e.get("result", "")).upper() == "TP1_HIT")
    sl_hits = sum(1 for e in executed if str(e.get("result", "")).upper() == "SL_HIT")
    no_hits = sum(1 for e in executed if str(e.get("result", "")).upper() == "NO_HIT")
    no_entry = sum(1 for e in events if not bool(e.get("executed")))
    resolved = tp_hits + sl_hits
    latest_signal_ms = int(events[0]["signal_time_ms"]) if events else 0
    now_ms = int(time() * 1000)
    body = {
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "period": period,
        "pass_count": int(pass_cnt),
        "executed_count": int(executed_cnt),
        "no_entry_count": int(no_entry),
        "tp1_hit_count": int(tp_hits),
        "sl_hit_count": int(sl_hits),
        "no_hit_count": int(no_hits),
        "resolved_count": int(resolved),
        "latest_signal_time_ms": latest_signal_ms,
        "updated_ms": now_ms,
    }
    existing = await _pc_get_summary(symbol, market, interval, period)
    if existing:
        await _sb_request("PATCH", "/rest/v1/pass_check_summary", params={"id": f"eq.{existing.get('id')}"}, json_body=body)
    else:
        await _sb_request("POST", "/rest/v1/pass_check_summary", json_body=[body])


def _build_pass_check_payload(
    *,
    symbol: str,
    market: str,
    interval: str,
    period: str,
    pass_count: int,
    executed_count: int,
    no_entry_count: int,
    tp1_hit_count: int,
    sl_hit_count: int,
    no_hit_count: int,
    resolved_count: int,
    first_signal_time_ms: int,
    latest_signal_time_ms: int,
    updated_ms: int,
    source: str,
) -> Dict[str, Any]:
    p = int(pass_count)
    e = int(executed_count)
    r = int(resolved_count)
    tp = int(tp1_hit_count)
    return {
        "ok": True,
        "source": source,
        "symbol": symbol.upper(),
        "market": market.lower(),
        "interval": interval,
        "period": period,
        "pass_count": p,
        "executed_count": int(executed_count),
        "no_entry_count": int(no_entry_count),
        "tp1_hit_count": tp,
        "sl_hit_count": int(sl_hit_count),
        "no_hit_count": int(no_hit_count),
        "resolved_count": r,
        "tp1_hit_rate": round((tp / p), 4) if p > 0 else 0.0,
        "executed_tp1_hit_rate": round((tp / e), 4) if e > 0 else 0.0,
        "resolved_tp1_hit_rate": round((tp / r), 4) if r > 0 else 0.0,
        "first_signal_time_ms": int(first_signal_time_ms),
        "latest_signal_time_ms": int(latest_signal_time_ms),
        "updated_ms": int(updated_ms),
    }


def _is_isotonic_degenerate(model: Any) -> bool:
    xs = [float(v) for v in (getattr(model, "xs", []) or [])]
    ys = [float(v) for v in (getattr(model, "ys", []) or [])]
    if len(xs) < 4 or len(xs) != len(ys):
        return True
    x_min = min(xs)
    x_max = max(xs)
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_max <= x_min:
        return True

    # 긴 평탄 구간(plateau)이 있으면 다양한 raw 점수가 같은 확률로 눌릴 수 있다.
    max_flat_span = 0.0
    for i in range(1, len(xs)):
        if abs(ys[i] - ys[i - 1]) <= 1e-9:
            span = max(0.0, xs[i] - xs[i - 1])
            if span > max_flat_span:
                max_flat_span = span

    # 실사용 구간에서 출력 다양성이 부족하면 보정을 비활성화한다.
    probe = (0.25, 0.35, 0.45, 0.55, 0.65)
    probe_vals = {round(float(apply_isotonic(model, p)), 4) for p in probe}
    if max_flat_span >= 0.22:
        return True
    if len(probe_vals) <= 2:
        return True
    return False


def _apply_calibration(raw_buy: float) -> tuple[float, bool, str | None]:
    isotonic = load_isotonic_model(_ISOTONIC_MODEL_PATH)
    if isotonic is not None and (not _is_isotonic_degenerate(isotonic)):
        return apply_isotonic(isotonic, raw_buy), True, "isotonic"
    platt = load_platt_model(_PLATT_MODEL_PATH)
    if platt is not None:
        return apply_platt(platt, raw_buy), True, "platt"
    return raw_buy, False, None


def _build_levels(df: pd.DataFrame) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pivots = find_pivots(df, left=3, right=3)
    anchors = select_avwap_anchors(df, pivots)
    avwap_levels = []
    for a in anchors:
        idx = int(a["index"])
        avwap_last = compute_avwap(df, anchor_index=idx)
        if avwap_last is None:
            continue
        avwap_levels.append(
            {
                "anchor": a["kind"],
                "anchor_index": idx,
                "anchor_time_ms": int(df["open_time_ms"].iloc[idx]),
                "value": round(float(avwap_last), 6),
            }
        )
    vp = compute_volume_profile(df, lookback=min(len(df), 300), bins=48)
    return avwap_levels, vp


def _symbol_profile(symbol: str) -> str:
    s = str(symbol or "").upper()
    if s in {"BTCUSDT", "ETHUSDT"}:
        return "major"
    if s in {"DOGEUSDT", "SUIUSDT", "SOLUSDT", "XRPUSDT", "CROSSUSDT"}:
        return "beta"
    return "default"


def _decision_params_by_regime(regime: str, symbol: str = "") -> Dict[str, float]:
    r = str(regime or "RANGE").upper()
    profile = _symbol_profile(symbol)
    if r == "TREND":
        out = {
            "side_strong": 8.0,
            "side_weak": 4.0,
            "conf_weak": 0.20,
            "conf_floor": 0.05,
            "prob_min_pct": 52.0,
            "prob_min_conf": 0.30,
            "prob_soft_conf": 0.36,
            "pass_regime_conf": 0.33,
            "pass_regime_diff": 8.0,
            "fib_tol_pct": 0.0035,
        }
    elif r == "HIGH_VOL":
        out = {
            "side_strong": 12.0,
            "side_weak": 6.0,
            "conf_weak": 0.30,
            "conf_floor": 0.10,
            "prob_min_pct": 57.0,
            "prob_min_conf": 0.42,
            "prob_soft_conf": 0.50,
            "pass_regime_conf": 0.45,
            "pass_regime_diff": 12.0,
            "fib_tol_pct": 0.006,
        }
    else:
        out = {
            "side_strong": 10.0,
            "side_weak": 5.0,
            "conf_weak": 0.26,
            "conf_floor": 0.07,
            "prob_min_pct": 55.0,
            "prob_min_conf": 0.36,
            "prob_soft_conf": 0.42,
            "pass_regime_conf": 0.38,
            "pass_regime_diff": 10.0,
            "fib_tol_pct": 0.004,
        }

    # 메이저 코인은 과도한 필터를 완화, 변동성 높은 알트는 약간 더 엄격하게 적용
    if profile == "major":
        out["prob_min_pct"] = max(50.0, out["prob_min_pct"] - 1.0)
        out["prob_min_conf"] = max(0.28, out["prob_min_conf"] - 0.03)
        out["prob_soft_conf"] = max(0.33, out["prob_soft_conf"] - 0.03)
        out["pass_regime_conf"] = max(0.30, out["pass_regime_conf"] - 0.03)
    elif profile == "beta":
        out["prob_min_conf"] = min(0.48, out["prob_min_conf"] + 0.02)
        out["prob_soft_conf"] = min(0.58, out["prob_soft_conf"] + 0.02)
        out["pass_regime_conf"] = min(0.52, out["pass_regime_conf"] + 0.02)
        out["side_strong"] = out["side_strong"] + 1.0
    return out


def _latest_swing(df: pd.DataFrame, lookback: int = 200) -> Dict[str, float] | None:
    if len(df) < 60:
        return None
    recent = df.iloc[-min(lookback, len(df)) :].reset_index(drop=True)
    piv = find_pivots(recent, left=3, right=3)
    points: List[tuple[int, str, float]] = []
    for i in piv.get("pivot_highs", []):
        points.append((int(i), "H", float(recent["high"].iloc[int(i)])))
    for i in piv.get("pivot_lows", []):
        points.append((int(i), "L", float(recent["low"].iloc[int(i)])))
    points.sort(key=lambda x: x[0])
    for j in range(len(points) - 1, 0, -1):
        a = points[j - 1]
        b = points[j]
        if a[1] == b[1]:
            continue
        is_up = a[1] == "L" and b[1] == "H"
        lo = min(float(a[2]), float(b[2]))
        hi = max(float(a[2]), float(b[2]))
        if hi <= lo:
            continue
        return {"is_up": 1.0 if is_up else 0.0, "lo": lo, "hi": hi}
    return None


def _fib_price(is_up: bool, lo: float, hi: float, ratio: float) -> float:
    rng = hi - lo
    end = hi if is_up else lo
    return (end - rng * ratio) if is_up else (end + rng * ratio)


def _norm_tp(entry_hi: float, a: float, b: float, min_gap: float) -> tuple[float, float]:
    lo = min(a, b)
    hi = max(a, b)
    min_lo = entry_hi * (1.0 + min_gap)
    if lo < min_lo:
        lo = min_lo
    if hi <= lo:
        hi = lo * (1.0 + min_gap)
    return lo, hi


def _period_to_horizon_bars(interval: str, period: str) -> int:
    p = str(period or "3d").lower().strip()
    if interval == "5m":
        return {"24h": 288, "3d": 864, "7d": 2016}.get(p, 864)
    if interval == "1h":
        return {"24h": 24, "3d": 72, "7d": 168}.get(p, 72)
    if interval == "4h":
        return {"24h": 6, "3d": 18, "7d": 42}.get(p, 18)
    return 72


def _horizon_to_period(interval: str, horizon_bars: int) -> str:
    h = max(1, int(horizon_bars))
    candidates = {
        "5m": {"24h": 288, "3d": 864, "7d": 2016},
        "1h": {"24h": 24, "3d": 72, "7d": 168},
        "4h": {"24h": 6, "3d": 18, "7d": 42},
    }.get(interval, {"24h": 24, "3d": 72, "7d": 168})
    best_period = "3d"
    best_diff = 10**9
    for p, bars in candidates.items():
        d = abs(int(bars) - h)
        if d < best_diff:
            best_period = p
            best_diff = d
    return best_period


def _bars_for_days(interval: str, days: int) -> int:
    d = max(1, int(days))
    if interval == "5m":
        return d * 24 * 12
    if interval == "1h":
        return d * 24
    if interval == "4h":
        return d * 6
    return d * 24


def _build_mtf_side_lookup_from_5m(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 120:
        return {}
    work = df.copy()
    work["dt"] = pd.to_datetime(work["open_time_ms"], unit="ms", utc=True)
    work = work.set_index("dt")

    def _resample_side(rule: str) -> tuple[list[int], list[str]]:
        rs = work.resample(rule, label="right", closed="right").agg(
            {
                "open_time_ms": "last",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "close_time_ms": "last",
            }
        )
        rs = rs.dropna(subset=["close"]).copy()
        if len(rs) < 60:
            return [], []
        ema20 = rs["close"].ewm(span=20, adjust=False).mean()
        ema50 = rs["close"].ewm(span=50, adjust=False).mean()
        times: list[int] = []
        sides: list[str] = []
        for i in range(0, len(rs)):
            t = int(rs["open_time_ms"].iloc[i])
            s = "BUY" if float(ema20.iloc[i]) >= float(ema50.iloc[i]) else "SELL"
            times.append(t)
            sides.append(s)
        return times, sides

    t1, s1 = _resample_side("1h")
    t4, s4 = _resample_side("4h")
    return {"1h_t": t1, "1h_s": s1, "4h_t": t4, "4h_s": s4}


def _lookup_side_at_or_before(times: List[int], sides: List[str], t_ms: int) -> Optional[str]:
    if not times or not sides:
        return None
    idx = bisect_right(times, int(t_ms)) - 1
    if idx < 0 or idx >= len(sides):
        return None
    return sides[idx]


def _calc_pass_check_from_df(
    *,
    df: pd.DataFrame,
    symbol: str,
    market: str,
    interval: str,
    horizon_bars: int,
    entry_window_bars: int,
    min_signal_time_ms: int = 0,
    signal_step: int = 1,
    use_mtf_gate: bool = True,
    use_volume_filter: bool = True,
    use_reaction_entry: bool = True,
) -> Dict[str, Any]:
    start = 220
    end = len(df) - int(horizon_bars) - 1
    if end <= start:
        return {
            "pass_count": 0,
            "executed_count": 0,
            "no_entry_count": 0,
            "tp1_hit_count": 0,
            "sl_hit_count": 0,
            "no_hit_count": 0,
            "fail_count": 0,
            "resolved_count": 0,
            "tp1_hit_rate": 0.0,
            "executed_tp1_hit_rate": 0.0,
            "resolved_tp1_hit_rate": 0.0,
            "samples": [],
            "events": [],
        }

    pass_cnt = 0
    executed_cnt = 0
    no_entry_cnt = 0
    win_cnt = 0
    sl_cnt = 0
    no_hit_cnt = 0
    fail_cnt = 0
    samples: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    mtf_map = _build_mtf_side_lookup_from_5m(df) if (use_mtf_gate and interval == "5m") else {}

    step = max(1, int(signal_step))
    scan_span = max(0, end - start)
    # 긴 백테스트 구간에서 5m 1봉 단위 풀스캔은 과도하게 느리므로 적응형 step 적용
    if step == 1 and interval == "5m":
        if scan_span >= 2200:
            step = 3
        elif scan_span >= 1400:
            step = 2
    for t in range(start, end, step):
        signal_time_ms = int(df["open_time_ms"].iloc[t])
        if min_signal_time_ms > 0 and signal_time_ms <= min_signal_time_ms:
            continue
        # 부트스트랩/장구간 계산 성능을 위해 지표 계산 창을 제한
        # (지표/스윙에 필요한 범위만 유지)
        window_start = max(0, t - 1800)
        view = df.iloc[window_start : t + 1]
        ind = compute_indicators(view)
        if ind is None:
            continue
        regime = classify_regime(view)
        avwap_levels, vp = _build_levels(view)
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

        buy = float(score.buy_pct)
        sell = float(score.sell_pct)
        conf = float(score.confidence)
        raw_diff = buy - sell
        swing = _latest_swing(view, lookback=200)
        if swing is None:
            continue
        is_up = bool(int(swing["is_up"]))
        lo = float(swing["lo"])
        hi = float(swing["hi"])
        if hi <= lo:
            continue
        swing_bias = 6.0 if is_up else -6.0
        diff = raw_diff + swing_bias
        params = _decision_params_by_regime(regime.regime, symbol)

        side = "WAIT"
        if diff >= params["side_strong"] or (diff >= params["side_weak"] and conf >= params["conf_weak"]):
            side = "BUY"
        elif diff <= -params["side_strong"] or (diff <= -params["side_weak"] and conf >= params["conf_weak"]):
            side = "SELL"
        if conf < params["conf_floor"]:
            side = "WAIT"
        if regime.regime == "HIGH_VOL" and conf < params["pass_regime_conf"] and abs(diff) < params["pass_regime_diff"]:
            side = "WAIT"
        if (is_up and side == "SELL") or ((not is_up) and side == "BUY"):
            side = "WAIT"
        if market == "spot" and side == "SELL":
            side = "WAIT"
        if side != "WAIT" and mtf_map:
            s1 = _lookup_side_at_or_before(mtf_map.get("1h_t", []), mtf_map.get("1h_s", []), signal_time_ms)
            s4 = _lookup_side_at_or_before(mtf_map.get("4h_t", []), mtf_map.get("4h_s", []), signal_time_ms)
            if s1 is None or s4 is None or s1 != s4 or s1 != side:
                side = "WAIT"
        if side != "WAIT" and use_volume_filter and t >= 20:
            cur_vol = float(df["volume"].iloc[t])
            avg_vol = float(df["volume"].iloc[t - 20 : t].mean())
            if (not math.isfinite(cur_vol)) or (not math.isfinite(avg_vol)) or avg_vol <= 0 or cur_vol < avg_vol:
                side = "WAIT"
        if side == "WAIT":
            continue

        p0 = _fib_price(is_up, lo, hi, 0.0)
        p0236 = _fib_price(is_up, lo, hi, 0.236)
        p0382 = _fib_price(is_up, lo, hi, 0.382)
        p05 = _fib_price(is_up, lo, hi, 0.5)
        p0618 = _fib_price(is_up, lo, hi, 0.618)
        p0786 = _fib_price(is_up, lo, hi, 0.786)

        atr_pct_raw = (float(ind.atr14) / close) if close > 0 else 0.0
        atr_pct = min(0.05, max(0.004, atr_pct_raw))
        # 1차 익절 최소 폭은 1.0%로 유지 (과도하게 짧은 익절 방지)
        tp1_gap = min(0.022, max(0.01, atr_pct * 0.85))
        stop_gap = min(0.028, max(0.007, atr_pct * 0.9))
        fib_tol = close * params["fib_tol_pct"]

        if side == "BUY" and is_up:
            entry_lo, entry_hi = min(p05, p0618), max(p05, p0618)
            stop = min(p0786, entry_lo * (1.0 - stop_gap))
            tp1_lo, tp1_hi = _norm_tp(entry_hi, p0236, p0382, tp1_gap)
        elif side == "BUY" and (not is_up):
            entry_lo, entry_hi = min(p0, p0236), max(p0, p0236)
            stop = min(p0 * 0.997, entry_lo * (1.0 - stop_gap))
            tp1_lo, tp1_hi = _norm_tp(entry_hi, p0236, p0382, tp1_gap)
        elif side == "SELL" and is_up:
            entry_lo, entry_hi = min(p0382, p05), max(p0382, p05)
            stop = min(p0618 * 0.997, entry_lo * (1.0 - stop_gap))
            tp1_lo, tp1_hi = _norm_tp(entry_hi, p0236, p0382, tp1_gap * 1.1)
        else:
            entry_lo, entry_hi = min(p0, p0236), max(p0, p0236)
            stop = min(p0 * 0.996, entry_lo * (1.0 - stop_gap))
            tp1_lo, tp1_hi = _norm_tp(entry_hi, p0382, p05, tp1_gap * 1.1)

        pass_prob = (
            side == "BUY"
            and buy >= params["prob_min_pct"]
            and conf >= params["prob_min_conf"]
            and (conf >= params["prob_soft_conf"] or diff >= params["side_strong"])
        ) or (
            side == "SELL"
            and sell >= params["prob_min_pct"]
            and conf >= params["prob_min_conf"]
            and (conf >= params["prob_soft_conf"] or diff <= -params["side_strong"])
        )
        pass_regime = (regime.regime != "HIGH_VOL") or conf >= params["pass_regime_conf"] or abs(diff) >= params["pass_regime_diff"]
        pass_fib = (close >= entry_lo - fib_tol) and (
            close <= (entry_hi + fib_tol * 1.6 if side == "SELL" else entry_hi + fib_tol)
        )
        entry_mid = (entry_lo + entry_hi) / 2.0
        tp1_mid = (tp1_lo + tp1_hi) / 2.0
        rr = (tp1_mid - entry_mid) / max(1e-12, (entry_mid - stop)) if entry_mid > stop else 0.0
        min_tp_pass = max(0.01, tp1_gap * 0.55)
        min_stop_pass = max(0.006, stop_gap * 0.7)
        pass_plan = (
            tp1_mid > entry_mid * (1.0 + min_tp_pass)
            and stop < entry_mid * (1.0 - min_stop_pass)
            and rr >= 0.9
        )
        if not (pass_prob and pass_regime and pass_fib and pass_plan):
            continue

        pass_cnt += 1
        entry_idx = -1
        entry_px = entry_mid
        entry_search_end = min(len(df), t + 1 + int(entry_window_bars))
        for i in range(t + 1, entry_search_end):
            hi_bar = float(df["high"].iloc[i])
            lo_bar = float(df["low"].iloc[i])
            if hi_bar >= entry_lo and lo_bar <= entry_hi:
                if use_reaction_entry:
                    confirm_idx = i + 1
                    if confirm_idx >= len(df):
                        break
                    confirm_close = float(df["close"].iloc[confirm_idx])
                    if side == "BUY" and confirm_close >= entry_hi:
                        entry_idx = confirm_idx
                        entry_px = min(entry_hi, max(entry_lo, confirm_close))
                        break
                    if side == "SELL" and confirm_close <= entry_lo:
                        entry_idx = confirm_idx
                        entry_px = min(entry_hi, max(entry_lo, confirm_close))
                        break
                    continue
                entry_idx = i
                close_bar = float(df["close"].iloc[i])
                entry_px = min(entry_hi, max(entry_lo, close_bar))
                break
        if entry_idx < 0:
            no_entry_cnt += 1
            events.append(
                {
                    "signal_time_ms": signal_time_ms,
                    "entry_time_ms": None,
                    "executed": False,
                    "result": "NO_ENTRY",
                    "side": side,
                    "entry": round(entry_mid, 8),
                    "tp1": round(tp1_mid, 8),
                    "stop": round(stop, 8),
                }
            )
            continue

        executed_cnt += 1
        tp_hit = False
        sl_hit = False
        for i in range(entry_idx + 1, min(len(df), entry_idx + 1 + int(horizon_bars))):
            hi_bar = float(df["high"].iloc[i])
            lo_bar = float(df["low"].iloc[i])
            if hi_bar >= tp1_mid and lo_bar <= stop:
                sl_hit = True
                break
            if lo_bar <= stop:
                sl_hit = True
                break
            if hi_bar >= tp1_mid:
                tp_hit = True
                break
        if tp_hit:
            win_cnt += 1
            res = "TP1_HIT"
        elif sl_hit:
            sl_cnt += 1
            fail_cnt += 1
            res = "SL_HIT"
        else:
            no_hit_cnt += 1
            fail_cnt += 1
            res = "NO_HIT"
        if len(samples) < 10:
            samples.append(
                {
                    "time_ms": signal_time_ms,
                    "entry_time_ms": int(df["open_time_ms"].iloc[entry_idx]),
                    "side": side,
                    "entry": round(entry_px, 8),
                    "tp1": round(tp1_mid, 8),
                    "stop": round(stop, 8),
                    "result": res,
                }
            )
        events.append(
            {
                "signal_time_ms": signal_time_ms,
                "entry_time_ms": int(df["open_time_ms"].iloc[entry_idx]),
                "executed": True,
                "result": res,
                "side": side,
                "entry": round(entry_px, 8),
                "tp1": round(tp1_mid, 8),
                "stop": round(stop, 8),
            }
        )

    win_rate = (win_cnt / pass_cnt) if pass_cnt > 0 else 0.0
    exec_win_rate = (win_cnt / executed_cnt) if executed_cnt > 0 else 0.0
    resolved_cnt = win_cnt + sl_cnt
    resolved_win_rate = (win_cnt / resolved_cnt) if resolved_cnt > 0 else 0.0
    return {
        "pass_count": int(pass_cnt),
        "executed_count": int(executed_cnt),
        "no_entry_count": int(no_entry_cnt),
        "tp1_hit_count": int(win_cnt),
        "sl_hit_count": int(sl_cnt),
        "no_hit_count": int(no_hit_cnt),
        "fail_count": int(fail_cnt),
        "resolved_count": int(resolved_cnt),
        "tp1_hit_rate": round(win_rate, 4),
        "executed_tp1_hit_rate": round(exec_win_rate, 4),
        "resolved_tp1_hit_rate": round(resolved_win_rate, 4),
        "samples": samples,
        "events": events,
    }


async def _get_usdt_krw() -> float:
    now = time()
    if now - float(_FX_CACHE.get("ts", 0.0)) < 600 and float(_FX_CACHE.get("usdt_krw", 0.0)) > 0:
        return float(_FX_CACHE["usdt_krw"])

    rate = float(_FX_CACHE.get("usdt_krw", 1350.0))
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get("https://open.er-api.com/v6/latest/USD")
            r.raise_for_status()
            j = r.json()
            fx = j.get("rates", {})
            krw = fx.get("KRW")
            if krw is not None:
                rate = float(krw)
    except Exception:
        pass

    _FX_CACHE["usdt_krw"] = rate
    _FX_CACHE["ts"] = now
    return rate


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


def _sim_eval_trade_status(trade: Dict[str, Any], klines: List[Any]) -> tuple[str, Optional[float]]:
    now_ms = int(time() * 1000)
    expire_ms = 6 * 24 * 60 * 60 * 1000
    tp = float(trade.get("take_profit", 0.0) or 0.0)
    sl = float(trade.get("stop_loss", 0.0) or 0.0)
    entry = float(trade.get("entry_price", 0.0) or 0.0)
    created_ms = int(trade.get("created_ms", 0) or 0)
    if tp <= 0 or sl <= 0 or entry <= 0 or created_ms <= 0:
        return "IN_PROGRESS", None
    is_expired = now_ms >= (created_ms + expire_ms)
    entry_touched = False
    last_close = None
    for k in klines:
        k_open = int(getattr(k, "open_time_ms", 0))
        k_close = int(getattr(k, "close_time_ms", 0))
        if k_open < created_ms:
            # 작성 시점이 봉 중간일 수 있으므로, close 시각까지도 작성 이전이면 제외
            if k_close > 0 and k_close < created_ms:
                continue
        is_partial_candle = k_open < created_ms <= (k_close or k_open)
        hi = float(getattr(k, "high", 0.0))
        lo = float(getattr(k, "low", 0.0))
        op = float(getattr(k, "open", 0.0))
        last_close = float(getattr(k, "close", 0.0))
        if not entry_touched:
            # 봉 범위 체결 + 현재가/시가 근접치 체결(작성 직후 체결 누락 방지)
            eps = max(entry * 0.0006, 1e-9)  # 약 0.06%
            # 부분 봉에서는 과거 틱(작성 이전) 포함 가능성이 있어 범위 체결을 사용하지 않음
            touched_by_range = (not is_partial_candle) and (lo <= entry <= hi)
            touched_by_close = abs(last_close - entry) <= eps
            touched_by_open = op > 0 and abs(op - entry) <= eps
            entry_touched = touched_by_range or touched_by_close or touched_by_open
            if not entry_touched:
                continue
        # 작성 시점이 포함된 부분 봉은 고가/저가로 TP/SL 확정하지 않고 다음 봉부터 판정
        if is_partial_candle:
            continue
        both = hi >= tp and lo <= sl
        if both:
            return "SL", last_close
        if lo <= sl:
            return "SL", last_close
        if hi >= tp:
            return "TP", last_close
    if entry_touched:
        return ("FAIL", last_close) if is_expired else ("IN_PROGRESS", last_close)
    return ("FAIL", last_close) if is_expired else ("UNFILLED", last_close)


async def _sim_fetch_kline_cache(symbols: List[str], now_ts: float) -> Dict[str, List[Any]]:
    if not symbols:
        return {}
    async def _fetch_one(sym: str) -> tuple[str, List[Any]]:
        ks: List[Any] = []
        # 기본은 spot, 실패 시 futures 폴백 (예: CROSSUSDT)
        try:
            ks = await app.state.binance.klines(symbol=sym, interval="5m", limit=1000, now_ts=now_ts, market="spot")
        except Exception:
            try:
                ks = await app.state.binance.klines(
                    symbol=sym, interval="5m", limit=1000, now_ts=now_ts, market="futures"
                )
            except Exception:
                ks = []
        return sym, list(ks)

    tasks = [_fetch_one(sym) for sym in symbols]
    pairs = await asyncio.gather(*tasks, return_exceptions=True)
    out: Dict[str, List[Any]] = {}
    for item in pairs:
        if isinstance(item, Exception):
            continue
        sym, ks = item
        out[sym] = ks
    for sym in symbols:
        out.setdefault(sym, [])
    return out


@app.post("/api/sim/trades")
async def api_sim_create_trade(
    payload: Dict[str, Any] = Body(...),
):
    public_user = await _sim_get_or_create_public_user()
    nickname = str(payload.get("nickname", "")).strip()[:20] or "guest"
    symbol = str(payload.get("symbol", "")).upper().strip()
    try:
        entry = float(payload.get("entry"))
        tp = float(payload.get("take_profit"))
        sl = float(payload.get("stop_loss"))
    except Exception:
        raise HTTPException(status_code=400, detail="entry/take_profit/stop_loss must be numbers")
    if symbol.endswith("USDT") is False:
        raise HTTPException(status_code=400, detail="symbol must be usdt pair")
    if min(entry, tp, sl) <= 0:
        raise HTTPException(status_code=400, detail="price must be positive")
    if not (tp > entry > sl):
        raise HTTPException(status_code=400, detail="가격 관계가 맞지 않습니다. 익절가는 진입가 이상, 손절가는 진입가 이하로 입력해주세요.")
    row_base = {
        "user_id": public_user.get("id"),
        "nickname": nickname,
        "symbol": symbol,
        "entry_price": entry,
        "take_profit": tp,
        "stop_loss": sl,
        "status": "OPEN",
        "result_price": None,
        "resolved_ms": None,
        "created_ms": int(time() * 1000),
    }
    try:
        rows = await _sb_request(
            "POST",
            "/rest/v1/sim_trades",
            json_body=[row_base],
        )
    except HTTPException as first_error:
        # 구 스키마 호환: 첫 저장 실패 시 legacy 컬럼 포함으로 1회 재시도
        row_legacy = dict(row_base)
        row_legacy["market"] = "spot"
        row_legacy["interval"] = "5m"
        row_legacy["side"] = "BUY"
        try:
            rows = await _sb_request("POST", "/rest/v1/sim_trades", json_body=[row_legacy])
        except HTTPException:
            raise first_error
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=500, detail="failed to create trade")
    return {"ok": True, "trade": rows[0]}


@app.get("/api/sim/trades")
async def api_sim_list_trades(
    symbol: str = Query("", max_length=20),
    limit: int = Query(20, ge=1, le=100),
    nickname: str = Query("", max_length=20),
    page: int = Query(1, ge=1, le=10000),
    status_filter: str = Query("", alias="status", max_length=20),
    sync_updates: bool = Query(False, alias="sync"),
):
    public_user = await _sim_get_or_create_public_user()
    status_f = status_filter.strip().upper()
    open_state_filter = status_f in {"UNFILLED", "IN_PROGRESS"}
    per_page = max(1, min(int(limit), 100))
    fetch_n = per_page + 1
    offset = (int(page) - 1) * per_page
    params = {
        "select": "id,user_id,nickname,symbol,entry_price,take_profit,stop_loss,status,result_price,created_ms,resolved_ms",
        "user_id": f"eq.{public_user.get('id')}",
        "order": "created_ms.desc",
        "limit": str(1500 if open_state_filter else fetch_n),
        "offset": str(0 if open_state_filter else offset),
    }
    symbol_u = symbol.upper().strip()
    if symbol_u:
        params["symbol"] = f"eq.{symbol_u}"
    nn = nickname.strip()
    if nn:
        params["nickname"] = f"eq.{nn}"
    if status_f in {"TP", "SL", "FAIL"}:
        params["status"] = f"eq.{status_f}"
    rows = await _sb_request("GET", "/rest/v1/sim_trades", params=params)
    trades = rows if isinstance(rows, list) else []

    touched: Dict[str, List[int]] = {}
    now_ms = int(time() * 1000)
    recheck_window_ms = 10 * 60 * 1000
    for tr in trades:
        st = str(tr.get("status", "IN_PROGRESS")).upper()
        created_ms = int(tr.get("created_ms", 0) or 0)
        should_recheck_final = st in {"TP", "SL"} and created_ms > 0 and (now_ms - created_ms) <= recheck_window_ms
        if st in {"FAIL"} or (st in {"TP", "SL"} and not should_recheck_final):
            continue
        key = str(tr.get("symbol", ""))
        touched.setdefault(key, []).append(int(tr.get("id")))

    now_ts = time()
    kline_cache = await _sim_fetch_kline_cache(list(touched.keys()), now_ts)

    out: List[Dict[str, Any]] = []
    for tr in trades:
        cur = dict(tr)
        cur_status = str(cur.get("status", "IN_PROGRESS")).upper()
        created_ms = int(cur.get("created_ms", 0) or 0)
        should_recheck_final = cur_status in {"TP", "SL"} and created_ms > 0 and (now_ms - created_ms) <= recheck_window_ms
        if cur_status not in {"TP", "SL", "FAIL"} or should_recheck_final:
            key = str(cur.get("symbol", ""))
            eval_status, last_close = _sim_eval_trade_status(cur, kline_cache.get(key, []))
            if eval_status != cur_status:
                cur["status"] = eval_status
                if eval_status in {"TP", "SL", "FAIL"}:
                    cur["result_price"] = last_close
                    cur["resolved_ms"] = int(time() * 1000)
                else:
                    cur["result_price"] = None
                    cur["resolved_ms"] = None
                if sync_updates:
                    await _sb_request(
                        "PATCH",
                        "/rest/v1/sim_trades",
                        params={"id": f"eq.{cur.get('id')}"},
                        json_body={
                            "status": eval_status,
                            "result_price": cur.get("result_price"),
                            "resolved_ms": cur.get("resolved_ms"),
                        },
                    )
            else:
                cur["status"] = eval_status
        out.append(cur)

    if status_f:
        out = [x for x in out if str(x.get("status", "")).upper() == status_f]
    if open_state_filter:
        start = offset
        end = offset + per_page
        has_next = len(out) > end
        page_rows = out[start:end]
    else:
        has_next = len(out) > per_page
        page_rows = out[:per_page]
    return {"ok": True, "trades": page_rows, "page": page, "limit": per_page, "has_next": has_next}


@app.get("/api/sim/stats")
async def api_sim_stats(
    sync_updates: bool = Query(False, alias="sync"),
):
    public_user = await _sim_get_or_create_public_user()
    params = {
        "select": "id,symbol,entry_price,take_profit,stop_loss,status,created_ms,resolved_ms,result_price",
        "user_id": f"eq.{public_user.get('id')}",
        "order": "created_ms.desc",
        "limit": "5000",
    }
    rows = await _sb_request("GET", "/rest/v1/sim_trades", params=params)
    trades = rows if isinstance(rows, list) else []

    touched: Dict[str, List[int]] = {}
    for tr in trades:
        st = str(tr.get("status", "")).upper()
        if st in {"TP", "SL", "FAIL"}:
            continue
        sym = str(tr.get("symbol", "")).upper()
        if not sym:
            continue
        touched.setdefault(sym, []).append(int(tr.get("id")))

    now_ts = time()
    kline_cache = await _sim_fetch_kline_cache(list(touched.keys()), now_ts)

    normalized: List[Dict[str, Any]] = []
    for tr in trades:
        cur = dict(tr)
        st = str(cur.get("status", "")).upper()
        if st in {"TP", "SL", "FAIL"}:
            normalized.append(cur)
            continue
        sym = str(cur.get("symbol", "")).upper()
        eval_status, last_close = _sim_eval_trade_status(cur, kline_cache.get(sym, []))
        cur["status"] = eval_status
        if eval_status in {"TP", "SL", "FAIL"}:
            cur["result_price"] = last_close
            cur["resolved_ms"] = int(time() * 1000)
        if sync_updates and eval_status != st:
            await _sb_request(
                "PATCH",
                "/rest/v1/sim_trades",
                params={"id": f"eq.{cur.get('id')}"},
                json_body={
                    "status": eval_status,
                    "result_price": cur.get("result_price"),
                    "resolved_ms": cur.get("resolved_ms"),
                },
            )
        normalized.append(cur)

    agg: Dict[str, Dict[str, Any]] = {}
    for tr in normalized:
        sym = str(tr.get("symbol", "")).upper()
        if not sym:
            continue
        st = str(tr.get("status", "")).upper()
        if sym not in agg:
            agg[sym] = {
                "symbol": sym,
                "total": 0,
                "tp": 0,
                "sl": 0,
                "fail": 0,
                "unfilled": 0,
                "in_progress": 0,
                "win_rate": 0.0,
            }
        a = agg[sym]
        a["total"] += 1
        if st == "TP":
            a["tp"] += 1
        elif st == "SL":
            a["sl"] += 1
        elif st == "FAIL":
            a["fail"] += 1
        elif st == "UNFILLED":
            a["unfilled"] += 1
        else:
            a["in_progress"] += 1

    items = list(agg.values())
    for a in items:
        done = int(a["tp"]) + int(a["sl"]) + int(a["fail"])
        a["win_rate"] = round((float(a["tp"]) / done) * 100.0, 1) if done > 0 else 0.0
    items.sort(key=lambda x: (int(x.get("total", 0)), str(x.get("symbol", ""))), reverse=True)
    return {"ok": True, "stats": items}


@app.get("/api/usdt_krw")
async def api_usdt_krw():
    rate = await _get_usdt_krw()
    return {"usdt_krw": round(rate, 4)}


@app.get("/api/news_sentiment")
async def api_news_sentiment():
    now = time()
    cached = _NEWS_CACHE.get("data")
    if cached is not None and (now - float(_NEWS_CACHE.get("ts", 0.0))) < 300:
        return {"asof_ms": int(now * 1000), "cached": True, "symbols": cached}

    symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "SOLUSDT", "CROSSUSDT"]
    data = await fetch_news_sentiment(symbols=symbols, limit_items=160, timeout_s=6.0, lookback_hours=24)
    _NEWS_CACHE["ts"] = now
    _NEWS_CACHE["data"] = data
    return {"asof_ms": int(now * 1000), "cached": False, "symbols": data}


@app.get("/api/klines")
async def api_klines(
    symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
    interval: str = Query("5m"),
    market: str = Query("spot"),
    limit: int = Query(500, ge=50, le=1500),
):
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
    if market not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail=f"market must be one of {sorted(ALLOWED_MARKETS)}")
    now_ts = time()
    try:
        if int(limit) <= 1500:
            klines = await app.state.binance.klines(
                symbol=symbol, interval=interval, limit=limit, now_ts=now_ts, market=market
            )
        else:
            klines = await _fetch_klines_paged_for_pass_check(
                symbol=symbol, interval=interval, market=market, total_limit=int(limit)
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to fetch binance data: {e}") from e

    out = []
    for k in klines:
        out.append(
            {
                "time": k.open_time_ms,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
        )
    return {"symbol": symbol.upper(), "interval": interval, "market": market, "klines": out}


@app.get("/api/analysis")
async def api_analysis(
    symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
    interval: str = Query("5m"),
    market: str = Query("spot"),
    limit: int = Query(500, ge=60, le=1500),
):
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
    if market not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail=f"market must be one of {sorted(ALLOWED_MARKETS)}")

    now_ts = time()
    try:
        klines = await app.state.binance.klines(
            symbol=symbol, interval=interval, limit=limit, now_ts=now_ts, market=market
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to fetch binance data: {e}") from e

    df = _klines_to_df(klines)
    ind = compute_indicators(df)
    if ind is None:
        raise HTTPException(status_code=400, detail="not enough data to compute indicators")

    close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    score = score_signal(close=close, prev_close=prev_close, ind=ind)
    regime = classify_regime(df)
    avwap_levels, vp = _build_levels(df)

    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "market": market,
        "asof_open_time_ms": int(df["open_time_ms"].iloc[-1]),
        "close": close,
        "regime": regime.regime,
        "buy_pct": score.buy_pct,
        "sell_pct": score.sell_pct,
        "confidence": score.confidence,
        "reasons": score.reasons,
        "regime_features": {
            "adx14": regime.adx14,
            "atr_pct": regime.atr_pct,
            "bb_width_pct": regime.bb_width_pct,
            "ema_gap_ratio": regime.ema_gap_ratio,
        },
        "levels": {
            "avwap": avwap_levels,
            "volume_profile": vp,
        },
        "indicators": {
            "ema20": ind.ema20,
            "ema50": ind.ema50,
            "rsi14": ind.rsi14,
            "macd": ind.macd,
            "macd_signal": ind.macd_signal,
            "macd_hist": ind.macd_hist,
            "bb_mid": ind.bb_mid,
            "bb_upper": ind.bb_upper,
            "bb_lower": ind.bb_lower,
            "atr14": ind.atr14,
            "stoch_k": ind.stoch_k,
            "stoch_d": ind.stoch_d,
        },
    }
    payload["explain"] = build_single_explain(
        buy_pct=payload["buy_pct"],
        sell_pct=payload["sell_pct"],
        confidence=payload["confidence"],
        regime=payload["regime"],
        close=payload["close"],
        indicators=payload["indicators"],
        levels=payload["levels"],
        regime_features=payload["regime_features"],
        feature_contrib=payload.get("feature_contrib"),
        feature_raw=payload.get("feature_raw"),
        feature_weights=payload.get("feature_weights"),
        feature_detail=payload.get("feature_detail"),
        score_x=payload.get("score_x"),
    )
    return payload


@app.get("/api/analysis_trendy")
async def api_analysis_trendy(
    symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
    interval: str = Query("5m"),
    market: str = Query("spot"),
    limit: int = Query(500, ge=60, le=1500),
):
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
    if market not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail=f"market must be one of {sorted(ALLOWED_MARKETS)}")

    now_ts = time()
    try:
        klines = await app.state.binance.klines(
            symbol=symbol, interval=interval, limit=limit, now_ts=now_ts, market=market
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to fetch binance data: {e}") from e

    df = _klines_to_df(klines)
    ind = compute_indicators(df)
    if ind is None:
        raise HTTPException(status_code=400, detail="not enough data to compute indicators")

    close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    regime = classify_regime(df)
    avwap_levels, vp = _build_levels(df)
    score = score_signal_trendy(
        close=close,
        prev_close=prev_close,
        ind=ind,
        regime=regime.regime,
        avwap_levels=avwap_levels,
        volume_profile=vp,
    )
    raw_buy = score.buy_pct / 100.0
    buy_prob, calibrated, calibration_method = _apply_calibration(raw_buy)

    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "market": market,
        "asof_open_time_ms": int(df["open_time_ms"].iloc[-1]),
        "close": close,
        "model": "trendy_v1",
        "regime": regime.regime,
        "buy_pct": round(buy_prob * 100.0, 2),
        "sell_pct": round((1.0 - buy_prob) * 100.0, 2),
        "confidence": score.confidence,
        "raw_buy_pct": round(raw_buy * 100.0, 2),
        "calibrated": calibrated,
        "calibration_method": calibration_method,
        "reasons": score.reasons,
        "feature_contrib": score.feature_contrib or {},
        "feature_raw": score.feature_raw or {},
        "feature_weights": score.feature_weights or {},
        "feature_detail": score.feature_detail or {},
        "score_x": score.score_x,
        "regime_features": {
            "adx14": regime.adx14,
            "atr_pct": regime.atr_pct,
            "bb_width_pct": regime.bb_width_pct,
            "ema_gap_ratio": regime.ema_gap_ratio,
        },
        "levels": {
            "avwap": avwap_levels,
            "volume_profile": vp,
        },
        "indicators": {
            "ema20": ind.ema20,
            "ema50": ind.ema50,
            "rsi14": ind.rsi14,
            "macd": ind.macd,
            "macd_signal": ind.macd_signal,
            "macd_hist": ind.macd_hist,
            "bb_mid": ind.bb_mid,
            "bb_upper": ind.bb_upper,
            "bb_lower": ind.bb_lower,
            "atr14": ind.atr14,
            "stoch_k": ind.stoch_k,
            "stoch_d": ind.stoch_d,
        },
    }
    payload["explain"] = build_single_explain(
        buy_pct=payload["buy_pct"],
        sell_pct=payload["sell_pct"],
        confidence=payload["confidence"],
        regime=payload["regime"],
        close=payload["close"],
        indicators=payload["indicators"],
        levels=payload["levels"],
        regime_features=payload["regime_features"],
        feature_contrib=payload.get("feature_contrib"),
        feature_raw=payload.get("feature_raw"),
        feature_weights=payload.get("feature_weights"),
        feature_detail=payload.get("feature_detail"),
        score_x=payload.get("score_x"),
    )
    return payload


@app.get("/api/analysis_trendy_mtf")
async def api_analysis_trendy_mtf(
    symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
    market: str = Query("spot"),
    limit: int = Query(500, ge=120, le=1500),
):
    if market not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail=f"market must be one of {sorted(ALLOWED_MARKETS)}")
    now_ts = time()
    try:
        k4h, k1h, k5m = await app.state.binance.klines(
            symbol=symbol, interval="4h", limit=limit, now_ts=now_ts, market=market
        ), await app.state.binance.klines(
            symbol=symbol, interval="1h", limit=limit, now_ts=now_ts, market=market
        ), await app.state.binance.klines(
            symbol=symbol, interval="5m", limit=limit, now_ts=now_ts, market=market
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to fetch binance data: {e}") from e

    dfs = {
        "4h": _klines_to_df(k4h),
        "1h": _klines_to_df(k1h),
        "5m": _klines_to_df(k5m),
    }

    tf_result: Dict[str, Any] = {}
    tf_buy_prob: Dict[str, float] = {}
    tf_regimes: Dict[str, str] = {}
    tf_weight = {"4h": 0.20, "1h": 0.35, "5m": 0.45}

    for tf in ("4h", "1h", "5m"):
        df = dfs[tf]
        ind = compute_indicators(df)
        if ind is None:
            raise HTTPException(status_code=400, detail=f"not enough data for {tf} indicators")
        close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        regime = classify_regime(df)
        avwap_levels, vp = _build_levels(df)
        score = score_signal_trendy(
            close=close,
            prev_close=prev_close,
            ind=ind,
            regime=regime.regime,
            avwap_levels=avwap_levels,
            volume_profile=vp,
        )
        raw_buy = score.buy_pct / 100.0
        tf_buy_prob[tf] = raw_buy
        tf_regimes[tf] = regime.regime
        tf_indicators = {
            "ema20": ind.ema20,
            "ema50": ind.ema50,
            "rsi14": ind.rsi14,
            "macd": ind.macd,
            "macd_signal": ind.macd_signal,
            "macd_hist": ind.macd_hist,
            "bb_mid": ind.bb_mid,
            "bb_upper": ind.bb_upper,
            "bb_lower": ind.bb_lower,
            "atr14": ind.atr14,
            "stoch_k": ind.stoch_k,
            "stoch_d": ind.stoch_d,
        }
        tf_regime_features = {
            "adx14": regime.adx14,
            "atr_pct": regime.atr_pct,
            "bb_width_pct": regime.bb_width_pct,
            "ema_gap_ratio": regime.ema_gap_ratio,
        }
        tf_levels = {"avwap": avwap_levels, "volume_profile": vp}
        tf_result[tf] = {
            "buy_pct": round(raw_buy * 100.0, 2),
            "sell_pct": round((1.0 - raw_buy) * 100.0, 2),
            "regime": regime.regime,
            "close": close,
            "asof_open_time_ms": int(df["open_time_ms"].iloc[-1]),
            "regime_features": tf_regime_features,
            "levels": tf_levels,
            "indicators": tf_indicators,
            "feature_contrib": score.feature_contrib or {},
            "feature_raw": score.feature_raw or {},
            "feature_weights": score.feature_weights or {},
            "feature_detail": score.feature_detail or {},
            "score_x": score.score_x,
            "reasons": score.reasons[:6],
        }
        tf_result[tf]["explain"] = build_single_explain(
            buy_pct=tf_result[tf]["buy_pct"],
            sell_pct=tf_result[tf]["sell_pct"],
            confidence=min(1.0, abs(raw_buy - 0.5) * 2.0),
            regime=tf_result[tf]["regime"],
            close=tf_result[tf]["close"],
            indicators=tf_indicators,
            levels=tf_levels,
            regime_features=tf_regime_features,
            feature_contrib=tf_result[tf]["feature_contrib"],
            feature_raw=tf_result[tf]["feature_raw"],
            feature_weights=tf_result[tf]["feature_weights"],
            feature_detail=tf_result[tf]["feature_detail"],
            score_x=tf_result[tf]["score_x"],
        )

    # 4h 필터 성격: 4h 확률에 따라 5m/1h 결합값의 방향성을 조정
    base = tf_weight["4h"] * tf_buy_prob["4h"] + tf_weight["1h"] * tf_buy_prob["1h"] + tf_weight["5m"] * tf_buy_prob["5m"]
    filter_shift = (tf_buy_prob["4h"] - 0.5) * 0.28
    agreement_raw = (
        (tf_buy_prob["4h"] - 0.5) + (tf_buy_prob["1h"] - 0.5) + (tf_buy_prob["5m"] - 0.5)
    ) / 3.0
    agreement_shift = max(-0.06, min(0.06, agreement_raw * 0.22))
    raw_buy = min(1.0, max(0.0, base + filter_shift + agreement_shift))
    buy_prob, calibrated, calibration_method = _apply_calibration(raw_buy)
    confidence = min(1.0, abs(raw_buy - 0.5) * 2.0)

    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "market": market,
        "model": "trendy_mtf_v1",
        "timeframes": ["4h", "1h", "5m"],
        "buy_pct": round(buy_prob * 100.0, 2),
        "sell_pct": round((1.0 - buy_prob) * 100.0, 2),
        "raw_buy_pct": round(raw_buy * 100.0, 2),
        "confidence": round(confidence, 3),
        "calibrated": calibrated,
        "calibration_method": calibration_method,
        "tf_weights": tf_weight,
        "tf": tf_result,
        "meta": {
            "mtf_rule": "4h filter + 1h setup + 5m trigger (weighted blend)",
            "agreement_shift": round(agreement_shift, 6),
        },
    }
    payload["explain"] = build_mtf_explain(
        buy_pct=payload["buy_pct"],
        sell_pct=payload["sell_pct"],
        confidence=payload["confidence"],
        tf_weights=tf_weight,
        tf_regimes=tf_regimes,
        tf_buy_prob=tf_buy_prob,
        filter_shift=filter_shift,
        agreement_shift=agreement_shift,
        raw_buy=raw_buy,
    )
    return payload


@app.get("/api/pass_check")
async def api_pass_check(
    symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
    interval: str = Query("5m"),
    market: str = Query("spot"),
    limit: int = Query(700, ge=220, le=5000),
    horizon_bars: int = Query(24, ge=3, le=3000),
    entry_window_bars: int = Query(6, ge=1, le=24),
):
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
    if market not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail=f"market must be one of {sorted(ALLOWED_MARKETS)}")
    # 실시간 계산 금지: horizon을 period로 매핑해 DB 통계만 반환
    period = _horizon_to_period(interval, int(horizon_bars))
    payload = await api_pass_check_db(symbol=symbol, interval=interval, market=market, period=period)
    out = dict(payload)
    out["cached"] = True
    out["horizon_bars"] = int(horizon_bars)
    out["entry_window_bars"] = int(entry_window_bars)
    out["limit"] = int(limit)
    out["source"] = "db_only"
    return out


@app.get("/api/pass_check_db")
async def api_pass_check_db(
    symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
    interval: str = Query("5m"),
    market: str = Query("spot"),
    period: str = Query("3d", pattern="^(24h|3d|7d)$"),
):
    symbol_u = symbol.upper()
    market_u = market.lower()
    interval_u = interval if interval in ALLOWED_INTERVALS else "5m"
    # 1) 요청 interval 요약 통계만 조회 (5m 강제 대체 금지)
    try:
        summary = await _pc_get_summary(symbol_u, market_u, interval_u, period)
    except HTTPException:
        summary = None
    try:
        baseline = await _pc_get_entry_baseline_counts(symbol_u, market_u, interval_u)
    except HTTPException:
        baseline = None
    if summary:
        summary_pass = int(summary.get("pass_count", 0) or 0)
        summary_exec = int(summary.get("executed_count", 0) or 0)
        summary_no_entry = int(summary.get("no_entry_count", 0) or 0)
        if baseline and int(baseline.get("pass_count", 0) or 0) > 0:
            pass_count = int(baseline.get("pass_count", 0) or 0)
            executed_count = int(baseline.get("executed_count", 0) or 0)
            no_entry_count = int(baseline.get("no_entry_count", 0) or 0)
            first_signal_time_ms = int(baseline.get("first_signal_time_ms", 0) or 0)
            latest_signal_time_ms = int(baseline.get("latest_signal_time_ms", 0) or 0)
        else:
            pass_count = summary_pass
            executed_count = summary_exec
            no_entry_count = summary_no_entry
            first_signal_time_ms = 0
            if pass_count > 0:
                try:
                    first_signal_time_ms = await _pc_get_first_signal_time_ms(symbol_u, market_u, interval_u, period)
                except HTTPException:
                    first_signal_time_ms = 0
            latest_signal_time_ms = int(summary.get("latest_signal_time_ms", 0) or 0)
        return _build_pass_check_payload(
            symbol=symbol_u,
            market=market_u,
            interval=interval_u,
            period=period,
            pass_count=pass_count,
            executed_count=executed_count,
            no_entry_count=no_entry_count,
            tp1_hit_count=int(summary.get("tp1_hit_count", 0) or 0),
            sl_hit_count=int(summary.get("sl_hit_count", 0) or 0),
            no_hit_count=int(summary.get("no_hit_count", 0) or 0),
            resolved_count=int(summary.get("resolved_count", 0) or 0),
            first_signal_time_ms=first_signal_time_ms,
            latest_signal_time_ms=latest_signal_time_ms,
            updated_ms=int(summary.get("updated_ms", 0) or 0),
            source="db_summary",
        )

    # 2) 요약이 없으면 기존 이벤트 집계로 fallback
    start_ms = _pc_window_start_ms(90)
    rows = await _sb_request(
        "GET",
        "/rest/v1/pass_check_events",
        params={
            "select": "signal_time_ms,executed,result",
            "symbol": f"eq.{symbol_u}",
            "market": f"eq.{market_u}",
            "interval": f"eq.{interval_u}",
            "period": f"eq.{period}",
            "signal_time_ms": f"gte.{start_ms}",
            "order": "signal_time_ms.desc",
            "limit": "100000",
        },
    )
    events = rows if isinstance(rows, list) else []
    pass_cnt = len(events)
    executed = [e for e in events if bool(e.get("executed"))]
    executed_cnt = len(executed)
    tp_hits = sum(1 for e in executed if str(e.get("result", "")).upper() == "TP1_HIT")
    sl_hits = sum(1 for e in executed if str(e.get("result", "")).upper() == "SL_HIT")
    no_hits = sum(1 for e in executed if str(e.get("result", "")).upper() == "NO_HIT")
    no_entry = sum(1 for e in events if not bool(e.get("executed")))
    resolved = tp_hits + sl_hits
    base_pass = pass_cnt
    base_exec = executed_cnt
    base_no_entry = no_entry
    base_first_signal = int(events[-1]["signal_time_ms"]) if events else 0
    base_latest_signal = int(events[0]["signal_time_ms"]) if events else 0
    if baseline and int(baseline.get("pass_count", 0) or 0) > 0:
        base_pass = int(baseline.get("pass_count", 0) or 0)
        base_exec = int(baseline.get("executed_count", 0) or 0)
        base_no_entry = int(baseline.get("no_entry_count", 0) or 0)
        base_first_signal = int(baseline.get("first_signal_time_ms", 0) or 0)
        base_latest_signal = int(baseline.get("latest_signal_time_ms", 0) or 0)

    payload = _build_pass_check_payload(
        symbol=symbol_u,
        market=market_u,
        interval=interval_u,
        period=period,
        pass_count=base_pass,
        executed_count=base_exec,
        no_entry_count=base_no_entry,
        tp1_hit_count=tp_hits,
        sl_hit_count=sl_hits,
        no_hit_count=no_hits,
        resolved_count=resolved,
        first_signal_time_ms=base_first_signal,
        latest_signal_time_ms=base_latest_signal,
        updated_ms=int(time() * 1000),
        source="db_events_fallback",
    )
    # fallback 집계 결과를 summary로 시드 (실패해도 본 응답은 반환)
    try:
        now_ms = int(time() * 1000)
        body = {
            "symbol": symbol_u,
            "market": market_u,
            "interval": interval_u,
            "period": period,
            "pass_count": int(pass_cnt),
            "executed_count": int(executed_cnt),
            "no_entry_count": int(no_entry),
            "tp1_hit_count": int(tp_hits),
            "sl_hit_count": int(sl_hits),
            "no_hit_count": int(no_hits),
            "resolved_count": int(resolved),
            "latest_signal_time_ms": int(events[0]["signal_time_ms"]) if events else 0,
            "updated_ms": now_ms,
        }
        existing = await _pc_get_summary(symbol_u, market_u, interval_u, period)
        if existing:
            await _sb_request("PATCH", "/rest/v1/pass_check_summary", params={"id": f"eq.{existing.get('id')}"}, json_body=body)
        else:
            await _sb_request("POST", "/rest/v1/pass_check_summary", json_body=[body])
    except HTTPException:
        pass
    return payload


@app.post("/api/pass_check_batch")
async def api_pass_check_batch(
    payload: Dict[str, Any] = Body(default={}),
):
    market = str(payload.get("market", "spot")).lower().strip()
    if market not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail=f"market must be one of {sorted(ALLOWED_MARKETS)}")
    symbols_in = payload.get("symbols")
    if isinstance(symbols_in, list) and symbols_in:
        symbols = [str(x).upper().strip() for x in symbols_in if str(x).strip()]
    else:
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "SOLUSDT", "CROSSUSDT"]
    interval = str(payload.get("interval", "5m"))
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
    periods = payload.get("periods")
    if not isinstance(periods, list) or not periods:
        periods = ["24h", "3d", "7d"]
    periods = [str(p) for p in periods if str(p) in {"24h", "3d", "7d"}]
    if not periods:
        periods = ["24h", "3d", "7d"]
    seed_days = int(payload.get("seed_days", 90) or 90)
    seed_days = max(30, min(seed_days, 3660))
    bootstrap_year = bool(payload.get("bootstrap_year", True))
    bootstrap_signal_step = int(payload.get("bootstrap_signal_step", 12) or 12)
    bootstrap_signal_step = max(1, min(bootstrap_signal_step, 48))
    initial_signal_step = int(payload.get("initial_signal_step", 3) or 3)
    initial_signal_step = max(1, min(initial_signal_step, 24))

    updated: List[Dict[str, Any]] = []
    now_ms = int(time() * 1000)
    for sym in symbols:
        # 심볼당 캔들은 1회만 조회하고, 기간별(24h/3d/7d) 계산에 재사용한다.
        used_market = market
        max_horizon = max(_period_to_horizon_bars(interval, p) for p in periods)
        base_limit = max(220 + max_horizon + 320, max_horizon + 1200)
        try:
            progress_probe = await _pc_get_progress(sym, used_market, interval, periods[0])
            has_progress = bool(progress_probe and int(progress_probe.get("last_signal_time_ms", 0)) > 0)
            limit = base_limit if has_progress else max(base_limit, _bars_for_days(interval, seed_days) + max_horizon + 260)
            if not bootstrap_year and not has_progress:
                # 초기 얕은 적재는 과도한 봉 조회를 제한해 응답 지연을 줄인다.
                limit = min(base_limit, max_horizon + 900)
            try:
                klines = await _fetch_klines_paged_for_pass_check(
                    symbol=sym, interval=interval, market=used_market, total_limit=limit
                )
            except Exception:
                if used_market == "spot":
                    used_market = "futures"
                    progress_probe = await _pc_get_progress(sym, used_market, interval, periods[0])
                    has_progress = bool(progress_probe and int(progress_probe.get("last_signal_time_ms", 0)) > 0)
                    limit = base_limit if has_progress else max(base_limit, _bars_for_days(interval, seed_days) + max_horizon + 260)
                    if not bootstrap_year and not has_progress:
                        limit = min(base_limit, max_horizon + 900)
                    klines = await _fetch_klines_paged_for_pass_check(
                        symbol=sym, interval=interval, market=used_market, total_limit=limit
                    )
                else:
                    raise
        except Exception as e:
            for period in periods:
                updated.append(
                    {
                        "symbol": sym,
                        "market": used_market,
                        "interval": interval,
                        "period": period,
                        "new_events": 0,
                        "from_signal_time_ms": 0,
                        "seed_days": int(seed_days),
                        "bootstrap_year": bool(bootstrap_year),
                        "bootstrap_signal_step": 0,
                        "error": f"symbol_init_failed:{type(e).__name__}",
                    }
                )
            _compact_runtime_memory()
            continue
        df = _klines_to_df(klines)

        for period in periods:
            try:
                horizon_bars = _period_to_horizon_bars(interval, period)
                progress = await _pc_get_progress(sym, used_market, interval, period)
                last_signal_ms = int(progress.get("last_signal_time_ms", 0)) if progress else 0
                if bootstrap_year and last_signal_ms <= 0:
                    signal_step = bootstrap_signal_step
                elif (not bootstrap_year) and last_signal_ms <= 0 and interval == "5m":
                    signal_step = initial_signal_step
                else:
                    signal_step = 1
                calc = _calc_pass_check_from_df(
                    df=df,
                    symbol=sym,
                    market=used_market,
                    interval=interval,
                    horizon_bars=horizon_bars,
                    entry_window_bars=5,
                    min_signal_time_ms=last_signal_ms,
                    signal_step=signal_step,
                    use_mtf_gate=True,
                    use_volume_filter=True,
                    use_reaction_entry=True,
                )
                events = calc.get("events", [])
                if events:
                    body = []
                    max_signal_ms = last_signal_ms
                    for e in events:
                        stm = int(e.get("signal_time_ms", 0) or 0)
                        if stm <= 0:
                            continue
                        max_signal_ms = max(max_signal_ms, stm)
                        body.append(
                            {
                                "symbol": sym,
                                "market": used_market,
                                "interval": interval,
                                "period": period,
                                "signal_time_ms": stm,
                                "entry_time_ms": e.get("entry_time_ms"),
                                "executed": bool(e.get("executed")),
                                "result": str(e.get("result", "NO_ENTRY")),
                                "side": str(e.get("side", "WAIT")),
                                "entry_price": float(e.get("entry", 0.0) or 0.0),
                                "tp_price": float(e.get("tp1", 0.0) or 0.0),
                                "stop_price": float(e.get("stop", 0.0) or 0.0),
                                "created_ms": now_ms,
                            }
                        )
                    if body:
                        await _sb_request("POST", "/rest/v1/pass_check_events", json_body=body)
                        if progress:
                            await _sb_request(
                                "PATCH",
                                "/rest/v1/pass_check_progress",
                                params={"id": f"eq.{progress.get('id')}"},
                                json_body={"last_signal_time_ms": max_signal_ms, "updated_ms": now_ms},
                            )
                        else:
                            await _sb_request(
                                "POST",
                                "/rest/v1/pass_check_progress",
                                json_body=[
                                    {
                                        "symbol": sym,
                                        "market": used_market,
                                        "interval": interval,
                                        "period": period,
                                        "last_signal_time_ms": max_signal_ms,
                                        "updated_ms": now_ms,
                                    }
                                ],
                            )
                # summary는 이벤트 원본 기준으로 매번 재집계(드리프트 방지)
                try:
                    await _pc_refresh_summary_from_events(sym, used_market, interval, period)
                except HTTPException:
                    pass
                updated.append(
                    {
                        "symbol": sym,
                        "market": used_market,
                        "interval": interval,
                        "period": period,
                        "new_events": int(len(events)),
                        "from_signal_time_ms": int(last_signal_ms),
                        "seed_days": int(seed_days),
                        "bootstrap_year": bool(bootstrap_year),
                        "bootstrap_signal_step": int(signal_step),
                    }
                )
            except Exception as e:
                updated.append(
                    {
                        "symbol": sym,
                        "market": used_market,
                        "interval": interval,
                        "period": period,
                        "new_events": 0,
                        "from_signal_time_ms": 0,
                        "seed_days": int(seed_days),
                        "bootstrap_year": bool(bootstrap_year),
                        "bootstrap_signal_step": 0,
                        "error": f"period_failed:{type(e).__name__}",
                    }
                )
                continue
        # 심볼 단위 계산 완료 시 큰 객체 해제 후 메모리 반환 시도
        del klines
        del df
        _compact_runtime_memory()
    _compact_runtime_memory()
    return {"ok": True, "updated": updated}


@app.get("/api/pass_check_tune")
async def api_pass_check_tune(
    symbol: str = Query("BTCUSDT", min_length=3, max_length=20),
    interval: str = Query("5m"),
    market: str = Query("spot"),
    limit: int = Query(700, ge=220, le=1500),
):
    now_ts = time()
    _trim_ttl_cache(_PASS_TUNE_CACHE, now_ts=now_ts, ttl_s=1800.0)
    _trim_ttl_cache(_PASS_TUNE_PROFILE_CACHE, now_ts=now_ts, ttl_s=1800.0)
    symbol_u = symbol.upper()
    profile = _symbol_profile(symbol_u)
    cache_key = f"{symbol_u}:{market}:{interval}:{limit}"
    profile_key = f"{profile}:{market}:{interval}:{limit}"
    ttl_s = 600.0

    cached_exact = _PASS_TUNE_CACHE.get(cache_key)
    if isinstance(cached_exact, dict) and (now_ts - float(cached_exact.get("ts", 0.0)) <= ttl_s):
        payload = dict(cached_exact.get("payload", {}))
        payload["cached"] = "symbol"
        return payload

    # 과도한 연산을 피하기 위해 소형 그리드만 탐색
    base_grid = [
        {"horizon_bars": 18, "entry_window_bars": 4},
        {"horizon_bars": 18, "entry_window_bars": 6},
        {"horizon_bars": 24, "entry_window_bars": 4},
        {"horizon_bars": 24, "entry_window_bars": 6},
        {"horizon_bars": 30, "entry_window_bars": 6},
        {"horizon_bars": 30, "entry_window_bars": 8},
    ]
    grid = list(base_grid)
    grid_mode = "full"
    profile_hint = _PASS_TUNE_PROFILE_CACHE.get(profile_key)
    if isinstance(profile_hint, dict) and (now_ts - float(profile_hint.get("ts", 0.0)) <= ttl_s):
        b = profile_hint.get("best") or {}
        h = int(b.get("horizon_bars", 24))
        e = int(b.get("entry_window_bars", 6))
        near = [
            {"horizon_bars": max(12, min(36, h - 6)), "entry_window_bars": max(2, min(12, e))},
            {"horizon_bars": h, "entry_window_bars": e},
            {"horizon_bars": h, "entry_window_bars": max(2, min(12, e + 2))},
            {"horizon_bars": min(36, h + 6), "entry_window_bars": e},
        ]
        # 중복 제거
        uniq = {(x["horizon_bars"], x["entry_window_bars"]): x for x in near}
        grid = list(uniq.values())
        grid_mode = "profile_hint"

    results: List[Dict[str, Any]] = []
    for g in grid:
        period = _horizon_to_period(interval, int(g["horizon_bars"]))
        r = await api_pass_check_db(symbol=symbol_u, interval=interval, market=market, period=period)
        executed = int(r.get("executed_count", 0))
        signal_rate = float(r.get("tp1_hit_rate", 0.0))
        exec_rate = float(r.get("executed_tp1_hit_rate", 0.0))
        resolved_rate = float(r.get("resolved_tp1_hit_rate", 0.0))
        # 체결 수가 너무 적은 조합은 점수를 낮춤
        sample_penalty = 0.0 if executed >= 40 else (40 - executed) * 0.004
        score = (exec_rate * 0.55) + (resolved_rate * 0.35) + (signal_rate * 0.10) - sample_penalty
        results.append(
            {
                "horizon_bars": g["horizon_bars"],
                "entry_window_bars": g["entry_window_bars"],
                "score": round(score, 6),
                "pass_count": r.get("pass_count", 0),
                "executed_count": executed,
                "tp1_hit_rate": signal_rate,
                "executed_tp1_hit_rate": exec_rate,
                "resolved_tp1_hit_rate": resolved_rate,
            }
        )
    results.sort(key=lambda x: (x["score"], x["executed_count"]), reverse=True)
    best = results[0] if results else None
    payload = {
        "symbol": symbol_u,
        "interval": interval,
        "market": market,
        "limit": limit,
        "profile": profile,
        "grid_mode": grid_mode,
        "best": best,
        "candidates": results,
        "cached": "none",
    }
    _set_bounded_cache(
        _PASS_TUNE_CACHE,
        cache_key,
        {"ts": now_ts, "payload": payload},
        max_entries=_PASS_TUNE_CACHE_MAX,
    )
    if best is not None:
        _set_bounded_cache(
            _PASS_TUNE_PROFILE_CACHE,
            profile_key,
            {"ts": now_ts, "best": best},
            max_entries=_PASS_TUNE_PROFILE_CACHE_MAX,
        )
    return payload


@app.websocket("/ws/analysis")
async def ws_analysis(websocket: WebSocket):
    await websocket.accept()
    query = parse_qs(websocket.url.query)
    symbol = str(query.get("symbol", ["BTCUSDT"])[0]).upper()
    market = str(query.get("market", ["spot"])[0]).lower()
    interval = str(query.get("interval", ["5m"])[0])
    mode = str(query.get("mode", ["single"])[0]).lower()
    limit = int(str(query.get("limit", ["500"])[0]))
    try:
        while True:
            if market not in ALLOWED_MARKETS:
                try:
                    await websocket.send_json({"error": f"market must be one of {sorted(ALLOWED_MARKETS)}"})
                except WebSocketDisconnect:
                    return
                except RuntimeError:
                    return
                await asyncio.sleep(_WS_ANALYSIS_POLL_S)
                continue

            try:
                cache_key = f"{mode}:{symbol}:{market}:{interval}:{limit}"
                now_ts = time()
                _trim_ttl_cache(
                    _WS_ANALYSIS_CACHE,
                    now_ts=now_ts,
                    ttl_s=max(_WS_ANALYSIS_CACHE_TTL_S * 3.0, 10.0),
                )
                cached = _WS_ANALYSIS_CACHE.get(cache_key)
                if isinstance(cached, dict) and (now_ts - float(cached.get("ts", 0.0))) <= _WS_ANALYSIS_CACHE_TTL_S:
                    payload = cached.get("payload")
                else:
                    if mode == "mtf":
                        payload = await api_analysis_trendy_mtf(symbol=symbol, market=market, limit=limit)
                    else:
                        payload = await api_analysis_trendy(
                            symbol=symbol,
                            interval=interval,
                            market=market,
                            limit=limit,
                        )
                    _set_bounded_cache(
                        _WS_ANALYSIS_CACHE,
                        cache_key,
                        {"ts": now_ts, "payload": payload},
                        max_entries=_WS_ANALYSIS_CACHE_MAX,
                    )
                await websocket.send_json(payload)
            except HTTPException as e:
                try:
                    await websocket.send_json({"error": str(e.detail)})
                except WebSocketDisconnect:
                    return
                except RuntimeError:
                    return
            except Exception as e:
                try:
                    await websocket.send_json({"error": f"analysis stream error: {e}"})
                except WebSocketDisconnect:
                    return
                except RuntimeError:
                    return

            await asyncio.sleep(_WS_ANALYSIS_POLL_S)
    except WebSocketDisconnect:
        return


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    p = (full_path or "").lstrip("/")
    if p.startswith("api/") or p.startswith("ws/") or p.startswith("static/"):
        raise HTTPException(status_code=404, detail="not found")
    index_path = Path("static/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index not found")
    return FileResponse(index_path)
