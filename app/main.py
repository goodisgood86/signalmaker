from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import re
from bisect import bisect_right
from contextlib import asynccontextmanager
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from time import time
from urllib.parse import parse_qs, urlencode
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
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
    global _AUTO_AUDIT_QUEUE
    global _AUTO_AUDIT_TASKS
    cache_max = max(48, int(str(os.getenv("BINANCE_CACHE_MAX_ENTRIES", "96")) or "96"))
    app.state.binance = CachedBinanceClient(ttl_seconds=10.0, cache_max_entries=cache_max)
    _AUTO_AUDIT_QUEUE = asyncio.Queue(maxsize=_AUTO_AUDIT_QUEUE_MAX)
    _AUTO_AUDIT_TASKS = [
        asyncio.create_task(_auto_audit_worker(i + 1)) for i in range(max(1, int(_AUTO_AUDIT_WORKERS)))
    ]
    auto_task: asyncio.Task | None = None
    if _AUTO_BG_ENABLED:
        auto_task = asyncio.create_task(_auto_trade_bg_loop(app))
    app.state.auto_trade_task = auto_task
    try:
        yield
    finally:
        if auto_task is not None:
            auto_task.cancel()
            try:
                await auto_task
            except asyncio.CancelledError:
                pass
        if _AUTO_AUDIT_QUEUE is not None:
            try:
                await asyncio.wait_for(_AUTO_AUDIT_QUEUE.join(), timeout=2.0)
            except Exception:
                pass
        for t in list(_AUTO_AUDIT_TASKS):
            t.cancel()
        for t in list(_AUTO_AUDIT_TASKS):
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        _AUTO_AUDIT_TASKS = []
        _AUTO_AUDIT_QUEUE = None
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
_WS_ANALYSIS_POLL_S = max(3.0, float(str(os.getenv("WS_ANALYSIS_POLL_S", "5")) or "5"))
_WS_ANALYSIS_FORCE_REFRESH_S = max(10.0, float(str(os.getenv("WS_ANALYSIS_FORCE_REFRESH_S", "30")) or "30"))
_PASS_TUNE_CACHE_MAX = max(16, int(str(os.getenv("PASS_TUNE_CACHE_MAX", "32")) or "32"))
_PASS_TUNE_PROFILE_CACHE_MAX = max(16, int(str(os.getenv("PASS_TUNE_PROFILE_CACHE_MAX", "32")) or "32"))
_WS_ANALYSIS_CACHE_MAX = max(16, int(str(os.getenv("WS_ANALYSIS_CACHE_MAX", "40")) or "40"))
_AUTO_ALLOWED_MODES = {"balanced", "aggressive"}
_AUTO_ALLOWED_RECORD_STATUS = {"OPEN", "TP", "SL", "CLOSED_FAIL"}
_AUTO_LINK_STATUS = {"CONNECTED"}
_AUTO_ALLOWED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "SOLUSDT", "CROSSUSDT"}
_AUTH_COOKIE_NAME = "coin_auth_session"
_AUTH_SESSION_TTL_S = max(1800, int(str(os.getenv("APP_AUTH_SESSION_TTL_S", "43200")) or "43200"))
_AUTH_SESSIONS_MAX = max(64, int(str(os.getenv("APP_AUTH_SESSIONS_MAX", "512")) or "512"))
_AUTH_SESSIONS: Dict[str, Dict[str, Any]] = {}
_CFG_UNLOCK_COOKIE_NAME = "coin_cfg_unlock_session"
_CFG_UNLOCK_SESSION_TTL_S = max(600, int(str(os.getenv("APP_CFG_UNLOCK_SESSION_TTL_S", "43200")) or "43200"))
_CFG_UNLOCK_SESSIONS_MAX = max(64, int(str(os.getenv("APP_CFG_UNLOCK_SESSIONS_MAX", "512")) or "512"))
_CFG_UNLOCK_SESSIONS: Dict[str, Dict[str, Any]] = {}
_AUTO_TICK_INTERVAL_S = max(5.0, float(str(os.getenv("AUTO_TRADE_TICK_INTERVAL_S", "15")) or "15"))
_AUTO_BG_ENABLED = str(os.getenv("AUTO_TRADE_BG_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
_AUTO_LIVE_TRADING_ENABLED = str(os.getenv("AUTO_TRADE_LIVE_ENABLED", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_AUTO_TICK_LOCK: asyncio.Lock | None = None
_AUTO_AUDIT_SKIP_UNTIL_MS = 0
_AUTO_AUDIT_QUEUE_MAX = max(200, int(str(os.getenv("AUTO_AUDIT_QUEUE_MAX", "2000")) or "2000"))
_AUTO_AUDIT_WORKERS = max(1, min(4, int(str(os.getenv("AUTO_AUDIT_WORKERS", "1")) or "1")))
_AUTO_AUDIT_QUEUE: asyncio.Queue | None = None
_AUTO_AUDIT_TASKS: List[asyncio.Task] = []
_AUTO_AUDIT_DROP_COUNT = 0
_KST_OFFSET_MS = 9 * 60 * 60 * 1000
_AUTO_LOG = logging.getLogger("coin.auto_trade")


def _auth_expected_hash() -> str:
    h = str(os.getenv("APP_ACCESS_KEY_HASH", "")).strip().lower()
    if h:
        return h
    raw = str(os.getenv("APP_ACCESS_KEY", "")).strip()
    if raw:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return ""


def _auth_enabled() -> bool:
    return bool(_auth_expected_hash())


def _auth_cleanup_sessions(now_ms: int) -> None:
    stale = []
    for sid, sess in _AUTH_SESSIONS.items():
        exp_ms = int(sess.get("exp_ms", 0) or 0)
        if exp_ms <= now_ms:
            stale.append(sid)
    for sid in stale:
        _AUTH_SESSIONS.pop(sid, None)
    overflow = len(_AUTH_SESSIONS) - _AUTH_SESSIONS_MAX
    if overflow > 0:
        for sid in list(_AUTH_SESSIONS.keys())[:overflow]:
            _AUTH_SESSIONS.pop(sid, None)


def _auth_create_session(*, ip: str, ua: str) -> tuple[str, int]:
    now_ms = int(time() * 1000)
    exp_ms = now_ms + (_AUTH_SESSION_TTL_S * 1000)
    sid = secrets.token_urlsafe(32)
    _AUTH_SESSIONS[sid] = {
        "created_ms": now_ms,
        "exp_ms": exp_ms,
        "ip": str(ip or ""),
        "ua": str(ua or ""),
    }
    _auth_cleanup_sessions(now_ms)
    return sid, exp_ms


def _auth_valid_session(sid: str | None) -> bool:
    if not sid:
        return False
    now_ms = int(time() * 1000)
    _auth_cleanup_sessions(now_ms)
    sess = _AUTH_SESSIONS.get(str(sid))
    if not isinstance(sess, dict):
        return False
    exp_ms = int(sess.get("exp_ms", 0) or 0)
    if exp_ms <= now_ms:
        _AUTH_SESSIONS.pop(str(sid), None)
        return False
    return True


def _cfg_unlock_expected_hash() -> str:
    h = str(os.getenv("APP_CONFIG_UNLOCK_KEY_HASH", "")).strip().lower()
    if h:
        return h
    raw = str(os.getenv("APP_CONFIG_UNLOCK_KEY", "")).strip()
    if raw:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # 로컬 기본값(기존 UI 비밀번호) - 운영에서는 반드시 환경변수로 교체 권장
    return "a0a4054e33da8599fda33e887347373e38dc9c99c1ad770ad06292a8a3c89487"


def _cfg_unlock_enabled() -> bool:
    return bool(_cfg_unlock_expected_hash())


def _cfg_unlock_cleanup_sessions(now_ms: int) -> None:
    stale = []
    for sid, sess in _CFG_UNLOCK_SESSIONS.items():
        exp_ms = int(sess.get("exp_ms", 0) or 0)
        if exp_ms <= now_ms:
            stale.append(sid)
    for sid in stale:
        _CFG_UNLOCK_SESSIONS.pop(sid, None)
    overflow = len(_CFG_UNLOCK_SESSIONS) - _CFG_UNLOCK_SESSIONS_MAX
    if overflow > 0:
        for sid in list(_CFG_UNLOCK_SESSIONS.keys())[:overflow]:
            _CFG_UNLOCK_SESSIONS.pop(sid, None)


def _cfg_unlock_create_session(*, ip: str, ua: str) -> tuple[str, int]:
    now_ms = int(time() * 1000)
    exp_ms = now_ms + (_CFG_UNLOCK_SESSION_TTL_S * 1000)
    sid = secrets.token_urlsafe(32)
    _CFG_UNLOCK_SESSIONS[sid] = {
        "created_ms": now_ms,
        "exp_ms": exp_ms,
        "ip": str(ip or ""),
        "ua": str(ua or ""),
    }
    _cfg_unlock_cleanup_sessions(now_ms)
    return sid, exp_ms


def _cfg_unlock_valid_session(sid: str | None) -> bool:
    if not sid:
        return False
    now_ms = int(time() * 1000)
    _cfg_unlock_cleanup_sessions(now_ms)
    sess = _CFG_UNLOCK_SESSIONS.get(str(sid))
    if not isinstance(sess, dict):
        return False
    exp_ms = int(sess.get("exp_ms", 0) or 0)
    if exp_ms <= now_ms:
        _CFG_UNLOCK_SESSIONS.pop(str(sid), None)
        return False
    return True


def _require_cfg_unlock(request: Request) -> None:
    if not _cfg_unlock_enabled():
        return
    sid = request.cookies.get(_CFG_UNLOCK_COOKIE_NAME)
    if _cfg_unlock_valid_session(sid):
        return
    raise HTTPException(status_code=403, detail="config is locked")


def _auto_get_tick_lock() -> asyncio.Lock:
    global _AUTO_TICK_LOCK
    if _AUTO_TICK_LOCK is None:
        _AUTO_TICK_LOCK = asyncio.Lock()
    return _AUTO_TICK_LOCK


def _auth_is_public_path(path: str) -> bool:
    p = str(path or "")
    if p in {"/api/health", "/favicon.ico"}:
        return True
    if p.startswith("/api/auth"):
        return True
    if p.startswith("/static/auth"):
        return True
    return False


@app.middleware("http")
async def app_auth_middleware(request: Request, call_next):
    if not _auth_enabled():
        return await call_next(request)

    path = request.url.path
    sid = request.cookies.get(_AUTH_COOKIE_NAME)
    authed = _auth_valid_session(sid)

    if _auth_is_public_path(path):
        if path == "/static/auth.html" and authed:
            return RedirectResponse(url="/", status_code=302)
        return await call_next(request)

    if authed:
        return await call_next(request)

    if path.startswith("/api/"):
        return JSONResponse(status_code=401, content={"ok": False, "detail": "unauthorized"})
    return RedirectResponse(url="/static/auth.html", status_code=302)


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


async def _ws_analysis_marker(*, symbol: str, market: str, mode: str, interval: str) -> tuple | None:
    now_ts = time()
    try:
        if mode == "mtf":
            k4h, k1h, k5m = await asyncio.gather(
                app.state.binance.klines(symbol=symbol, interval="4h", limit=2, now_ts=now_ts, market=market),
                app.state.binance.klines(symbol=symbol, interval="1h", limit=2, now_ts=now_ts, market=market),
                app.state.binance.klines(symbol=symbol, interval="5m", limit=2, now_ts=now_ts, market=market),
            )
            o4 = int(k4h[-1].open_time_ms) if k4h else 0
            o1 = int(k1h[-1].open_time_ms) if k1h else 0
            o5 = int(k5m[-1].open_time_ms) if k5m else 0
            return ("mtf", o4, o1, o5)

        ks = await app.state.binance.klines(symbol=symbol, interval=interval, limit=2, now_ts=now_ts, market=market)
        ot = int(ks[-1].open_time_ms) if ks else 0
        return ("single", interval, ot)
    except Exception:
        return None


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


@app.get("/api/auth/status")
def api_auth_status(request: Request):
    enabled = _auth_enabled()
    authed = False
    if enabled:
        sid = request.cookies.get(_AUTH_COOKIE_NAME)
        authed = _auth_valid_session(sid)
    return {"ok": True, "enabled": enabled, "authenticated": authed}


@app.post("/api/auth/unlock")
def api_auth_unlock(
    request: Request,
    payload: Dict[str, Any] = Body(default={}),
):
    enabled = _auth_enabled()
    if not enabled:
        return {"ok": True, "enabled": False, "authenticated": True}
    access_key = str(payload.get("access_key", "")).strip()
    if not access_key:
        raise HTTPException(status_code=400, detail="access_key is required")
    expected_hash = _auth_expected_hash()
    supplied_hash = hashlib.sha256(access_key.encode("utf-8")).hexdigest()
    if not secrets.compare_digest(expected_hash, supplied_hash):
        raise HTTPException(status_code=401, detail="invalid access key")

    ip = request.client.host if request.client else ""
    ua = request.headers.get("user-agent", "")
    sid, exp_ms = _auth_create_session(ip=ip, ua=ua)
    resp = JSONResponse(
        {
            "ok": True,
            "enabled": True,
            "authenticated": True,
            "expires_ms": exp_ms,
        }
    )
    resp.set_cookie(
        key=_AUTH_COOKIE_NAME,
        value=sid,
        max_age=_AUTH_SESSION_TTL_S,
        httponly=True,
        samesite="lax",
        secure=bool(request.url.scheme == "https"),
        path="/",
    )
    return resp


@app.post("/api/auth/logout")
def api_auth_logout(
    request: Request,
):
    sid = request.cookies.get(_AUTH_COOKIE_NAME)
    if sid:
        _AUTH_SESSIONS.pop(str(sid), None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(key=_AUTH_COOKIE_NAME, path="/")
    return resp


@app.get("/api/auto_trade/config_lock/status")
def api_auto_trade_config_lock_status(
    request: Request,
):
    enabled = _cfg_unlock_enabled()
    unlocked = False
    if enabled:
        sid = request.cookies.get(_CFG_UNLOCK_COOKIE_NAME)
        unlocked = _cfg_unlock_valid_session(sid)
    return {"ok": True, "enabled": enabled, "unlocked": unlocked}


@app.post("/api/auto_trade/config_lock/unlock")
def api_auto_trade_config_lock_unlock(
    request: Request,
    payload: Dict[str, Any] = Body(default={}),
):
    enabled = _cfg_unlock_enabled()
    if not enabled:
        return {"ok": True, "enabled": False, "unlocked": True}
    password = str(payload.get("password", "")).strip()
    if not password:
        raise HTTPException(status_code=400, detail="password is required")
    expected_hash = _cfg_unlock_expected_hash()
    supplied_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if not secrets.compare_digest(expected_hash, supplied_hash):
        raise HTTPException(status_code=401, detail="invalid password")

    ip = request.client.host if request.client else ""
    ua = request.headers.get("user-agent", "")
    sid, exp_ms = _cfg_unlock_create_session(ip=ip, ua=ua)
    resp = JSONResponse(
        {
            "ok": True,
            "enabled": True,
            "unlocked": True,
            "expires_ms": exp_ms,
        }
    )
    resp.set_cookie(
        key=_CFG_UNLOCK_COOKIE_NAME,
        value=sid,
        max_age=_CFG_UNLOCK_SESSION_TTL_S,
        httponly=True,
        samesite="lax",
        secure=bool(request.url.scheme == "https"),
        path="/",
    )
    return resp


@app.post("/api/auto_trade/config_lock/lock")
def api_auto_trade_config_lock_lock(
    request: Request,
):
    sid = request.cookies.get(_CFG_UNLOCK_COOKIE_NAME)
    if sid:
        _CFG_UNLOCK_SESSIONS.pop(str(sid), None)
    resp = JSONResponse({"ok": True, "enabled": _cfg_unlock_enabled(), "unlocked": False})
    resp.delete_cookie(key=_CFG_UNLOCK_COOKIE_NAME, path="/")
    return resp


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


def _auto_default_config() -> Dict[str, Any]:
    return {
        "enabled": False,
        "mode": "balanced",
        "symbol": "BTCUSDT",
        "market": "spot",
        "interval": "5m",
        "order_size_usdt": 120.0,
        "take_profit_pct": 0.0,
        "stop_loss_pct": 0.0,
        "daily_max_loss_usdt": 120.0,
        "cooldown_min": 30,
        "max_open_positions": 1,
        "last_run_ms": 0,
    }


def _auto_table_missing_error(table_name: str) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail=f"{table_name} table is missing. apply docs/supabase_auto_trade_schema.sql first",
    )


def _auto_is_missing_table_error(err: HTTPException, table_name: str) -> bool:
    msg = str(getattr(err, "detail", "")).lower()
    t = str(table_name or "").lower()
    return t in msg and (
        "42p01" in msg
        or "does not exist" in msg
        or "schema cache" in msg
        or "could not find table" in msg
        or "relation" in msg
    )


def _auto_int_or_none(v: Any) -> int | None:
    try:
        n = int(v)
    except Exception:
        return None
    return n if n > 0 else None


def _auto_float_or_none(v: Any, digits: int = 8) -> float | None:
    try:
        n = float(v)
    except Exception:
        return None
    if not math.isfinite(n):
        return None
    return round(n, digits)


def _auto_to_json_text(payload: Any, *, max_len: int = 3000) -> str:
    if payload is None:
        return ""
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        text = str(payload)
    if len(text) > max_len:
        return f"{text[:max_len]}...(truncated)"
    return text


async def _auto_audit_write_body(body: Dict[str, Any]) -> None:
    global _AUTO_AUDIT_SKIP_UNTIL_MS
    now_ms = int(time() * 1000)
    if _AUTO_AUDIT_SKIP_UNTIL_MS > now_ms:
        return
    try:
        await _sb_request("POST", "/rest/v1/auto_trade_exec_audit", json_body=[body])
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_exec_audit"):
            # 스키마 적용 전 반복 호출 로그 폭주 방지
            _AUTO_AUDIT_SKIP_UNTIL_MS = now_ms + (10 * 60 * 1000)
            return
    except Exception:
        return


async def _auto_audit_worker(worker_no: int) -> None:
    while True:
        q = _AUTO_AUDIT_QUEUE
        if q is None:
            await asyncio.sleep(0.05)
            continue
        body = await q.get()
        try:
            if isinstance(body, dict) and body:
                await _auto_audit_write_body(body)
        except asyncio.CancelledError:
            raise
        except Exception:
            _AUTO_LOG.exception("auto audit worker failed (worker=%s)", worker_no)
        finally:
            q.task_done()


async def _auto_audit_log(
    *,
    user_id: int,
    event: str,
    level: str = "INFO",
    record_id: int | None = None,
    symbol: str = "",
    market: str = "",
    mode: str = "",
    side: str = "",
    status: str = "",
    qty: Any = None,
    price: Any = None,
    pnl_usdt: Any = None,
    order_id: str = "",
    client_order_id: str = "",
    detail: str = "",
    payload: Any = None,
) -> None:
    global _AUTO_AUDIT_DROP_COUNT
    now_ms = int(time() * 1000)
    if _AUTO_AUDIT_SKIP_UNTIL_MS > now_ms:
        return
    body: Dict[str, Any] = {
        "user_id": int(user_id),
        "record_id": _auto_int_or_none(record_id),
        "event": str(event or "UNKNOWN").upper()[:80],
        "level": str(level or "INFO").upper()[:16],
        "symbol": str(symbol or "").upper()[:20] or None,
        "market": str(market or "").lower()[:20] or None,
        "mode": str(mode or "").lower()[:20] or None,
        "side": str(side or "").upper()[:20] or None,
        "status": str(status or "").upper()[:24] or None,
        "qty": _auto_float_or_none(qty),
        "price": _auto_float_or_none(price),
        "pnl_usdt": _auto_float_or_none(pnl_usdt),
        "order_id": str(order_id or "")[:120] or None,
        "client_order_id": str(client_order_id or "")[:120] or None,
        "detail": str(detail or "")[:800] or None,
        "payload": _auto_to_json_text(payload) or None,
        "created_ms": now_ms,
    }
    q = _AUTO_AUDIT_QUEUE
    if q is None:
        await _auto_audit_write_body(body)
        return
    try:
        q.put_nowait(body)
    except asyncio.QueueFull:
        _AUTO_AUDIT_DROP_COUNT += 1
        if _AUTO_AUDIT_DROP_COUNT % 100 == 1:
            _AUTO_LOG.warning("auto audit queue is full; dropping logs (dropped=%s)", _AUTO_AUDIT_DROP_COUNT)


def _auto_clip_number(v: Any, lo: float, hi: float, default_v: float) -> float:
    try:
        n = float(v)
    except Exception:
        n = float(default_v)
    if not math.isfinite(n):
        n = float(default_v)
    return max(float(lo), min(float(hi), float(n)))


def _auto_normalize_config(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    base = _auto_default_config()
    src = dict(raw or {})
    mode = str(src.get("mode", base["mode"])).strip().lower()
    market = str(src.get("market", base["market"])).strip().lower()
    symbol = str(src.get("symbol", base["symbol"])).strip().upper()
    interval = str(src.get("interval", base["interval"])).strip()
    out = dict(base)
    out["enabled"] = bool(src.get("enabled", base["enabled"]))
    out["mode"] = mode if mode in _AUTO_ALLOWED_MODES else base["mode"]
    out["market"] = market if market in ALLOWED_MARKETS else base["market"]
    out["symbol"] = symbol if symbol.endswith("USDT") else base["symbol"]
    out["interval"] = interval if interval in ALLOWED_INTERVALS else base["interval"]
    out["order_size_usdt"] = _auto_clip_number(src.get("order_size_usdt"), 10.0, 100000.0, base["order_size_usdt"])
    out["take_profit_pct"] = _auto_clip_number(src.get("take_profit_pct"), 0.0, 20.0, base["take_profit_pct"])
    out["stop_loss_pct"] = _auto_clip_number(src.get("stop_loss_pct"), 0.0, 20.0, base["stop_loss_pct"])
    out["daily_max_loss_usdt"] = _auto_clip_number(
        src.get("daily_max_loss_usdt"), 1.0, 300000.0, base["daily_max_loss_usdt"]
    )
    out["cooldown_min"] = int(_auto_clip_number(src.get("cooldown_min"), 0.0, 720.0, float(base["cooldown_min"])))
    out["max_open_positions"] = int(
        _auto_clip_number(src.get("max_open_positions"), 1.0, 5.0, float(base["max_open_positions"]))
    )
    out["last_run_ms"] = int(_auto_clip_number(src.get("last_run_ms"), 0.0, 10**15, float(base["last_run_ms"])))
    return out


def _auto_apply_mode_relax(params: Dict[str, float], mode: str) -> Dict[str, float]:
    out = dict(params)
    if mode != "aggressive":
        return out
    out["side_strong"] = max(4.0, out["side_strong"] - 2.0)
    out["side_weak"] = max(2.0, out["side_weak"] - 1.0)
    out["prob_min_pct"] = max(48.0, out["prob_min_pct"] - 4.0)
    out["prob_min_conf"] = max(0.22, out["prob_min_conf"] - 0.06)
    out["prob_soft_conf"] = max(0.28, out["prob_soft_conf"] - 0.06)
    out["pass_regime_conf"] = max(0.25, out["pass_regime_conf"] - 0.05)
    out["pass_regime_diff"] = max(5.0, out["pass_regime_diff"] - 2.0)
    return out


def _auto_decide_signal(
    *,
    buy_pct: float,
    sell_pct: float,
    confidence: float,
    regime: str,
    symbol: str,
    market: str,
    mode: str,
    swing_is_up: Optional[bool],
) -> Dict[str, Any]:
    params = _auto_apply_mode_relax(_decision_params_by_regime(regime, symbol), mode)
    raw_diff = float(buy_pct) - float(sell_pct)
    swing_bias = 0.0
    if swing_is_up is True:
        swing_bias = 6.0
    elif swing_is_up is False:
        swing_bias = -6.0
    diff = raw_diff + swing_bias

    side = "WAIT"
    if diff >= params["side_strong"] or (diff >= params["side_weak"] and confidence >= params["conf_weak"]):
        side = "BUY"
    elif diff <= -params["side_strong"] or (diff <= -params["side_weak"] and confidence >= params["conf_weak"]):
        side = "SELL"
    if confidence < params["conf_floor"]:
        side = "WAIT"
    if str(regime).upper() == "HIGH_VOL" and confidence < params["pass_regime_conf"] and abs(diff) < params["pass_regime_diff"]:
        side = "WAIT"
    if market == "spot" and side == "SELL":
        side = "WAIT"
    if swing_is_up is True and side == "SELL":
        side = "WAIT"
    if swing_is_up is False and side == "BUY":
        side = "WAIT"

    pass_prob = (
        side == "BUY"
        and buy_pct >= params["prob_min_pct"]
        and confidence >= params["prob_min_conf"]
        and (confidence >= params["prob_soft_conf"] or diff >= params["side_strong"])
    ) or (
        side == "SELL"
        and sell_pct >= params["prob_min_pct"]
        and confidence >= params["prob_min_conf"]
        and (confidence >= params["prob_soft_conf"] or diff <= -params["side_strong"])
    )
    pass_regime = (
        str(regime).upper() != "HIGH_VOL"
        or confidence >= params["pass_regime_conf"]
        or abs(diff) >= params["pass_regime_diff"]
    )
    return {
        "side": side,
        "pass_prob": bool(pass_prob),
        "pass_regime": bool(pass_regime),
        "raw_diff": float(raw_diff),
        "swing_bias": float(swing_bias),
        "diff": float(diff),
        "params": params,
    }


_AUTO_HOLD_MAX_MS = 2 * 24 * 60 * 60 * 1000


def _auto_side_matches_swing(side: str, swing_is_up: bool) -> bool:
    s = str(side or "").upper()
    return (s == "BUY" and bool(swing_is_up)) or (s == "SELL" and (not bool(swing_is_up)))


def _auto_pick_aggressive_side(*, buy_pct: float, sell_pct: float, market: str) -> str:
    b = float(buy_pct)
    s = float(sell_pct)
    if b >= 60.0 and b > s:
        return "BUY"
    if s >= 60.0 and s > b:
        return "SELL"
    return "WAIT"


def _auto_calc_pnl(*, side: str, entry_price: float, exit_price: float, qty: float) -> float:
    q = float(qty)
    e = float(entry_price)
    x = float(exit_price)
    if str(side or "BUY").upper() == "SELL":
        return (e - x) * q
    return (x - e) * q


def _auto_reason_pack(state: Dict[str, Any]) -> str:
    payload = dict(state or {})
    return "plan_json:" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _auto_reason_unpack(reason: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    raw = str(reason or "").strip()
    if raw.startswith("plan_json:"):
        try:
            obj = json.loads(raw[10:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    tp2 = float(record.get("take_profit_price", 0.0) or 0.0)
    qty = float(record.get("qty", 0.0) or 0.0)
    return {
        "v": 1,
        "side": str(record.get("side", "BUY")).upper(),
        "tp1_price": tp2,
        "tp2_price": tp2,
        "tp1_hit": False,
        "init_qty": qty,
        "remaining_qty": qty,
        "realized_pnl_usdt": float(record.get("pnl_usdt", 0.0) or 0.0),
        "hold_max_ms": _AUTO_HOLD_MAX_MS,
        "note": raw[:180],
    }


def _auto_build_fib_trade_plan(
    *,
    side: str,
    swing: Dict[str, float],
    close: float,
    atr14: float,
    mode: str,
) -> Optional[Dict[str, float]]:
    s = str(side or "").upper()
    if s not in {"BUY", "SELL"}:
        return None
    is_up = bool(int(swing.get("is_up", 0)))
    lo = float(swing.get("lo", 0.0) or 0.0)
    hi = float(swing.get("hi", 0.0) or 0.0)
    if hi <= lo or close <= 0:
        return None

    p0 = _fib_price(is_up, lo, hi, 0.0)
    p0236 = _fib_price(is_up, lo, hi, 0.236)
    p0382 = _fib_price(is_up, lo, hi, 0.382)
    p05 = _fib_price(is_up, lo, hi, 0.5)
    p0618 = _fib_price(is_up, lo, hi, 0.618)
    p0786 = _fib_price(is_up, lo, hi, 0.786)
    atr_pct_raw = (float(atr14) / float(close)) if close > 0 else 0.0
    atr_pct = min(0.05, max(0.003, atr_pct_raw))
    if str(mode or "").lower() == "aggressive":
        atr_pct *= 1.1
    stop_gap = min(0.03, max(0.006, atr_pct * 0.9))
    tp_gap = min(0.03, max(0.005, atr_pct * 0.8))

    entry = 0.0
    stop = 0.0
    tp1 = 0.0
    tp2 = 0.0
    if s == "BUY":
        if is_up:
            entry = (p05 + p0618) / 2.0
            stop = min(p0786, entry * (1.0 - stop_gap))
            tp1 = max(p0236, entry * (1.0 + tp_gap))
            tp2 = max(p0, tp1 * (1.0 + tp_gap * 0.65))
        else:
            # 하락 우세에서는 추격하지 않고 반등 구간(0.236 근처) 도달을 기다린다.
            entry = (p0 + p0236) / 2.0
            stop = min(lo * (1.0 - stop_gap * 0.8), entry * (1.0 - stop_gap))
            tp1 = max(p0382, entry * (1.0 + tp_gap))
            tp2 = max(p05, tp1 * (1.0 + tp_gap * 0.65))
        if not (stop < entry < tp1 < tp2):
            return None
    else:
        if not is_up:
            entry = (p05 + p0618) / 2.0
            stop = max(p0786, entry * (1.0 + stop_gap))
            tp1 = min(p0236, entry * (1.0 - tp_gap))
            tp2 = min(p0, tp1 * (1.0 - tp_gap * 0.65))
        else:
            entry = (p0 + p0236) / 2.0
            stop = max(hi * (1.0 + stop_gap * 0.8), entry * (1.0 + stop_gap))
            tp1 = min(p0382, entry * (1.0 - tp_gap))
            tp2 = min(p05, tp1 * (1.0 - tp_gap * 0.65))
        if not (tp2 < tp1 < entry < stop):
            return None

    return {
        "entry_price": float(entry),
        "stop_price": float(stop),
        "tp1_price": float(tp1),
        "tp2_price": float(tp2),
        "swing_is_up": 1.0 if is_up else 0.0,
        "swing_lo": float(lo),
        "swing_hi": float(hi),
    }


def _auto_entry_touched(*, entry_price: float, high: float, low: float, close: float) -> bool:
    entry = float(entry_price)
    hi = float(high)
    lo = float(low)
    cl = float(close)
    tol = max(entry * 0.0012, 1e-9)
    if lo - tol <= entry <= hi + tol:
        return True
    return abs(cl - entry) <= tol


def _kst_day_start_ms(now_ms: int) -> int:
    local_ms = int(now_ms) + _KST_OFFSET_MS
    local_day_start = int(local_ms // 86400000) * 86400000
    return local_day_start - _KST_OFFSET_MS


def _auto_eval_record_status(record: Dict[str, Any], klines: List[Any], now_ms: int) -> Dict[str, Any]:
    status = str(record.get("status", "OPEN")).upper()
    if status != "OPEN":
        return {"changed": False}
    entry = float(record.get("entry_price", 0.0) or 0.0)
    sl = float(record.get("stop_loss_price", 0.0) or 0.0)
    opened_ms = int(record.get("opened_ms", 0) or 0)
    if min(entry, sl) <= 0 or opened_ms <= 0:
        return {"changed": False}

    state = _auto_reason_unpack(record.get("reason"), record)
    side = str(state.get("side", record.get("side", "BUY"))).upper()
    if side not in {"BUY", "SELL"}:
        side = "BUY"
    qty_initial = float(state.get("init_qty", record.get("qty", 0.0)) or 0.0)
    rem_qty = float(state.get("remaining_qty", record.get("qty", 0.0)) or 0.0)
    realized = float(state.get("realized_pnl_usdt", record.get("pnl_usdt", 0.0)) or 0.0)
    tp1_hit = bool(state.get("tp1_hit", False))
    tp1 = float(state.get("tp1_price", record.get("take_profit_price", 0.0)) or 0.0)
    tp2 = float(state.get("tp2_price", record.get("take_profit_price", 0.0)) or 0.0)
    if qty_initial <= 0 or rem_qty <= 0 or tp1 <= 0 or tp2 <= 0:
        return {"changed": False}

    hold_max_ms = int(state.get("hold_max_ms", _AUTO_HOLD_MAX_MS) or _AUTO_HOLD_MAX_MS)
    expire_ms = opened_ms + max(3600000, hold_max_ms)
    last_close: Optional[float] = None
    changed_open = False

    for k in klines:
        k_open = int(getattr(k, "open_time_ms", 0))
        k_close = int(getattr(k, "close_time_ms", 0))
        if k_open < opened_ms and k_close > 0 and k_close < opened_ms:
            continue
        hi = float(getattr(k, "high", 0.0))
        lo = float(getattr(k, "low", 0.0))
        last_close = float(getattr(k, "close", 0.0))

        if side == "BUY":
            if not tp1_hit:
                # 동시 터치 시 보수적으로 손절 우선
                if lo <= sl:
                    final_pnl = realized + _auto_calc_pnl(side=side, entry_price=entry, exit_price=sl, qty=rem_qty)
                    return {
                        "changed": True,
                        "patch": {
                            "status": "SL",
                            "close_price": round(sl, 8),
                            "pnl_usdt": round(final_pnl, 8),
                            "closed_ms": now_ms,
                            "qty": round(rem_qty, 8),
                            "reason": _auto_reason_pack(
                                {
                                    **state,
                                    "side": side,
                                    "tp1_hit": False,
                                    "remaining_qty": rem_qty,
                                    "realized_pnl_usdt": round(realized, 8),
                                }
                            ),
                        },
                    }
                if hi >= tp1:
                    take_qty = min(rem_qty, max(0.0, qty_initial * 0.5))
                    realized += _auto_calc_pnl(side=side, entry_price=entry, exit_price=tp1, qty=take_qty)
                    rem_qty = max(0.0, rem_qty - take_qty)
                    tp1_hit = True
                    changed_open = True
                    if hi >= tp2 and rem_qty > 0:
                        final_pnl = realized + _auto_calc_pnl(side=side, entry_price=entry, exit_price=tp2, qty=rem_qty)
                        return {
                            "changed": True,
                            "patch": {
                                "status": "TP",
                                "close_price": round(tp2, 8),
                                "pnl_usdt": round(final_pnl, 8),
                                "closed_ms": now_ms,
                                "qty": round(rem_qty, 8),
                                "reason": _auto_reason_pack(
                                    {
                                        **state,
                                        "side": side,
                                        "tp1_hit": True,
                                        "remaining_qty": 0.0,
                                        "realized_pnl_usdt": round(realized, 8),
                                    }
                                ),
                            },
                        }
            else:
                if lo <= sl:
                    final_pnl = realized + _auto_calc_pnl(side=side, entry_price=entry, exit_price=sl, qty=rem_qty)
                    return {
                        "changed": True,
                        "patch": {
                            "status": "SL",
                            "close_price": round(sl, 8),
                            "pnl_usdt": round(final_pnl, 8),
                            "closed_ms": now_ms,
                            "qty": round(rem_qty, 8),
                            "reason": _auto_reason_pack(
                                {
                                    **state,
                                    "side": side,
                                    "tp1_hit": True,
                                    "remaining_qty": rem_qty,
                                    "realized_pnl_usdt": round(realized, 8),
                                }
                            ),
                        },
                    }
                if hi >= tp2:
                    final_pnl = realized + _auto_calc_pnl(side=side, entry_price=entry, exit_price=tp2, qty=rem_qty)
                    return {
                        "changed": True,
                        "patch": {
                            "status": "TP",
                            "close_price": round(tp2, 8),
                            "pnl_usdt": round(final_pnl, 8),
                            "closed_ms": now_ms,
                            "qty": round(rem_qty, 8),
                            "reason": _auto_reason_pack(
                                {
                                    **state,
                                    "side": side,
                                    "tp1_hit": True,
                                    "remaining_qty": 0.0,
                                    "realized_pnl_usdt": round(realized, 8),
                                }
                            ),
                        },
                    }
        else:
            if not tp1_hit:
                if hi >= sl:
                    final_pnl = realized + _auto_calc_pnl(side=side, entry_price=entry, exit_price=sl, qty=rem_qty)
                    return {
                        "changed": True,
                        "patch": {
                            "status": "SL",
                            "close_price": round(sl, 8),
                            "pnl_usdt": round(final_pnl, 8),
                            "closed_ms": now_ms,
                            "qty": round(rem_qty, 8),
                            "reason": _auto_reason_pack(
                                {
                                    **state,
                                    "side": side,
                                    "tp1_hit": False,
                                    "remaining_qty": rem_qty,
                                    "realized_pnl_usdt": round(realized, 8),
                                }
                            ),
                        },
                    }
                if lo <= tp1:
                    take_qty = min(rem_qty, max(0.0, qty_initial * 0.5))
                    realized += _auto_calc_pnl(side=side, entry_price=entry, exit_price=tp1, qty=take_qty)
                    rem_qty = max(0.0, rem_qty - take_qty)
                    tp1_hit = True
                    changed_open = True
                    if lo <= tp2 and rem_qty > 0:
                        final_pnl = realized + _auto_calc_pnl(side=side, entry_price=entry, exit_price=tp2, qty=rem_qty)
                        return {
                            "changed": True,
                            "patch": {
                                "status": "TP",
                                "close_price": round(tp2, 8),
                                "pnl_usdt": round(final_pnl, 8),
                                "closed_ms": now_ms,
                                "qty": round(rem_qty, 8),
                                "reason": _auto_reason_pack(
                                    {
                                        **state,
                                        "side": side,
                                        "tp1_hit": True,
                                        "remaining_qty": 0.0,
                                        "realized_pnl_usdt": round(realized, 8),
                                    }
                                ),
                            },
                        }
            else:
                if hi >= sl:
                    final_pnl = realized + _auto_calc_pnl(side=side, entry_price=entry, exit_price=sl, qty=rem_qty)
                    return {
                        "changed": True,
                        "patch": {
                            "status": "SL",
                            "close_price": round(sl, 8),
                            "pnl_usdt": round(final_pnl, 8),
                            "closed_ms": now_ms,
                            "qty": round(rem_qty, 8),
                            "reason": _auto_reason_pack(
                                {
                                    **state,
                                    "side": side,
                                    "tp1_hit": True,
                                    "remaining_qty": rem_qty,
                                    "realized_pnl_usdt": round(realized, 8),
                                }
                            ),
                        },
                    }
                if lo <= tp2:
                    final_pnl = realized + _auto_calc_pnl(side=side, entry_price=entry, exit_price=tp2, qty=rem_qty)
                    return {
                        "changed": True,
                        "patch": {
                            "status": "TP",
                            "close_price": round(tp2, 8),
                            "pnl_usdt": round(final_pnl, 8),
                            "closed_ms": now_ms,
                            "qty": round(rem_qty, 8),
                            "reason": _auto_reason_pack(
                                {
                                    **state,
                                    "side": side,
                                    "tp1_hit": True,
                                    "remaining_qty": 0.0,
                                    "realized_pnl_usdt": round(realized, 8),
                                }
                            ),
                        },
                    }

    if now_ms >= expire_ms and last_close is not None:
        final_pnl = realized + _auto_calc_pnl(side=side, entry_price=entry, exit_price=float(last_close), qty=rem_qty)
        return {
            "changed": True,
            "patch": {
                "status": "CLOSED_FAIL",
                "close_price": round(float(last_close), 8),
                "pnl_usdt": round(final_pnl, 8),
                "closed_ms": now_ms,
                "qty": round(rem_qty, 8),
                "reason": _auto_reason_pack(
                    {
                        **state,
                        "side": side,
                        "tp1_hit": bool(tp1_hit),
                        "remaining_qty": rem_qty,
                        "realized_pnl_usdt": round(realized, 8),
                    }
                ),
            },
        }

    if changed_open:
        return {
            "changed": True,
            "patch": {
                "status": "OPEN",
                "pnl_usdt": round(realized, 8),
                "qty": round(rem_qty, 8),
                "reason": _auto_reason_pack(
                    {
                        **state,
                        "side": side,
                        "tp1_hit": bool(tp1_hit),
                        "remaining_qty": rem_qty,
                        "realized_pnl_usdt": round(realized, 8),
                    }
                ),
            },
        }
    return {"changed": False}


async def _auto_fetch_kline_cache(records: List[Dict[str, Any]], now_ts: float) -> Dict[str, List[Any]]:
    keys: Dict[str, tuple[str, str, str]] = {}
    for r in records:
        sym = str(r.get("symbol", "")).upper()
        market = str(r.get("market", "spot")).lower()
        interval = str(r.get("interval", "5m"))
        if not sym:
            continue
        k = f"{sym}:{market}:{interval}"
        keys[k] = (sym, market, interval)
    if not keys:
        return {}

    async def _fetch_one(key: str, sym: str, market: str, interval: str) -> tuple[str, List[Any]]:
        rows: List[Any] = []
        try:
            rows = await app.state.binance.klines(symbol=sym, interval=interval, limit=1000, now_ts=now_ts, market=market)
        except Exception:
            rows = []
        return key, list(rows)

    tasks = [_fetch_one(k, v[0], v[1], v[2]) for k, v in keys.items()]
    pairs = await asyncio.gather(*tasks, return_exceptions=True)
    out: Dict[str, List[Any]] = {}
    for p in pairs:
        if isinstance(p, Exception):
            continue
        out[p[0]] = p[1]
    for k in keys.keys():
        out.setdefault(k, [])
    return out


async def _auto_get_or_create_config(user_id: int) -> Dict[str, Any]:
    try:
        rows = await _sb_request(
            "GET",
            "/rest/v1/auto_trade_settings",
            params={
                "select": "id,user_id,enabled,mode,symbol,market,interval,order_size_usdt,take_profit_pct,stop_loss_pct,daily_max_loss_usdt,cooldown_min,max_open_positions,last_run_ms,created_ms,updated_ms",
                "user_id": f"eq.{int(user_id)}",
                "limit": "1",
            },
        )
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_settings"):
            raise _auto_table_missing_error("auto_trade_settings")
        raise
    if isinstance(rows, list) and rows:
        row = dict(rows[0])
        norm = _auto_normalize_config(row)
        norm["id"] = row.get("id")
        norm["user_id"] = int(user_id)
        return norm

    now_ms = int(time() * 1000)
    body = _auto_default_config()
    body["user_id"] = int(user_id)
    body["created_ms"] = now_ms
    body["updated_ms"] = now_ms
    try:
        created = await _sb_request("POST", "/rest/v1/auto_trade_settings", json_body=[body])
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_settings"):
            raise _auto_table_missing_error("auto_trade_settings")
        raise
    if isinstance(created, list) and created:
        row = dict(created[0])
        norm = _auto_normalize_config(row)
        norm["id"] = row.get("id")
        norm["user_id"] = int(user_id)
        return norm
    raise HTTPException(status_code=500, detail="failed to initialize auto trade settings")


async def _auto_update_config(user_id: int, config_id: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    now_ms = int(time() * 1000)
    body = _auto_normalize_config(payload)
    body["updated_ms"] = now_ms
    body["last_run_ms"] = int(_auto_clip_number(payload.get("last_run_ms"), 0, 10**15, body["last_run_ms"]))
    try:
        rows = await _sb_request(
            "PATCH",
            "/rest/v1/auto_trade_settings",
            params={"id": f"eq.{config_id}", "user_id": f"eq.{int(user_id)}"},
            json_body=body,
        )
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_settings"):
            raise _auto_table_missing_error("auto_trade_settings")
        raise
    if isinstance(rows, list) and rows:
        row = dict(rows[0])
        norm = _auto_normalize_config(row)
        norm["id"] = row.get("id")
        norm["user_id"] = int(user_id)
        norm["created_ms"] = int(row.get("created_ms", 0) or 0)
        norm["updated_ms"] = int(row.get("updated_ms", 0) or 0)
        return norm
    fresh = await _auto_get_or_create_config(user_id)
    return fresh


async def _auto_apply_live_partial_close(
    *,
    user_id: int,
    record: Dict[str, Any],
    old_state: Dict[str, Any],
    new_state: Dict[str, Any],
) -> Dict[str, Any]:
    if not _AUTO_LIVE_TRADING_ENABLED:
        return {}
    side = str(old_state.get("side", record.get("side", "BUY"))).upper()
    market = str(record.get("market", "spot")).lower()
    symbol = str(record.get("symbol", "")).upper()
    entry = float(record.get("entry_price", 0.0) or 0.0)
    old_rem = float(old_state.get("remaining_qty", record.get("qty", 0.0)) or 0.0)
    new_rem = float(new_state.get("remaining_qty", old_rem) or old_rem)
    close_qty = max(0.0, old_rem - new_rem)
    if close_qty <= 0:
        return {}
    close_side = "SELL" if side == "BUY" else "BUY"
    if market == "spot" and close_side == "SELL":
        close_qty *= 0.999
    rec_id = _auto_int_or_none(record.get("id"))
    tp1_client_id = _auto_client_id("tp1", user_id, record.get("id"), symbol, market)
    api_key, api_secret = await _auto_get_binance_credentials(user_id)
    # 부분익절 전에 기존 손절 보호주문을 취소하고, 체결 후 잔량 기준으로 다시 생성한다.
    prev_sl_oid = str(old_state.get("protect_sl_order_id", "") or "")
    prev_sl_cid = str(old_state.get("protect_sl_client_id", "") or "")
    if prev_sl_oid or prev_sl_cid:
        try:
            await _auto_cancel_order(
                market=market,
                symbol=symbol,
                api_key=api_key,
                api_secret=api_secret,
                order_id=prev_sl_oid,
                client_order_id=prev_sl_cid,
            )
        except HTTPException as e:
            await _auto_audit_log(
                user_id=user_id,
                event="TP1_PREV_SL_CANCEL_FAIL",
                level="WARN",
                record_id=rec_id,
                symbol=symbol,
                market=market,
                side=close_side,
                detail=str(getattr(e, "detail", "") or "cancel failed"),
                order_id=prev_sl_oid,
                client_order_id=prev_sl_cid,
            )
            raise
    try:
        fill = await _auto_place_market_order(
            market=market,
            symbol=symbol,
            side=close_side,
            qty=close_qty,
            api_key=api_key,
            api_secret=api_secret,
            reduce_only=(market == "futures"),
            fallback_price=float(record.get("entry_price", 0.0) or 0.0),
            client_order_id=tp1_client_id,
        )
    except HTTPException as e:
        await _auto_audit_log(
            user_id=user_id,
            event="TP1_CLOSE_REJECTED",
            level="ERROR",
            record_id=rec_id,
            symbol=symbol,
            market=market,
            side=close_side,
            qty=close_qty,
            detail=str(getattr(e, "detail", "") or "partial close order rejected"),
            client_order_id=tp1_client_id,
        )
        raise
    exec_qty = float(fill.get("executed_qty", 0.0) or 0.0)
    avg_price = float(fill.get("avg_price", 0.0) or 0.0)
    if exec_qty <= 0 or avg_price <= 0:
        await _auto_audit_log(
            user_id=user_id,
            event="TP1_CLOSE_EMPTY_FILL",
            level="ERROR",
            record_id=rec_id,
            symbol=symbol,
            market=market,
            side=close_side,
            qty=close_qty,
            detail="partial close order not filled",
            order_id=str(fill.get("order_id", "") or ""),
            client_order_id=tp1_client_id,
        )
        raise HTTPException(status_code=400, detail="partial close order not filled")
    await _auto_audit_log(
        user_id=user_id,
        event="TP1_CLOSE_FILLED",
        record_id=rec_id,
        symbol=symbol,
        market=market,
        side=close_side,
        status="OPEN",
        qty=exec_qty,
        price=avg_price,
        order_id=str(fill.get("order_id", "") or ""),
        client_order_id=tp1_client_id,
    )
    realized_before = float(old_state.get("realized_pnl_usdt", record.get("pnl_usdt", 0.0)) or 0.0)
    realized_after = realized_before + _auto_calc_pnl(side=side, entry_price=entry, exit_price=avg_price, qty=exec_qty)
    rem_after = max(0.0, old_rem - exec_qty)
    sl_price = float(record.get("stop_loss_price", 0.0) or 0.0)
    protect_info: Dict[str, Any] = {}
    protect_err = ""
    if rem_after > 0 and sl_price > 0:
        protect_cid = _auto_client_id("psl", user_id, record.get("id"), symbol, market, "tp1")
        try:
            protect_info = await _auto_place_protective_stop_order(
                market=market,
                symbol=symbol,
                side=close_side,
                stop_price=sl_price,
                qty=rem_after,
                api_key=api_key,
                api_secret=api_secret,
                client_order_id=protect_cid,
            )
            await _auto_audit_log(
                user_id=user_id,
                event="TP1_SL_RECREATE_OK",
                record_id=rec_id,
                symbol=symbol,
                market=market,
                side=close_side,
                status="OPEN",
                qty=rem_after,
                price=sl_price,
                order_id=str(protect_info.get("order_id", "") or ""),
                client_order_id=str(protect_info.get("client_order_id", "") or ""),
            )
        except HTTPException as pe:
            protect_err = str(getattr(pe, "detail", "") or "protective stop update failed")[:180]
            await _auto_audit_log(
                user_id=user_id,
                event="TP1_SL_RECREATE_FAIL",
                level="WARN",
                record_id=rec_id,
                symbol=symbol,
                market=market,
                side=close_side,
                status="OPEN",
                qty=rem_after,
                price=sl_price,
                detail=protect_err,
                client_order_id=protect_cid,
            )
    merged_state = {
        **old_state,
        **new_state,
        "side": side,
        "tp1_hit": True,
        "remaining_qty": round(rem_after, 8),
        "realized_pnl_usdt": round(realized_after, 8),
        "tp1_exec": {
            "order_id": str(fill.get("order_id", "") or ""),
            "qty": round(exec_qty, 8),
            "avg_price": round(avg_price, 8),
            "ts_ms": int(time() * 1000),
            "client_order_id": tp1_client_id,
        },
        "protect_sl_order_id": str(protect_info.get("order_id", "") or ""),
        "protect_sl_client_id": str(protect_info.get("client_order_id", "") or ""),
        "protect_sl_type": str(protect_info.get("type", "") or ""),
        "last_exec_error": protect_err,
    }
    return {
        "status": "OPEN",
        "pnl_usdt": round(realized_after, 8),
        "qty": round(rem_after, 8),
        "reason": _auto_reason_pack(merged_state),
    }


async def _auto_apply_live_final_close(
    *,
    user_id: int,
    record: Dict[str, Any],
    old_state: Dict[str, Any],
    close_status: str,
) -> Dict[str, Any]:
    if not _AUTO_LIVE_TRADING_ENABLED:
        return {}
    side = str(old_state.get("side", record.get("side", "BUY"))).upper()
    market = str(record.get("market", "spot")).lower()
    symbol = str(record.get("symbol", "")).upper()
    entry = float(record.get("entry_price", 0.0) or 0.0)
    rem_qty = float(old_state.get("remaining_qty", record.get("qty", 0.0)) or 0.0)
    if rem_qty <= 0:
        return {}
    close_side = "SELL" if side == "BUY" else "BUY"
    if market == "spot" and close_side == "SELL":
        rem_qty *= 0.999
    rec_id = _auto_int_or_none(record.get("id"))
    api_key, api_secret = await _auto_get_binance_credentials(user_id)
    close_client_id = _auto_client_id("cl", user_id, record.get("id"), symbol, market, close_status)
    try:
        fill = await _auto_place_market_order(
            market=market,
            symbol=symbol,
            side=close_side,
            qty=rem_qty,
            api_key=api_key,
            api_secret=api_secret,
            reduce_only=(market == "futures"),
            fallback_price=float(record.get("entry_price", 0.0) or 0.0),
            client_order_id=close_client_id,
        )
    except HTTPException as e:
        await _auto_audit_log(
            user_id=user_id,
            event="FINAL_CLOSE_REJECTED",
            level="ERROR",
            record_id=rec_id,
            symbol=symbol,
            market=market,
            side=close_side,
            status=close_status,
            qty=rem_qty,
            detail=str(getattr(e, "detail", "") or "final close order rejected"),
            client_order_id=close_client_id,
        )
        raise
    exec_qty = float(fill.get("executed_qty", 0.0) or 0.0)
    avg_price = float(fill.get("avg_price", 0.0) or 0.0)
    if exec_qty <= 0 or avg_price <= 0:
        await _auto_audit_log(
            user_id=user_id,
            event="FINAL_CLOSE_EMPTY_FILL",
            level="ERROR",
            record_id=rec_id,
            symbol=symbol,
            market=market,
            side=close_side,
            status=close_status,
            qty=rem_qty,
            detail="close order not filled",
            order_id=str(fill.get("order_id", "") or ""),
            client_order_id=close_client_id,
        )
        raise HTTPException(status_code=400, detail="close order not filled")
    await _auto_audit_log(
        user_id=user_id,
        event="FINAL_CLOSE_FILLED",
        record_id=rec_id,
        symbol=symbol,
        market=market,
        side=close_side,
        status=close_status,
        qty=exec_qty,
        price=avg_price,
        order_id=str(fill.get("order_id", "") or ""),
        client_order_id=close_client_id,
    )
    prev_sl_oid = str(old_state.get("protect_sl_order_id", "") or "")
    prev_sl_cid = str(old_state.get("protect_sl_client_id", "") or "")
    if prev_sl_oid or prev_sl_cid:
        try:
            await _auto_cancel_order(
                market=market,
                symbol=symbol,
                api_key=api_key,
                api_secret=api_secret,
                order_id=prev_sl_oid,
                client_order_id=prev_sl_cid,
            )
        except HTTPException as e:
            await _auto_audit_log(
                user_id=user_id,
                event="FINAL_CLOSE_SL_CANCEL_FAIL",
                level="WARN",
                record_id=rec_id,
                symbol=symbol,
                market=market,
                side=close_side,
                status=close_status,
                detail=str(getattr(e, "detail", "") or "protective cancel failed"),
                order_id=prev_sl_oid,
                client_order_id=prev_sl_cid,
            )
    realized_before = float(old_state.get("realized_pnl_usdt", record.get("pnl_usdt", 0.0)) or 0.0)
    final_pnl = realized_before + _auto_calc_pnl(side=side, entry_price=entry, exit_price=avg_price, qty=exec_qty)
    closed_state = {
        **old_state,
        "side": side,
        "remaining_qty": 0.0,
        "realized_pnl_usdt": round(final_pnl, 8),
        "close_exec": {
            "order_id": str(fill.get("order_id", "") or ""),
            "qty": round(exec_qty, 8),
            "avg_price": round(avg_price, 8),
            "status": str(close_status),
            "ts_ms": int(time() * 1000),
            "client_order_id": close_client_id,
        },
        "protect_sl_order_id": "",
        "protect_sl_client_id": "",
        "protect_sl_type": "",
        "last_exec_error": "",
    }
    return {
        "status": str(close_status),
        "close_price": round(avg_price, 8),
        "pnl_usdt": round(final_pnl, 8),
        "closed_ms": int(time() * 1000),
        "qty": round(exec_qty, 8),
        "reason": _auto_reason_pack(closed_state),
    }


async def _auto_reconcile_exchange_state_for_open_record(
    *,
    user_id: int,
    record: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not _AUTO_LIVE_TRADING_ENABLED:
        return None
    if int(state.get("v", 0) or 0) < 3:
        return None
    rem_qty = float(state.get("remaining_qty", record.get("qty", 0.0)) or 0.0)
    if rem_qty <= 0:
        return None
    rec_id = _auto_int_or_none(record.get("id"))
    symbol = str(record.get("symbol", "")).upper()
    market = str(record.get("market", "spot")).lower()
    side = str(state.get("side", record.get("side", "BUY"))).upper()
    if side not in {"BUY", "SELL"}:
        side = "BUY"
    close_side = "SELL" if side == "BUY" else "BUY"
    stop_price = float(record.get("stop_loss_price", 0.0) or 0.0)
    entry = float(record.get("entry_price", 0.0) or 0.0)
    realized_before = float(state.get("realized_pnl_usdt", record.get("pnl_usdt", 0.0)) or 0.0)
    now_ms = int(time() * 1000)
    api_key, api_secret = await _auto_get_binance_credentials(user_id)

    protect_oid = str(state.get("protect_sl_order_id", "") or "")
    protect_cid = str(state.get("protect_sl_client_id", "") or "")
    if protect_cid:
        try:
            ord_body = await _auto_fetch_order_by_client_id(
                market=market,
                symbol=symbol,
                client_order_id=protect_cid,
                api_key=api_key,
                api_secret=api_secret,
            )
            st = str(ord_body.get("status", "")).upper()
            if st == "FILLED":
                fill = _auto_extract_order_fill(ord_body, fallback_price=stop_price)
                exec_qty = float(fill.get("executed_qty", 0.0) or 0.0)
                if exec_qty <= 0:
                    exec_qty = rem_qty
                close_px = float(fill.get("avg_price", 0.0) or 0.0)
                if close_px <= 0:
                    close_px = stop_price
                realized_before = float(state.get("realized_pnl_usdt", record.get("pnl_usdt", 0.0)) or 0.0)
                final_pnl = realized_before + _auto_calc_pnl(
                    side=side,
                    entry_price=entry,
                    exit_price=close_px,
                    qty=exec_qty,
                )
                close_state = {
                    **state,
                    "remaining_qty": 0.0,
                    "realized_pnl_usdt": round(final_pnl, 8),
                    "protect_sl_order_id": "",
                    "protect_sl_client_id": "",
                    "protect_sl_type": "",
                    "close_exec": {
                        "order_id": str(ord_body.get("orderId", protect_oid) or ""),
                        "client_order_id": protect_cid,
                        "qty": round(exec_qty, 8),
                        "avg_price": round(close_px, 8),
                        "status": "SL_PROTECT_FILLED",
                        "ts_ms": now_ms,
                    },
                    "last_exec_error": "",
                }
                await _auto_audit_log(
                    user_id=user_id,
                    event="RECON_PROTECT_FILLED",
                    level="WARN",
                    record_id=rec_id,
                    symbol=symbol,
                    market=market,
                    side=close_side,
                    status="SL",
                    qty=exec_qty,
                    price=close_px,
                    order_id=str(ord_body.get("orderId", protect_oid) or ""),
                    client_order_id=protect_cid,
                )
                return {
                    "status": "SL",
                    "close_price": round(close_px, 8),
                    "pnl_usdt": round(final_pnl, 8),
                    "closed_ms": now_ms,
                    "qty": round(exec_qty, 8),
                    "reason": _auto_reason_pack(close_state),
                }
            if st in {"CANCELED", "EXPIRED", "REJECTED"} and stop_price > 0:
                # spot은 계정 전체 보유수량으로 전략 포지션 잔량을 단정할 수 없어 futures에서만 no-position 판정한다.
                live_qty = rem_qty
                if market == "futures":
                    live_qty = await _auto_fetch_live_position_qty(
                        market=market,
                        symbol=symbol,
                        api_key=api_key,
                        api_secret=api_secret,
                    )
                if market == "futures" and live_qty <= 0:
                    mark_px = await _auto_fetch_last_price(market=market, symbol=symbol)
                    if mark_px <= 0:
                        mark_px = entry
                    final_pnl = realized_before + _auto_calc_pnl(
                        side=side,
                        entry_price=entry,
                        exit_price=mark_px,
                        qty=rem_qty,
                    )
                    close_state = {
                        **state,
                        "remaining_qty": 0.0,
                        "realized_pnl_usdt": round(final_pnl, 8),
                        "protect_sl_order_id": "",
                        "protect_sl_client_id": "",
                        "protect_sl_type": "",
                        "close_exec": {
                            "order_id": "",
                            "client_order_id": protect_cid,
                            "qty": round(rem_qty, 8),
                            "avg_price": round(mark_px, 8),
                            "status": "RECON_NO_POSITION",
                            "ts_ms": now_ms,
                        },
                        "last_exec_error": "",
                    }
                    await _auto_audit_log(
                        user_id=user_id,
                        event="RECON_NO_POSITION_CLOSE",
                        level="WARN",
                        record_id=rec_id,
                        symbol=symbol,
                        market=market,
                        side=close_side,
                        status="CLOSED_FAIL",
                        qty=rem_qty,
                        price=mark_px,
                        client_order_id=protect_cid,
                    )
                    return {
                        "status": "CLOSED_FAIL",
                        "close_price": round(mark_px, 8),
                        "pnl_usdt": round(final_pnl, 8),
                        "closed_ms": now_ms,
                        "qty": round(rem_qty, 8),
                        "reason": _auto_reason_pack(close_state),
                    }
                new_cid = _auto_client_id("psl", user_id, record.get("id"), symbol, market, now_ms)
                try:
                    p = await _auto_place_protective_stop_order(
                        market=market,
                        symbol=symbol,
                        side=close_side,
                        stop_price=stop_price,
                        qty=rem_qty,
                        api_key=api_key,
                        api_secret=api_secret,
                        client_order_id=new_cid,
                    )
                    new_state = {
                        **state,
                        "protect_sl_order_id": str(p.get("order_id", "") or ""),
                        "protect_sl_client_id": str(p.get("client_order_id", "") or ""),
                        "protect_sl_type": str(p.get("type", "") or ""),
                        "last_exec_error": "",
                    }
                    await _auto_audit_log(
                        user_id=user_id,
                        event="RECON_PROTECT_RECREATE_OK",
                        level="WARN",
                        record_id=rec_id,
                        symbol=symbol,
                        market=market,
                        side=close_side,
                        status="OPEN",
                        qty=rem_qty,
                        price=stop_price,
                        order_id=str(p.get("order_id", "") or ""),
                        client_order_id=str(p.get("client_order_id", "") or ""),
                    )
                except HTTPException as e2:
                    new_state = {
                        **state,
                        "last_exec_error": f"protective stop recreate failed: {str(getattr(e2, 'detail', '') or '')[:120]}",
                        "last_exec_error_ms": now_ms,
                    }
                    await _auto_audit_log(
                        user_id=user_id,
                        event="RECON_PROTECT_RECREATE_FAIL",
                        level="ERROR",
                        record_id=rec_id,
                        symbol=symbol,
                        market=market,
                        side=close_side,
                        status="OPEN",
                        qty=rem_qty,
                        price=stop_price,
                        detail=str(getattr(e2, "detail", "") or "protective stop recreate failed"),
                        client_order_id=new_cid,
                    )
                return {
                    "status": "OPEN",
                    "reason": _auto_reason_pack(new_state),
                }
        except HTTPException as e:
            detail = str(getattr(e, "detail", "") or "")
            if _auto_is_unknown_order_error(detail) and stop_price > 0:
                # spot은 계정 전체 보유수량으로 전략 포지션 잔량을 단정할 수 없어 futures에서만 no-position 판정한다.
                live_qty = rem_qty
                if market == "futures":
                    live_qty = await _auto_fetch_live_position_qty(
                        market=market,
                        symbol=symbol,
                        api_key=api_key,
                        api_secret=api_secret,
                    )
                if market == "futures" and live_qty <= 0:
                    mark_px = await _auto_fetch_last_price(market=market, symbol=symbol)
                    if mark_px <= 0:
                        mark_px = entry
                    final_pnl = realized_before + _auto_calc_pnl(
                        side=side,
                        entry_price=entry,
                        exit_price=mark_px,
                        qty=rem_qty,
                    )
                    close_state = {
                        **state,
                        "remaining_qty": 0.0,
                        "realized_pnl_usdt": round(final_pnl, 8),
                        "protect_sl_order_id": "",
                        "protect_sl_client_id": "",
                        "protect_sl_type": "",
                        "close_exec": {
                            "order_id": "",
                            "client_order_id": protect_cid,
                            "qty": round(rem_qty, 8),
                            "avg_price": round(mark_px, 8),
                            "status": "RECON_NO_POSITION",
                            "ts_ms": now_ms,
                        },
                        "last_exec_error": "",
                    }
                    await _auto_audit_log(
                        user_id=user_id,
                        event="RECON_NO_POSITION_CLOSE",
                        level="WARN",
                        record_id=rec_id,
                        symbol=symbol,
                        market=market,
                        side=close_side,
                        status="CLOSED_FAIL",
                        qty=rem_qty,
                        price=mark_px,
                        client_order_id=protect_cid,
                    )
                    return {
                        "status": "CLOSED_FAIL",
                        "close_price": round(mark_px, 8),
                        "pnl_usdt": round(final_pnl, 8),
                        "closed_ms": now_ms,
                        "qty": round(rem_qty, 8),
                        "reason": _auto_reason_pack(close_state),
                    }
                new_cid = _auto_client_id("psl", user_id, record.get("id"), symbol, market, now_ms)
                try:
                    p = await _auto_place_protective_stop_order(
                        market=market,
                        symbol=symbol,
                        side=close_side,
                        stop_price=stop_price,
                        qty=rem_qty,
                        api_key=api_key,
                        api_secret=api_secret,
                        client_order_id=new_cid,
                    )
                    new_state = {
                        **state,
                        "protect_sl_order_id": str(p.get("order_id", "") or ""),
                        "protect_sl_client_id": str(p.get("client_order_id", "") or ""),
                        "protect_sl_type": str(p.get("type", "") or ""),
                        "last_exec_error": "",
                    }
                    await _auto_audit_log(
                        user_id=user_id,
                        event="RECON_PROTECT_RECREATE_OK",
                        level="WARN",
                        record_id=rec_id,
                        symbol=symbol,
                        market=market,
                        side=close_side,
                        status="OPEN",
                        qty=rem_qty,
                        price=stop_price,
                        order_id=str(p.get("order_id", "") or ""),
                        client_order_id=str(p.get("client_order_id", "") or ""),
                    )
                except HTTPException as e2:
                    new_state = {
                        **state,
                        "last_exec_error": f"protective stop recreate failed: {str(getattr(e2, 'detail', '') or '')[:120]}",
                        "last_exec_error_ms": now_ms,
                    }
                    await _auto_audit_log(
                        user_id=user_id,
                        event="RECON_PROTECT_RECREATE_FAIL",
                        level="ERROR",
                        record_id=rec_id,
                        symbol=symbol,
                        market=market,
                        side=close_side,
                        status="OPEN",
                        qty=rem_qty,
                        price=stop_price,
                        detail=str(getattr(e2, "detail", "") or "protective stop recreate failed"),
                        client_order_id=new_cid,
                    )
                return {
                    "status": "OPEN",
                    "reason": _auto_reason_pack(new_state),
                }
            await _auto_audit_log(
                user_id=user_id,
                event="RECON_PROTECT_QUERY_FAIL",
                level="ERROR",
                record_id=rec_id,
                symbol=symbol,
                market=market,
                side=close_side,
                status="OPEN",
                qty=rem_qty,
                price=stop_price,
                detail=detail,
                client_order_id=protect_cid,
            )
            return {
                "status": "OPEN",
                "reason": _auto_reason_pack(
                    {
                        **state,
                        "last_exec_error": f"protective order query failed: {detail[:120]}",
                        "last_exec_error_ms": now_ms,
                    }
                ),
            }
        return None

    if stop_price > 0:
        new_cid = _auto_client_id("psl", user_id, record.get("id"), symbol, market, now_ms)
        try:
            p = await _auto_place_protective_stop_order(
                market=market,
                symbol=symbol,
                side=close_side,
                stop_price=stop_price,
                qty=rem_qty,
                api_key=api_key,
                api_secret=api_secret,
                client_order_id=new_cid,
            )
            new_state = {
                **state,
                "protect_sl_order_id": str(p.get("order_id", "") or ""),
                "protect_sl_client_id": str(p.get("client_order_id", "") or ""),
                "protect_sl_type": str(p.get("type", "") or ""),
                "last_exec_error": "",
            }
            await _auto_audit_log(
                user_id=user_id,
                event="RECON_PROTECT_CREATE_OK",
                level="WARN",
                record_id=rec_id,
                symbol=symbol,
                market=market,
                side=close_side,
                status="OPEN",
                qty=rem_qty,
                price=stop_price,
                order_id=str(p.get("order_id", "") or ""),
                client_order_id=str(p.get("client_order_id", "") or ""),
            )
        except HTTPException as e:
            new_state = {
                **state,
                "last_exec_error": f"protective stop create failed: {str(getattr(e, 'detail', '') or '')[:120]}",
                "last_exec_error_ms": now_ms,
            }
            await _auto_audit_log(
                user_id=user_id,
                event="RECON_PROTECT_CREATE_FAIL",
                level="ERROR",
                record_id=rec_id,
                symbol=symbol,
                market=market,
                side=close_side,
                status="OPEN",
                qty=rem_qty,
                price=stop_price,
                detail=str(getattr(e, "detail", "") or "protective stop create failed"),
                client_order_id=new_cid,
            )
        return {
            "status": "OPEN",
            "reason": _auto_reason_pack(new_state),
        }
    return None


async def _auto_refresh_open_records(user_id: int, sync_updates: bool = True) -> List[Dict[str, Any]]:
    try:
        rows = await _sb_request(
            "GET",
            "/rest/v1/auto_trade_records",
            params={
                "select": "id,user_id,symbol,market,interval,mode,side,status,entry_price,take_profit_price,stop_loss_price,qty,notional_usdt,opened_ms,closed_ms,close_price,pnl_usdt,signal_buy_pct,signal_sell_pct,signal_confidence,decision_diff,reason",
                "user_id": f"eq.{int(user_id)}",
                "status": "eq.OPEN",
                "order": "opened_ms.desc",
                "limit": "200",
            },
        )
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_records"):
            raise _auto_table_missing_error("auto_trade_records")
        raise
    open_rows = rows if isinstance(rows, list) else []
    if not open_rows:
        return []
    now_ts = time()
    now_ms = int(now_ts * 1000)
    cache = await _auto_fetch_kline_cache(open_rows, now_ts)
    out: List[Dict[str, Any]] = []
    for row in open_rows:
        cur = dict(row)
        old_state = _auto_reason_unpack(cur.get("reason"), cur)
        if sync_updates and _AUTO_LIVE_TRADING_ENABLED and int(old_state.get("v", 0) or 0) >= 3:
            try:
                exch_patch = await _auto_reconcile_exchange_state_for_open_record(
                    user_id=user_id,
                    record=cur,
                    state=old_state,
                )
            except HTTPException as ex:
                await _auto_audit_log(
                    user_id=user_id,
                    event="RECONCILE_FAIL",
                    level="ERROR",
                    record_id=_auto_int_or_none(cur.get("id")),
                    symbol=str(cur.get("symbol", "")).upper(),
                    market=str(cur.get("market", "spot")).lower(),
                    mode=str(cur.get("mode", "")),
                    side=str(old_state.get("side", cur.get("side", ""))).upper(),
                    status="OPEN",
                    qty=float(old_state.get("remaining_qty", cur.get("qty", 0.0)) or 0.0),
                    price=float(cur.get("entry_price", 0.0) or 0.0),
                    detail=str(getattr(ex, "detail", "") or "exchange reconcile failed"),
                )
                exch_patch = {
                    "status": "OPEN",
                    "reason": _auto_reason_pack(
                        {
                            **old_state,
                            "last_exec_error": f"exchange reconcile failed: {str(getattr(ex, 'detail', '') or '')[:120]}",
                            "last_exec_error_ms": int(time() * 1000),
                        }
                    ),
                }
            if isinstance(exch_patch, dict) and exch_patch:
                for kx, vx in exch_patch.items():
                    cur[kx] = vx
                await _sb_request(
                    "PATCH",
                    "/rest/v1/auto_trade_records",
                    params={"id": f"eq.{cur.get('id')}", "user_id": f"eq.{int(user_id)}"},
                    json_body=exch_patch,
                )
                if str(cur.get("status", "")).upper() != "OPEN":
                    out.append(cur)
                    continue
                old_state = _auto_reason_unpack(cur.get("reason"), cur)
        key = f"{str(cur.get('symbol', '')).upper()}:{str(cur.get('market', 'spot')).lower()}:{str(cur.get('interval', '5m'))}"
        eval_result = _auto_eval_record_status(cur, cache.get(key, []), now_ms)
        if bool(eval_result.get("changed")):
            patch = dict(eval_result.get("patch") or {})
            if patch:
                new_state = _auto_reason_unpack(patch.get("reason"), {**cur, **patch})
                try:
                    can_live_exec = _AUTO_LIVE_TRADING_ENABLED and int(old_state.get("v", 0) or 0) >= 3
                    if sync_updates and can_live_exec:
                        patch_status = str(patch.get("status", "")).upper()
                        if patch_status == "OPEN":
                            tp1_old = bool(old_state.get("tp1_hit", False))
                            tp1_new = bool(new_state.get("tp1_hit", False))
                            if tp1_new and not tp1_old:
                                live_patch = await _auto_apply_live_partial_close(
                                    user_id=user_id,
                                    record=cur,
                                    old_state=old_state,
                                    new_state=new_state,
                                )
                                if live_patch:
                                    patch = live_patch
                        elif patch_status in {"TP", "SL", "CLOSED_FAIL"}:
                            live_patch = await _auto_apply_live_final_close(
                                user_id=user_id,
                                record=cur,
                                old_state=old_state,
                                close_status=patch_status,
                            )
                            if live_patch:
                                patch = live_patch
                except HTTPException as e:
                    err_msg = str(getattr(e, "detail", "") or "order execution failed")
                    await _auto_audit_log(
                        user_id=user_id,
                        event="SYNC_EXEC_FAIL",
                        level="ERROR",
                        record_id=_auto_int_or_none(cur.get("id")),
                        symbol=str(cur.get("symbol", "")).upper(),
                        market=str(cur.get("market", "spot")).lower(),
                        mode=str(cur.get("mode", "")),
                        side=str(old_state.get("side", cur.get("side", ""))).upper(),
                        status=str(patch.get("status", "OPEN")).upper(),
                        qty=float(old_state.get("remaining_qty", cur.get("qty", 0.0)) or 0.0),
                        price=float(cur.get("entry_price", 0.0) or 0.0),
                        detail=err_msg,
                    )
                    fail_state = {
                        **old_state,
                        "last_exec_error": err_msg[:180],
                        "last_exec_error_ms": int(time() * 1000),
                    }
                    patch = {
                        "status": "OPEN",
                        "qty": round(float(old_state.get("remaining_qty", cur.get("qty", 0.0)) or 0.0), 8),
                        "pnl_usdt": round(float(old_state.get("realized_pnl_usdt", cur.get("pnl_usdt", 0.0)) or 0.0), 8),
                        "reason": _auto_reason_pack(fail_state),
                    }
                for k2, v2 in patch.items():
                    cur[k2] = v2
                if sync_updates:
                    await _sb_request(
                        "PATCH",
                        "/rest/v1/auto_trade_records",
                        params={"id": f"eq.{cur.get('id')}", "user_id": f"eq.{int(user_id)}"},
                        json_body=patch,
                    )
        out.append(cur)
    return out


async def _auto_today_realized_pnl(user_id: int) -> float:
    now_ms = int(time() * 1000)
    day_start_ms = _kst_day_start_ms(now_ms)
    try:
        rows = await _sb_request(
            "GET",
            "/rest/v1/auto_trade_records",
            params={
                "select": "id,status,pnl_usdt,closed_ms",
                "user_id": f"eq.{int(user_id)}",
                "status": "in.(TP,SL,CLOSED_FAIL)",
                "closed_ms": f"gte.{day_start_ms}",
                "limit": "2000",
            },
        )
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_records"):
            raise _auto_table_missing_error("auto_trade_records")
        raise
    items = rows if isinstance(rows, list) else []
    pnl = 0.0
    for r in items:
        try:
            pnl += float(r.get("pnl_usdt", 0.0) or 0.0)
        except Exception:
            continue
    return float(round(pnl, 8))


async def _auto_rescue_open_record(
    *,
    user_id: int,
    create_body: Dict[str, Any],
    reason_state: Dict[str, Any],
    event_prefix: str,
    symbol: str,
    market: str,
    mode: str,
    side: str,
    qty: float,
    entry_price: float,
    client_order_id: str,
    detail: str,
    payload: Any = None,
) -> Dict[str, Any] | None:
    body = dict(create_body or {})
    body["status"] = "OPEN"
    body["close_price"] = None
    body["closed_ms"] = None
    body["pnl_usdt"] = round(float(body.get("pnl_usdt", 0.0) or 0.0), 8)
    body["reason"] = _auto_reason_pack(reason_state)
    event_head = str(event_prefix or "RESCUE").upper()[:20]
    try:
        created = await _sb_request("POST", "/rest/v1/auto_trade_records", json_body=[body])
        rec = created[0] if isinstance(created, list) and created else body
        await _auto_audit_log(
            user_id=user_id,
            event=f"{event_head}_RESCUE_RECORD_OK",
            level="WARN",
            record_id=_auto_int_or_none(rec.get("id") if isinstance(rec, dict) else None),
            symbol=symbol,
            market=market,
            mode=mode,
            side=side,
            status="OPEN",
            qty=qty,
            price=entry_price,
            client_order_id=client_order_id,
            detail=detail[:240],
            payload=payload,
        )
        return rec if isinstance(rec, dict) else body
    except HTTPException as e:
        await _auto_audit_log(
            user_id=user_id,
            event=f"{event_head}_RESCUE_RECORD_FAIL",
            level="ERROR",
            symbol=symbol,
            market=market,
            mode=mode,
            side=side,
            status="OPEN",
            qty=qty,
            price=entry_price,
            client_order_id=client_order_id,
            detail=f"rescue record save failed: {str(getattr(e, 'detail', '') or '')[:220]}",
            payload=payload,
        )
        return None
    except Exception as e:
        await _auto_audit_log(
            user_id=user_id,
            event=f"{event_head}_RESCUE_RECORD_FAIL",
            level="ERROR",
            symbol=symbol,
            market=market,
            mode=mode,
            side=side,
            status="OPEN",
            qty=qty,
            price=entry_price,
            client_order_id=client_order_id,
            detail=f"rescue record save failed: {type(e).__name__}",
            payload=payload,
        )
        return None


def _auto_mask_api_key(api_key: Any) -> str:
    raw = str(api_key or "").strip()
    if not raw:
        return ""
    if len(raw) <= 8:
        return "*" * len(raw)
    return f"{raw[:4]}{'*' * (len(raw) - 8)}{raw[-4:]}"


def _auto_secret_fernet() -> Any:
    try:
        from cryptography.fernet import Fernet
    except Exception:
        raise HTTPException(status_code=500, detail="cryptography package is required")
    raw = str(os.getenv("AUTO_TRADE_SECRET_ENC_KEY", "")).strip()
    if not raw:
        raw = str(os.getenv("APP_SECRET_KEY", "")).strip()
    if not raw:
        raise HTTPException(status_code=500, detail="AUTO_TRADE_SECRET_ENC_KEY is required")
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def _auto_encrypt_secret(secret: str) -> str:
    raw = str(secret or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="api_secret is required")
    token = _auto_secret_fernet().encrypt(raw.encode("utf-8")).decode("utf-8")
    return f"v1:{token}"


def _auto_decrypt_secret(enc_value: str) -> str:
    try:
        from cryptography.fernet import InvalidToken
    except Exception:
        raise HTTPException(status_code=500, detail="cryptography package is required")
    raw = str(enc_value or "").strip()
    if not raw:
        return ""
    if not raw.startswith("v1:"):
        return raw
    token = raw[3:]
    try:
        return _auto_secret_fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        raise HTTPException(status_code=500, detail="failed to decrypt stored api_secret")


def _auto_public_binance_link(row: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(row, dict) or not row:
        return {
            "linked": False,
            "market": "spot",
            "api_key_masked": "",
            "status": "DISCONNECTED",
            "linked_ms": 0,
            "updated_ms": 0,
        }
    status = str(row.get("status", "")).upper()
    api_key = str(row.get("api_key", "")).strip()
    linked = bool(api_key) and status in _AUTO_LINK_STATUS
    market_v = "both" if linked else str(row.get("market", "spot") or "spot").lower()
    return {
        "linked": linked,
        "market": market_v,
        "api_key_masked": _auto_mask_api_key(api_key),
        "status": status or ("CONNECTED" if linked else "DISCONNECTED"),
        "linked_ms": int(row.get("linked_ms", 0) or 0),
        "updated_ms": int(row.get("updated_ms", 0) or 0),
    }


async def _auto_get_binance_link_row(user_id: int) -> Dict[str, Any] | None:
    try:
        rows = await _sb_request(
            "GET",
            "/rest/v1/auto_trade_binance_links",
            params={
                "select": "id,user_id,market,api_key,api_secret,status,linked_ms,updated_ms",
                "user_id": f"eq.{int(user_id)}",
                "limit": "1",
            },
        )
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_binance_links"):
            raise _auto_table_missing_error("auto_trade_binance_links")
        raise
    if isinstance(rows, list) and rows:
        return dict(rows[0])
    return None


def _auto_binance_verify_target(market: str) -> tuple[str, str]:
    m = str(market or "spot").strip().lower()
    if m == "futures":
        return BINANCE_FUTURES_BASE_URL, "/fapi/v2/account"
    return BINANCE_SPOT_BASE_URL, "/api/v3/account"


async def _auto_binance_signed_call(
    *,
    base_url: str,
    path: str,
    api_key: str,
    api_secret: str,
    method: str = "GET",
    extra_params: Dict[str, Any] | None = None,
) -> Any:
    key = str(api_key or "").strip()
    sec = str(api_secret or "").strip()
    if not key or not sec:
        raise HTTPException(status_code=400, detail="api_key/api_secret is required")
    req_method = str(method or "GET").upper()
    if req_method not in {"GET", "POST", "DELETE"}:
        raise HTTPException(status_code=400, detail="unsupported binance method")
    timestamp = int(time() * 1000)
    params: Dict[str, Any] = {"timestamp": timestamp, "recvWindow": 5000}
    if isinstance(extra_params, dict):
        params.update(extra_params)
    query = urlencode(params, doseq=True)
    signature = hmac.new(sec.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    req_path = f"{path}?{query}&signature={signature}"
    headers = {"X-MBX-APIKEY": key}
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=8.0) as c:
            r = await c.request(req_method, req_path, headers=headers)
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.NetworkError):
        raise HTTPException(status_code=502, detail="binance request failed")

    if r.status_code < 400:
        try:
            return r.json()
        except Exception:
            raise HTTPException(status_code=502, detail="binance response parse failed")
    msg = ""
    try:
        body = r.json()
        code = body.get("code") if isinstance(body, dict) else None
        reason = str(body.get("msg", "")).strip() if isinstance(body, dict) else str(body).strip()
        if code is not None:
            msg = f"{code}: {reason}"
        else:
            msg = reason
    except Exception:
        msg = str(r.text or "").strip()
    msg = msg[:220]
    raise HTTPException(status_code=400, detail=f"binance api 실패: {msg or 'invalid api key/secret'}")


async def _auto_binance_signed_get(*, market: str, api_key: str, api_secret: str) -> Dict[str, Any]:
    m = str(market or "spot").strip().lower()
    if m not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail="market must be spot or futures")
    base_url, path = _auto_binance_verify_target(m)
    body = await _auto_binance_signed_call(
        base_url=base_url,
        path=path,
        api_key=api_key,
        api_secret=api_secret,
        method="GET",
    )
    if isinstance(body, dict):
        return body
    raise HTTPException(status_code=502, detail=f"binance api 검증 실패({m}): invalid response")


async def _auto_verify_binance_credentials(*, market: str, api_key: str, api_secret: str) -> None:
    await _auto_binance_signed_get(market=market, api_key=api_key, api_secret=api_secret)


async def _auto_verify_binance_credentials_both(*, api_key: str, api_secret: str) -> None:
    errors: List[str] = []
    for market in ("spot", "futures"):
        try:
            await _auto_binance_signed_get(market=market, api_key=api_key, api_secret=api_secret)
        except HTTPException as e:
            detail = str(getattr(e, "detail", "")).strip() or "verification failed"
            errors.append(f"{market}: {detail}")
    if errors:
        raise HTTPException(status_code=400, detail=" / ".join(errors))


def _auto_decimal_floor_step(value: float, step: float) -> float:
    v = float(value)
    s = float(step)
    if not math.isfinite(v) or not math.isfinite(s) or v <= 0 or s <= 0:
        return 0.0
    dv = Decimal(str(v))
    ds = Decimal(str(s))
    q = (dv / ds).to_integral_value(rounding=ROUND_DOWN) * ds
    out = float(q)
    if out < 0:
        return 0.0
    return out


def _auto_client_id(prefix: str, *parts: Any) -> str:
    p = re.sub(r"[^a-z0-9]", "", str(prefix or "").lower())[:6] or "o"
    raw = "|".join(str(x or "") for x in parts)
    hx = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
    return f"at_{p}_{hx}"[:36]


def _auto_is_duplicate_order_error(detail: str) -> bool:
    d = str(detail or "").lower()
    return "duplicate" in d or "already exists" in d or "-2010" in d


def _auto_is_unknown_order_error(detail: str) -> bool:
    d = str(detail or "").lower()
    return "unknown order" in d or "order does not exist" in d or "-2011" in d


async def _auto_binance_public_get(*, market: str, path: str, params: Dict[str, Any] | None = None) -> Any:
    base_url = BINANCE_FUTURES_BASE_URL if str(market).lower() == "futures" else BINANCE_SPOT_BASE_URL
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=8.0) as c:
            r = await c.get(path, params=params)
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.NetworkError):
        raise HTTPException(status_code=502, detail="binance public request failed")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"binance public api failed: {r.status_code}")
    try:
        return r.json()
    except Exception:
        raise HTTPException(status_code=502, detail="binance public response parse failed")


async def _auto_fetch_symbol_trade_rules(*, market: str, symbol: str) -> Dict[str, float]:
    m = str(market or "").lower().strip()
    sym = str(symbol or "").upper().strip()
    if m not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail="invalid market")
    path = "/fapi/v1/exchangeInfo" if m == "futures" else "/api/v3/exchangeInfo"
    body = await _auto_binance_public_get(market=m, path=path, params={"symbol": sym})
    if not isinstance(body, dict):
        raise HTTPException(status_code=502, detail="exchangeInfo invalid response")
    symbols = body.get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        raise HTTPException(status_code=400, detail=f"unsupported symbol: {sym}")
    info = None
    for item in symbols:
        if str(item.get("symbol", "")).upper() == sym:
            info = item
            break
    if not isinstance(info, dict):
        raise HTTPException(status_code=400, detail=f"unsupported symbol: {sym}")
    filters = info.get("filters", [])
    if not isinstance(filters, list):
        filters = []

    step_size = 0.0
    min_qty = 0.0
    min_notional = 0.0
    tick_size = 0.0
    for f in filters:
        if not isinstance(f, dict):
            continue
        ft = str(f.get("filterType", "")).upper()
        if ft in {"LOT_SIZE", "MARKET_LOT_SIZE"}:
            try:
                step_v = float(f.get("stepSize", 0.0) or 0.0)
                min_v = float(f.get("minQty", 0.0) or 0.0)
            except Exception:
                continue
            if step_v > 0 and (step_size <= 0 or step_v < step_size):
                step_size = step_v
            if min_v > 0 and (min_qty <= 0 or min_v > min_qty):
                min_qty = min_v
        elif ft in {"MIN_NOTIONAL", "NOTIONAL"}:
            try:
                n_v = float(f.get("minNotional", f.get("notional", 0.0)) or 0.0)
            except Exception:
                n_v = 0.0
            if n_v > 0 and n_v > min_notional:
                min_notional = n_v
        elif ft == "PRICE_FILTER":
            try:
                t_v = float(f.get("tickSize", 0.0) or 0.0)
            except Exception:
                t_v = 0.0
            if t_v > 0:
                tick_size = t_v
    if step_size <= 0:
        step_size = 1e-8
    if min_qty <= 0:
        min_qty = step_size
    return {
        "step_size": float(step_size),
        "min_qty": float(min_qty),
        "min_notional": float(min_notional),
        "tick_size": float(tick_size),
    }


def _auto_extract_order_fill(order: Any, fallback_price: float = 0.0) -> Dict[str, Any]:
    o = order if isinstance(order, dict) else {}
    fills = o.get("fills", [])
    exec_qty = 0.0
    avg_price = 0.0
    base_fee = 0.0
    base_asset = str(o.get("symbol", "")).upper().replace("USDT", "")
    if isinstance(fills, list) and fills:
        sum_px_qty = 0.0
        sum_qty = 0.0
        for f in fills:
            if not isinstance(f, dict):
                continue
            try:
                p = float(f.get("price", 0.0) or 0.0)
                q = float(f.get("qty", 0.0) or 0.0)
            except Exception:
                continue
            if p <= 0 or q <= 0:
                continue
            sum_px_qty += p * q
            sum_qty += q
            fee_asset = str(f.get("commissionAsset", "")).upper().strip()
            try:
                fee_qty = float(f.get("commission", 0.0) or 0.0)
            except Exception:
                fee_qty = 0.0
            if fee_asset and fee_asset == base_asset and fee_qty > 0:
                base_fee += fee_qty
        if sum_qty > 0:
            exec_qty = sum_qty
            avg_price = sum_px_qty / sum_qty
    if exec_qty <= 0:
        for k in ("executedQty", "origQty"):
            try:
                q = float(o.get(k, 0.0) or 0.0)
            except Exception:
                q = 0.0
            if q > 0:
                exec_qty = q
                break
    if avg_price <= 0 and exec_qty > 0:
        for qk in ("cummulativeQuoteQty", "cumQuote"):
            try:
                quote_v = float(o.get(qk, 0.0) or 0.0)
            except Exception:
                quote_v = 0.0
            if quote_v > 0:
                avg_price = quote_v / exec_qty
                break
    if avg_price <= 0:
        for pk in ("avgPrice", "price"):
            try:
                p = float(o.get(pk, 0.0) or 0.0)
            except Exception:
                p = 0.0
            if p > 0:
                avg_price = p
                break
    if avg_price <= 0:
        avg_price = float(fallback_price or 0.0)
    return {
        "order_id": str(o.get("orderId", o.get("clientOrderId", "")) or ""),
        "executed_qty": float(max(0.0, exec_qty)),
        "avg_price": float(max(0.0, avg_price)),
        "base_fee_qty": float(max(0.0, base_fee)),
        "raw": o,
    }


async def _auto_fetch_order_by_client_id(
    *,
    market: str,
    symbol: str,
    client_order_id: str,
    api_key: str,
    api_secret: str,
) -> Dict[str, Any]:
    m = str(market or "").lower().strip()
    sym = str(symbol or "").upper().strip()
    cid = str(client_order_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="client_order_id is required")
    if m == "futures":
        base_url = BINANCE_FUTURES_BASE_URL
        path = "/fapi/v1/order"
    else:
        base_url = BINANCE_SPOT_BASE_URL
        path = "/api/v3/order"
    body = await _auto_binance_signed_call(
        base_url=base_url,
        path=path,
        api_key=api_key,
        api_secret=api_secret,
        method="GET",
        extra_params={"symbol": sym, "origClientOrderId": cid},
    )
    if isinstance(body, dict):
        return body
    raise HTTPException(status_code=502, detail="order query invalid response")


async def _auto_cancel_order(
    *,
    market: str,
    symbol: str,
    api_key: str,
    api_secret: str,
    order_id: str = "",
    client_order_id: str = "",
) -> None:
    m = str(market or "").lower().strip()
    sym = str(symbol or "").upper().strip()
    oid = str(order_id or "").strip()
    cid = str(client_order_id or "").strip()
    if not oid and not cid:
        return
    if m == "futures":
        base_url = BINANCE_FUTURES_BASE_URL
        path = "/fapi/v1/order"
    else:
        base_url = BINANCE_SPOT_BASE_URL
        path = "/api/v3/order"
    params: Dict[str, Any] = {"symbol": sym}
    if oid:
        params["orderId"] = oid
    else:
        params["origClientOrderId"] = cid
    try:
        await _auto_binance_signed_call(
            base_url=base_url,
            path=path,
            api_key=api_key,
            api_secret=api_secret,
            method="DELETE",
            extra_params=params,
        )
    except HTTPException as e:
        # 이미 체결/취소된 주문은 취소 실패를 무시한다.
        msg = str(getattr(e, "detail", "") or "").lower()
        if "unknown order" in msg or "-2011" in msg:
            return
        raise


async def _auto_place_market_order(
    *,
    market: str,
    symbol: str,
    side: str,
    qty: float,
    api_key: str,
    api_secret: str,
    reduce_only: bool = False,
    fallback_price: float = 0.0,
    client_order_id: str = "",
) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    sym = str(symbol or "").strip().upper()
    s = str(side or "").strip().upper()
    if m not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail="invalid market")
    if s not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="invalid order side")
    rules = await _auto_fetch_symbol_trade_rules(market=m, symbol=sym)
    step_size = float(rules.get("step_size", 0.0) or 0.0)
    min_qty = float(rules.get("min_qty", 0.0) or 0.0)
    min_notional = float(rules.get("min_notional", 0.0) or 0.0)
    qty_norm = _auto_decimal_floor_step(float(qty or 0.0), step_size)
    if qty_norm <= 0 or qty_norm + 1e-12 < min_qty:
        raise HTTPException(status_code=400, detail="order quantity is below minQty")
    if min_notional > 0 and fallback_price > 0 and (qty_norm * fallback_price) + 1e-8 < min_notional:
        raise HTTPException(status_code=400, detail="order notional is below minNotional")
    qty_str = format(qty_norm, ".16f").rstrip("0").rstrip(".")
    if not qty_str:
        raise HTTPException(status_code=400, detail="invalid normalized quantity")

    if m == "futures":
        path = "/fapi/v1/order"
        base_url = BINANCE_FUTURES_BASE_URL
    else:
        path = "/api/v3/order"
        base_url = BINANCE_SPOT_BASE_URL
    params: Dict[str, Any] = {
        "symbol": sym,
        "side": s,
        "type": "MARKET",
        "quantity": qty_str,
    }
    cid = str(client_order_id or "").strip()
    if cid:
        params["newClientOrderId"] = cid
    if m == "futures":
        params["newOrderRespType"] = "RESULT"
        if reduce_only:
            params["reduceOnly"] = "true"
    else:
        params["newOrderRespType"] = "FULL"

    try:
        raw = await _auto_binance_signed_call(
            base_url=base_url,
            path=path,
            api_key=api_key,
            api_secret=api_secret,
            method="POST",
            extra_params=params,
        )
    except HTTPException as e:
        detail = str(getattr(e, "detail", "") or "")
        if cid and _auto_is_duplicate_order_error(detail):
            raw = await _auto_fetch_order_by_client_id(
                market=m,
                symbol=sym,
                client_order_id=cid,
                api_key=api_key,
                api_secret=api_secret,
            )
        else:
            raise
    fill = _auto_extract_order_fill(raw, fallback_price=fallback_price)
    fill["requested_qty"] = qty_norm
    fill["symbol"] = sym
    fill["market"] = m
    fill["side"] = s
    fill["client_order_id"] = cid
    return fill


async def _auto_place_protective_stop_order(
    *,
    market: str,
    symbol: str,
    side: str,
    stop_price: float,
    qty: float,
    api_key: str,
    api_secret: str,
    client_order_id: str,
) -> Dict[str, Any]:
    m = str(market or "").lower().strip()
    sym = str(symbol or "").upper().strip()
    s = str(side or "").upper().strip()
    if m not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail="invalid market")
    if s not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="invalid stop side")
    rules = await _auto_fetch_symbol_trade_rules(market=m, symbol=sym)
    step_size = float(rules.get("step_size", 0.0) or 0.0)
    tick_size = float(rules.get("tick_size", 0.0) or 0.0)
    qty_norm = _auto_decimal_floor_step(float(qty or 0.0), step_size)
    if qty_norm <= 0:
        raise HTTPException(status_code=400, detail="protective stop quantity is invalid")
    stop_raw = float(stop_price or 0.0)
    if stop_raw <= 0:
        raise HTTPException(status_code=400, detail="protective stop price is invalid")
    stop_norm = _auto_decimal_floor_step(stop_raw, tick_size) if tick_size > 0 else stop_raw
    stop_norm = max(0.0, stop_norm)
    if stop_norm <= 0:
        raise HTTPException(status_code=400, detail="protective stop price normalization failed")
    qty_str = format(qty_norm, ".16f").rstrip("0").rstrip(".")
    stop_str = format(stop_norm, ".16f").rstrip("0").rstrip(".")
    cid = str(client_order_id or "").strip()
    if m == "futures":
        try:
            body = await _auto_binance_signed_call(
                base_url=BINANCE_FUTURES_BASE_URL,
                path="/fapi/v1/order",
                api_key=api_key,
                api_secret=api_secret,
                method="POST",
                extra_params={
                    "symbol": sym,
                    "side": s,
                    "type": "STOP_MARKET",
                    "reduceOnly": "true",
                    "quantity": qty_str,
                    "stopPrice": stop_str,
                    "workingType": "MARK_PRICE",
                    "newClientOrderId": cid,
                },
            )
        except HTTPException as e:
            detail = str(getattr(e, "detail", "") or "")
            if cid and _auto_is_duplicate_order_error(detail):
                body = await _auto_fetch_order_by_client_id(
                    market=m,
                    symbol=sym,
                    client_order_id=cid,
                    api_key=api_key,
                    api_secret=api_secret,
                )
            else:
                raise
        if not isinstance(body, dict):
            raise HTTPException(status_code=502, detail="protective stop response invalid")
        return {
            "order_id": str(body.get("orderId", "") or ""),
            "client_order_id": cid,
            "type": "STOP_MARKET",
            "stop_price": round(stop_norm, 8),
            "qty": round(qty_norm, 8),
        }

    # spot: STOP_LOSS_LIMIT 보호주문
    limit_raw = stop_norm * (0.998 if s == "SELL" else 1.002)
    limit_norm = _auto_decimal_floor_step(limit_raw, tick_size) if tick_size > 0 else limit_raw
    if s == "SELL" and limit_norm > stop_norm:
        limit_norm = stop_norm
    if s == "BUY" and limit_norm < stop_norm:
        limit_norm = stop_norm
    if limit_norm <= 0:
        raise HTTPException(status_code=400, detail="protective limit price is invalid")
    price_str = format(limit_norm, ".16f").rstrip("0").rstrip(".")
    try:
        body = await _auto_binance_signed_call(
            base_url=BINANCE_SPOT_BASE_URL,
            path="/api/v3/order",
            api_key=api_key,
            api_secret=api_secret,
            method="POST",
            extra_params={
                "symbol": sym,
                "side": s,
                "type": "STOP_LOSS_LIMIT",
                "timeInForce": "GTC",
                "quantity": qty_str,
                "stopPrice": stop_str,
                "price": price_str,
                "newOrderRespType": "RESULT",
                "newClientOrderId": cid,
            },
        )
    except HTTPException as e:
        detail = str(getattr(e, "detail", "") or "")
        if cid and _auto_is_duplicate_order_error(detail):
            body = await _auto_fetch_order_by_client_id(
                market=m,
                symbol=sym,
                client_order_id=cid,
                api_key=api_key,
                api_secret=api_secret,
            )
        else:
            raise
    if not isinstance(body, dict):
        raise HTTPException(status_code=502, detail="protective stop response invalid")
    return {
        "order_id": str(body.get("orderId", "") or ""),
        "client_order_id": cid,
        "type": "STOP_LOSS_LIMIT",
        "stop_price": round(stop_norm, 8),
        "limit_price": round(limit_norm, 8),
        "qty": round(qty_norm, 8),
    }


async def _auto_fetch_live_position_qty(
    *,
    market: str,
    symbol: str,
    api_key: str,
    api_secret: str,
) -> float:
    m = str(market or "").lower().strip()
    sym = str(symbol or "").upper().strip()
    if m == "futures":
        body = await _auto_binance_signed_get(market="futures", api_key=api_key, api_secret=api_secret)
        rows = body.get("positions", []) if isinstance(body, dict) else []
        if not isinstance(rows, list):
            return 0.0
        for r in rows:
            if not isinstance(r, dict):
                continue
            if str(r.get("symbol", "")).upper() != sym:
                continue
            try:
                amt = abs(float(r.get("positionAmt", 0.0) or 0.0))
            except Exception:
                amt = 0.0
            return float(max(0.0, amt))
        return 0.0

    body = await _auto_binance_signed_get(market="spot", api_key=api_key, api_secret=api_secret)
    balances = body.get("balances", []) if isinstance(body, dict) else []
    if not isinstance(balances, list):
        return 0.0
    base_asset = sym[:-4] if sym.endswith("USDT") else sym
    for b in balances:
        if not isinstance(b, dict):
            continue
        if str(b.get("asset", "")).upper().strip() != base_asset:
            continue
        try:
            free_v = float(b.get("free", 0.0) or 0.0)
            locked_v = float(b.get("locked", 0.0) or 0.0)
        except Exception:
            return 0.0
        return float(max(0.0, free_v + locked_v))
    return 0.0


async def _auto_fetch_last_price(*, market: str, symbol: str) -> float:
    m = str(market or "").lower().strip()
    sym = str(symbol or "").upper().strip()
    path = "/fapi/v1/ticker/price" if m == "futures" else "/api/v3/ticker/price"
    body = await _auto_binance_public_get(market=m, path=path, params={"symbol": sym})
    if not isinstance(body, dict):
        return 0.0
    try:
        px = float(body.get("price", 0.0) or 0.0)
    except Exception:
        px = 0.0
    return float(max(0.0, px))


async def _auto_get_binance_credentials(user_id: int) -> tuple[str, str]:
    row = await _auto_get_binance_link_row(user_id)
    if not isinstance(row, dict):
        raise HTTPException(status_code=404, detail="binance link not found")
    status = str(row.get("status", "")).upper()
    api_key = str(row.get("api_key", "")).strip()
    enc_secret = str(row.get("api_secret", "")).strip()
    if not api_key or not enc_secret or status not in _AUTO_LINK_STATUS:
        raise HTTPException(status_code=400, detail="binance is not linked")
    api_secret = _auto_decrypt_secret(enc_secret)
    return api_key, api_secret


def _auto_parse_collateral_from_account(*, market: str, body: Dict[str, Any]) -> Dict[str, Any]:
    m = str(market or "spot").strip().lower()
    if m == "futures":
        available = float(body.get("availableBalance", 0.0) or 0.0)
        total = float(body.get("totalMarginBalance", body.get("totalWalletBalance", 0.0)) or 0.0)
        asset_count = 1 if total > 0 else 0
        return {
            "market": "futures",
            "asset": "USDT",
            "available_usdt": round(available, 4),
            "total_usdt": round(total, 4),
            "asset_count": asset_count,
        }

    return {"market": "spot", "asset": "USDT", "available_usdt": 0.0, "total_usdt": 0.0, "asset_count": 0}


async def _auto_fetch_spot_price_map() -> Dict[str, float]:
    path = "/api/v3/ticker/price"
    try:
        async with httpx.AsyncClient(base_url=BINANCE_SPOT_BASE_URL, timeout=8.0) as c:
            r = await c.get(path)
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.NetworkError):
        raise HTTPException(status_code=502, detail="binance spot ticker request failed")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"binance spot ticker failed: {r.status_code}")
    try:
        rows = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail="binance spot ticker parse failed")
    if not isinstance(rows, list):
        raise HTTPException(status_code=502, detail="binance spot ticker invalid response")
    out: Dict[str, float] = {}
    for row in rows:
        sym = str(row.get("symbol", "")).upper()
        try:
            px = float(row.get("price", 0.0) or 0.0)
        except Exception:
            continue
        if sym and math.isfinite(px) and px > 0:
            out[sym] = px
    return out


def _auto_asset_usdt_price(asset: str, price_map: Dict[str, float]) -> float:
    a = str(asset or "").upper().strip()
    if not a:
        return 0.0
    stable_assets = {"USDT", "USDC", "FDUSD", "BUSD", "TUSD"}
    if a in stable_assets:
        return 1.0
    direct = float(price_map.get(f"{a}USDT", 0.0) or 0.0)
    if direct > 0:
        return direct
    bridges = [
        ("BTC", "BTCUSDT"),
        ("ETH", "ETHUSDT"),
        ("BNB", "BNBUSDT"),
        ("FDUSD", "FDUSDUSDT"),
        ("USDC", "USDCUSDT"),
        ("BUSD", "BUSDUSDT"),
    ]
    for quote, quote_usdt in bridges:
        cross = float(price_map.get(f"{a}{quote}", 0.0) or 0.0)
        if cross <= 0:
            continue
        if quote in stable_assets:
            return cross
        bridge_px = float(price_map.get(quote_usdt, 0.0) or 0.0)
        if bridge_px > 0:
            return cross * bridge_px
    return 0.0


def _auto_parse_spot_portfolio_collateral(*, body: Dict[str, Any], price_map: Dict[str, float]) -> Dict[str, Any]:
    balances = body.get("balances", [])
    if not isinstance(balances, list):
        return {
            "market": "spot",
            "asset": "PORTFOLIO",
            "available_usdt": 0.0,
            "total_usdt": 0.0,
            "unknown_assets": 0,
            "asset_count": 0,
        }
    available_usdt = 0.0
    total_usdt = 0.0
    unknown_assets = 0
    asset_count = 0
    for item in balances:
        asset = str(item.get("asset", "")).upper().strip()
        if not asset:
            continue
        try:
            free_v = float(item.get("free", 0.0) or 0.0)
            locked_v = float(item.get("locked", 0.0) or 0.0)
        except Exception:
            continue
        if not math.isfinite(free_v) or not math.isfinite(locked_v):
            continue
        qty_total = free_v + locked_v
        if qty_total <= 0:
            continue
        asset_count += 1
        px = _auto_asset_usdt_price(asset, price_map)
        if px <= 0:
            unknown_assets += 1
            continue
        available_usdt += max(0.0, free_v) * px
        total_usdt += max(0.0, qty_total) * px
    return {
        "market": "spot",
        "asset": "PORTFOLIO",
        "available_usdt": round(available_usdt, 4),
        "total_usdt": round(total_usdt, 4),
        "unknown_assets": int(unknown_assets),
        "asset_count": int(asset_count),
    }


def _auto_parse_funding_portfolio_collateral(*, body: Any, price_map: Dict[str, float]) -> Dict[str, Any]:
    if not isinstance(body, list):
        return {
            "market": "funding",
            "asset": "PORTFOLIO",
            "available_usdt": 0.0,
            "total_usdt": 0.0,
            "unknown_assets": 0,
            "asset_count": 0,
        }
    available_usdt = 0.0
    total_usdt = 0.0
    unknown_assets = 0
    asset_count = 0
    for item in body:
        if not isinstance(item, dict):
            continue
        asset = str(item.get("asset", "")).upper().strip()
        if not asset:
            continue
        try:
            free_v = float(item.get("free", 0.0) or 0.0)
            locked_v = float(item.get("locked", 0.0) or 0.0)
            freeze_v = float(item.get("freeze", 0.0) or 0.0)
            withdrawing_v = float(item.get("withdrawing", 0.0) or 0.0)
        except Exception:
            continue
        if not all(math.isfinite(v) for v in [free_v, locked_v, freeze_v, withdrawing_v]):
            continue
        qty_total = free_v + locked_v + freeze_v + withdrawing_v
        if qty_total <= 0:
            continue
        asset_count += 1
        px = _auto_asset_usdt_price(asset, price_map)
        if px <= 0:
            unknown_assets += 1
            continue
        available_usdt += max(0.0, free_v) * px
        total_usdt += max(0.0, qty_total) * px
    return {
        "market": "funding",
        "asset": "PORTFOLIO",
        "available_usdt": round(available_usdt, 4),
        "total_usdt": round(total_usdt, 4),
        "unknown_assets": int(unknown_assets),
        "asset_count": int(asset_count),
    }


async def _auto_fetch_market_collateral(*, market: str, api_key: str, api_secret: str) -> Dict[str, Any]:
    try:
        body = await _auto_binance_signed_get(market=market, api_key=api_key, api_secret=api_secret)
        if str(market).lower() == "spot":
            price_map = await _auto_fetch_spot_price_map()
            parsed = _auto_parse_spot_portfolio_collateral(body=body, price_map=price_map)
        else:
            parsed = _auto_parse_collateral_from_account(market=market, body=body)
        return {
            "ok": True,
            "asset": str(parsed.get("asset", "USDT")),
            "available_usdt": float(parsed.get("available_usdt", 0.0) or 0.0),
            "total_usdt": float(parsed.get("total_usdt", 0.0) or 0.0),
            "unknown_assets": int(parsed.get("unknown_assets", 0) or 0),
            "asset_count": int(parsed.get("asset_count", 0) or 0),
            "error": "",
        }
    except HTTPException as e:
        return {
            "ok": False,
            "asset": "USDT",
            "available_usdt": 0.0,
            "total_usdt": 0.0,
            "unknown_assets": 0,
            "asset_count": 0,
            "error": str(getattr(e, "detail", "") or "request failed"),
        }


async def _auto_fetch_funding_collateral(*, api_key: str, api_secret: str) -> Dict[str, Any]:
    try:
        body = await _auto_binance_signed_call(
            base_url=BINANCE_SPOT_BASE_URL,
            path="/sapi/v1/asset/get-funding-asset",
            api_key=api_key,
            api_secret=api_secret,
            method="POST",
            extra_params={"needBtcValuation": "true"},
        )
        price_map = await _auto_fetch_spot_price_map()
        parsed = _auto_parse_funding_portfolio_collateral(body=body, price_map=price_map)
        return {
            "ok": True,
            "asset": str(parsed.get("asset", "USDT")),
            "available_usdt": float(parsed.get("available_usdt", 0.0) or 0.0),
            "total_usdt": float(parsed.get("total_usdt", 0.0) or 0.0),
            "unknown_assets": int(parsed.get("unknown_assets", 0) or 0),
            "asset_count": int(parsed.get("asset_count", 0) or 0),
            "error": "",
        }
    except HTTPException as e:
        return {
            "ok": False,
            "asset": "USDT",
            "available_usdt": 0.0,
            "total_usdt": 0.0,
            "unknown_assets": 0,
            "asset_count": 0,
            "error": str(getattr(e, "detail", "") or "request failed"),
        }


async def _auto_fetch_binance_collateral(user_id: int) -> Dict[str, Any]:
    row = await _auto_get_binance_link_row(user_id)
    if not isinstance(row, dict):
        raise HTTPException(status_code=404, detail="binance link not found")
    status = str(row.get("status", "")).upper()
    api_key = str(row.get("api_key", "")).strip()
    enc_secret = str(row.get("api_secret", "")).strip()
    if not api_key or not enc_secret or status not in _AUTO_LINK_STATUS:
        raise HTTPException(status_code=400, detail="binance is not linked")
    api_secret = _auto_decrypt_secret(enc_secret)
    spot, fut, funding = await asyncio.gather(
        _auto_fetch_market_collateral(market="spot", api_key=api_key, api_secret=api_secret),
        _auto_fetch_market_collateral(market="futures", api_key=api_key, api_secret=api_secret),
        _auto_fetch_funding_collateral(api_key=api_key, api_secret=api_secret),
    )
    if not bool(spot.get("ok")) and not bool(fut.get("ok")) and not bool(funding.get("ok")):
        spot_err = str(spot.get("error", "")).strip() or "failed"
        fut_err = str(fut.get("error", "")).strip() or "failed"
        fund_err = str(funding.get("error", "")).strip() or "failed"
        raise HTTPException(status_code=400, detail=f"spot: {spot_err} / futures: {fut_err} / funding: {fund_err}")
    total_available = 0.0
    total_balance = 0.0
    total_assets = 0
    for item in (spot, fut, funding):
        if not bool(item.get("ok")):
            continue
        total_available += float(item.get("available_usdt", 0.0) or 0.0)
        total_balance += float(item.get("total_usdt", 0.0) or 0.0)
        total_assets += int(item.get("asset_count", 0) or 0)
    return {
        "asset": "USDT",
        "spot": spot,
        "futures": fut,
        "funding": funding,
        "total_available_usdt": round(total_available, 4),
        "total_usdt": round(total_balance, 4),
        "asset_count": int(total_assets),
        "updated_ms": int(time() * 1000),
    }


async def _auto_upsert_binance_link(user_id: int, *, api_key: str, api_secret: str) -> Dict[str, Any]:
    now_ms = int(time() * 1000)
    enc_secret = _auto_encrypt_secret(api_secret)
    payload = {
        "market": "both",
        "api_key": str(api_key or "").strip(),
        "api_secret": enc_secret,
        "status": "CONNECTED",
        "linked_ms": now_ms,
        "updated_ms": now_ms,
    }
    existing = await _auto_get_binance_link_row(user_id)
    try:
        if isinstance(existing, dict) and existing.get("id") is not None:
            rows = await _sb_request(
                "PATCH",
                "/rest/v1/auto_trade_binance_links",
                params={"id": f"eq.{existing.get('id')}", "user_id": f"eq.{int(user_id)}"},
                json_body=payload,
            )
        else:
            rows = await _sb_request(
                "POST",
                "/rest/v1/auto_trade_binance_links",
                json_body=[{"user_id": int(user_id), **payload}],
            )
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_binance_links"):
            raise _auto_table_missing_error("auto_trade_binance_links")
        raise

    if isinstance(rows, list) and rows:
        return dict(rows[0])
    latest = await _auto_get_binance_link_row(user_id)
    if isinstance(latest, dict):
        return latest
    raise HTTPException(status_code=500, detail="failed to save binance link")


async def _auto_delete_binance_link(user_id: int) -> None:
    try:
        await _sb_request(
            "DELETE",
            "/rest/v1/auto_trade_binance_links",
            params={"user_id": f"eq.{int(user_id)}"},
        )
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_binance_links"):
            raise _auto_table_missing_error("auto_trade_binance_links")
        raise


def _auto_validate_config_required_fields(cfg: Dict[str, Any]) -> None:
    symbol = str(cfg.get("symbol", "")).upper().strip()
    interval = str(cfg.get("interval", "")).strip()
    mode = str(cfg.get("mode", "")).lower().strip()
    order_size = float(cfg.get("order_size_usdt", 0.0) or 0.0)
    daily_loss = float(cfg.get("daily_max_loss_usdt", 0.0) or 0.0)
    max_open = int(float(cfg.get("max_open_positions", 0) or 0))
    if symbol not in _AUTO_ALLOWED_SYMBOLS:
        raise HTTPException(status_code=400, detail="설정 저장 불가: 코인 선택 값이 올바르지 않습니다.")
    if interval not in {"5m", "1h", "4h"}:
        raise HTTPException(status_code=400, detail="설정 저장 불가: 분석 타임 값이 올바르지 않습니다.")
    if mode not in _AUTO_ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="설정 저장 불가: 투자모드 값이 올바르지 않습니다.")
    if order_size < 10.0:
        raise HTTPException(status_code=400, detail="설정 저장 불가: 1회 거래금액은 10 이상이어야 합니다.")
    if daily_loss < 1.0:
        raise HTTPException(status_code=400, detail="설정 저장 불가: 일 최대 손실은 1 이상이어야 합니다.")
    if max_open < 1 or max_open > 5:
        raise HTTPException(status_code=400, detail="설정 저장 불가: 동시 최대 포지션은 1~5 범위여야 합니다.")
    tp_pct = float(cfg.get("take_profit_pct", 0.0) or 0.0)
    sl_pct = float(cfg.get("stop_loss_pct", 0.0) or 0.0)
    if tp_pct < 0 or sl_pct < 0:
        raise HTTPException(status_code=400, detail="설정 저장 불가: 익절/손절 값이 올바르지 않습니다.")


async def _auto_require_binance_link(user_id: int) -> None:
    await _auto_get_binance_credentials(user_id)


async def _auto_market_available_usdt(user_id: int, market: str) -> float:
    api_key, api_secret = await _auto_get_binance_credentials(user_id)
    m = str(market or "").lower().strip()
    if m == "spot":
        body = await _auto_binance_signed_get(market="spot", api_key=api_key, api_secret=api_secret)
        balances = body.get("balances", []) if isinstance(body, dict) else []
        if not isinstance(balances, list):
            raise HTTPException(status_code=400, detail="spot balance format is invalid")
        usdt_free = 0.0
        for item in balances:
            if not isinstance(item, dict):
                continue
            if str(item.get("asset", "")).upper().strip() != "USDT":
                continue
            try:
                usdt_free = float(item.get("free", 0.0) or 0.0)
            except Exception:
                usdt_free = 0.0
            break
        return float(max(0.0, usdt_free))

    info = await _auto_fetch_market_collateral(market="futures", api_key=api_key, api_secret=api_secret)
    if not bool(info.get("ok")):
        msg = str(info.get("error", "")).strip() or "담보금 조회 실패"
        raise HTTPException(status_code=400, detail=msg)
    return float(info.get("available_usdt", 0.0) or 0.0)


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


@app.get("/api/auto_trade/config")
async def api_auto_trade_get_config(
    request: Request,
):
    _require_cfg_unlock(request)
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    cfg = await _auto_get_or_create_config(user_id)
    return {"ok": True, "config": cfg}


@app.post("/api/auto_trade/config")
async def api_auto_trade_save_config(
    request: Request,
    payload: Dict[str, Any] = Body(default={}),
):
    _require_cfg_unlock(request)
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    cur = await _auto_get_or_create_config(user_id)
    merged = dict(cur)
    merged.update(dict(payload or {}))
    _auto_validate_config_required_fields(merged)
    if bool(merged.get("enabled")):
        await _auto_require_binance_link(user_id)
    cfg = await _auto_update_config(user_id, cur.get("id"), merged)
    return {"ok": True, "config": cfg}


@app.get("/api/auto_trade/binance/link")
async def api_auto_trade_get_binance_link(
    request: Request,
):
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    row = await _auto_get_binance_link_row(user_id)
    return {"ok": True, "link": _auto_public_binance_link(row)}


@app.get("/api/auto_trade/binance/collateral")
async def api_auto_trade_get_binance_collateral(
    request: Request,
):
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    collateral = await _auto_fetch_binance_collateral(user_id)
    return {"ok": True, "collateral": collateral}


@app.post("/api/auto_trade/binance/link")
async def api_auto_trade_upsert_binance_link(
    request: Request,
    payload: Dict[str, Any] = Body(default={}),
):
    _require_cfg_unlock(request)
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    api_key = str(payload.get("api_key", "")).strip()
    api_secret = str(payload.get("api_secret", "")).strip()
    if len(api_key) < 12 or len(api_secret) < 12:
        raise HTTPException(status_code=400, detail="api_key/api_secret length is too short")
    await _auto_verify_binance_credentials_both(api_key=api_key, api_secret=api_secret)
    row = await _auto_upsert_binance_link(user_id, api_key=api_key, api_secret=api_secret)
    return {"ok": True, "link": _auto_public_binance_link(row)}


@app.delete("/api/auto_trade/binance/link")
async def api_auto_trade_delete_binance_link(
    request: Request,
):
    _require_cfg_unlock(request)
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    await _auto_delete_binance_link(user_id)
    return {"ok": True, "link": _auto_public_binance_link(None)}


async def _auto_trade_tick_core(
    user_id: int,
    *,
    force_run: bool = False,
    source: str = "api",
) -> Dict[str, Any]:
    async with _auto_get_tick_lock():
        now_ms = int(time() * 1000)
        cfg = await _auto_get_or_create_config(user_id)
        cfg_id = cfg.get("id")
        if not cfg.get("enabled") and not force_run:
            return {"ok": True, "action": "DISABLED", "message": "auto trade is disabled", "config": cfg}

        await _auto_refresh_open_records(user_id, sync_updates=True)
        try:
            rows = await _sb_request(
                "GET",
                "/rest/v1/auto_trade_records",
                params={
                    "select": "id,status,opened_ms",
                    "user_id": f"eq.{user_id}",
                    "status": "eq.OPEN",
                    "order": "opened_ms.desc",
                    "limit": "200",
                },
            )
        except HTTPException as e:
            if _auto_is_missing_table_error(e, "auto_trade_records"):
                raise _auto_table_missing_error("auto_trade_records")
            raise
        open_rows = rows if isinstance(rows, list) else []
        open_count = len(open_rows)
        max_open = int(cfg.get("max_open_positions", 1) or 1)
        if open_count >= max_open:
            await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
            return {"ok": True, "action": "SKIP_OPEN_LIMIT", "open_count": open_count, "max_open_positions": max_open}

        day_pnl = await _auto_today_realized_pnl(user_id)
        max_daily_loss = float(cfg.get("daily_max_loss_usdt", 0.0) or 0.0)
        if max_daily_loss > 0 and day_pnl <= -max_daily_loss:
            await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
            return {
                "ok": True,
                "action": "SKIP_DAILY_LOSS_LIMIT",
                "today_pnl_usdt": day_pnl,
                "daily_max_loss_usdt": max_daily_loss,
            }

        try:
            last_rows = await _sb_request(
                "GET",
                "/rest/v1/auto_trade_records",
                params={
                    "select": "id,opened_ms",
                    "user_id": f"eq.{user_id}",
                    "order": "opened_ms.desc",
                    "limit": "1",
                },
            )
        except HTTPException as e:
            if _auto_is_missing_table_error(e, "auto_trade_records"):
                raise _auto_table_missing_error("auto_trade_records")
            raise
        cooldown_min = int(cfg.get("cooldown_min", 0) or 0)
        if cooldown_min > 0 and isinstance(last_rows, list) and last_rows:
            last_opened = int(last_rows[0].get("opened_ms", 0) or 0)
            if last_opened > 0 and now_ms < (last_opened + cooldown_min * 60 * 1000):
                await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
                return {
                    "ok": True,
                    "action": "SKIP_COOLDOWN",
                    "cooldown_min": cooldown_min,
                    "last_opened_ms": last_opened,
                }

        symbol = str(cfg.get("symbol", "BTCUSDT")).upper()
        market = str(cfg.get("market", "spot")).lower()
        interval = str(cfg.get("interval", "5m"))
        mode = str(cfg.get("mode", "balanced")).lower()
        if mode not in _AUTO_ALLOWED_MODES:
            mode = "balanced"
        order_size_usdt = float(cfg.get("order_size_usdt", 120.0) or 120.0)
        try:
            available = await _auto_market_available_usdt(user_id, market)
        except HTTPException as e:
            await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
            return {
                "ok": True,
                "action": "SKIP_COLLATERAL_ERROR",
                "detail": str(getattr(e, "detail", "") or "collateral fetch failed"),
                "market": market,
            }
        if available + 1e-9 < order_size_usdt:
            await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
            return {
                "ok": True,
                "action": "SKIP_COLLATERAL_LOW",
                "market": market,
                "available_usdt": round(available, 6),
                "required_usdt": round(order_size_usdt, 6),
            }

        now_ts = time()
        klines = await app.state.binance.klines(symbol=symbol, interval=interval, limit=500, now_ts=now_ts, market=market)
        df = _klines_to_df(list(klines))
        ind = compute_indicators(df)
        if ind is None:
            raise HTTPException(status_code=400, detail="not enough data for auto trade indicators")
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
        buy_prob, _, _ = _apply_calibration(raw_buy)
        buy_pct = round(buy_prob * 100.0, 2)
        sell_pct = round((1.0 - buy_prob) * 100.0, 2)
        confidence = float(score.confidence)
        swing = _latest_swing(df, lookback=200)
        swing_up: Optional[bool] = None
        if swing is not None:
            swing_up = bool(int(swing.get("is_up", 0)))
        decision = _auto_decide_signal(
            buy_pct=buy_pct,
            sell_pct=sell_pct,
            confidence=confidence,
            regime=regime.regime,
            symbol=symbol,
            market=market,
            mode=mode,
            swing_is_up=swing_up,
        )
        signal_info = {
            "buy_pct": buy_pct,
            "sell_pct": sell_pct,
            "confidence": confidence,
            "regime": regime.regime,
            "side": decision["side"],
            "diff": decision["diff"],
        }

        if swing is None or swing_up is None:
            await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
            return {"ok": True, "action": "NO_SIGNAL", "reason": "NO_SWING", "signal": signal_info}

        atr14 = float(ind.atr14 or 0.0)

        def _auto_plan_payload_for_side(side_v: str) -> Dict[str, Any] | None:
            side_u = str(side_v or "").upper()
            if side_u not in {"BUY", "SELL"}:
                return None
            p = _auto_build_fib_trade_plan(
                side=side_u,
                swing=swing,
                close=close,
                atr14=atr14,
                mode=mode,
            )
            if not isinstance(p, dict):
                return None
            return {
                "side": side_u,
                "entry_price": round(float(p["entry_price"]), 8),
                "stop_price": round(float(p["stop_price"]), 8),
                "tp1_price": round(float(p["tp1_price"]), 8),
                "tp2_price": round(float(p["tp2_price"]), 8),
            }

        def _auto_pick_preview_side(*cands: Any) -> str:
            for c in cands:
                s = str(c or "").upper()
                if s not in {"BUY", "SELL"}:
                    continue
                if market == "spot" and s == "SELL":
                    continue
                return s
            fallback = "BUY" if float(buy_pct) >= float(sell_pct) else "SELL"
            if market == "spot" and fallback == "SELL":
                fallback = "BUY"
            return fallback

        trade_side = "WAIT"
        if mode == "aggressive":
            trade_side = _auto_pick_aggressive_side(buy_pct=buy_pct, sell_pct=sell_pct, market=market)
            if market == "spot" and trade_side == "SELL":
                trade_side = "WAIT"
            if trade_side == "WAIT":
                preview_plan = _auto_plan_payload_for_side(
                    _auto_pick_preview_side(decision.get("side", ""), "BUY" if bool(swing_up) else "SELL")
                )
                await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
                out = {"ok": True, "action": "NO_SIGNAL", "reason": "AGGRESSIVE_THRESHOLD", "signal": signal_info}
                if isinstance(preview_plan, dict):
                    out["plan"] = preview_plan
                return out
            if not _auto_side_matches_swing(trade_side, swing_up):
                preview_plan = _auto_plan_payload_for_side(_auto_pick_preview_side("BUY" if bool(swing_up) else "SELL"))
                await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
                out = {"ok": True, "action": "NO_SIGNAL", "reason": "SWING_MISMATCH", "signal": signal_info}
                if isinstance(preview_plan, dict):
                    out["plan"] = preview_plan
                return out
        else:
            trade_side = str(decision.get("side", "WAIT")).upper()
            pass_ready = bool(decision.get("pass_prob")) and bool(decision.get("pass_regime"))
            if market == "spot" and trade_side == "SELL":
                trade_side = "WAIT"
            if trade_side not in {"BUY", "SELL"} or (not pass_ready):
                preview_plan = _auto_plan_payload_for_side(
                    _auto_pick_preview_side(trade_side, decision.get("side", ""), "BUY" if bool(swing_up) else "SELL")
                )
                await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
                out = {"ok": True, "action": "NO_SIGNAL", "reason": "BASIC_PASS_FAIL", "signal": signal_info}
                if isinstance(preview_plan, dict):
                    out["plan"] = preview_plan
                return out

        plan = _auto_build_fib_trade_plan(
            side=trade_side,
            swing=swing,
            close=close,
            atr14=atr14,
            mode=mode,
        )
        if not isinstance(plan, dict):
            await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
            return {"ok": True, "action": "NO_SIGNAL", "reason": "INVALID_PLAN", "signal": signal_info}

        last_high = float(df["high"].iloc[-1])
        last_low = float(df["low"].iloc[-1])
        entry_price = float(plan["entry_price"])
        entry_touched = _auto_entry_touched(entry_price=entry_price, high=last_high, low=last_low, close=close)
        if not entry_touched:
            await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
            return {
                "ok": True,
                "action": "WAIT_FIB_ENTRY",
                "signal": signal_info,
                "plan": {
                    "side": trade_side,
                    "entry_price": round(float(plan["entry_price"]), 8),
                    "stop_price": round(float(plan["stop_price"]), 8),
                    "tp1_price": round(float(plan["tp1_price"]), 8),
                    "tp2_price": round(float(plan["tp2_price"]), 8),
                },
            }

        tp_cfg_pct = float(cfg.get("take_profit_pct", 0.0) or 0.0) / 100.0
        sl_cfg_pct = float(cfg.get("stop_loss_pct", 0.0) or 0.0) / 100.0
        qty = order_size_usdt / max(1e-12, entry_price)
        entry_exec = entry_price
        qty_exec = qty
        entry_exec_info: Dict[str, Any] = {}
        api_key, api_secret = await _auto_get_binance_credentials(user_id)
        entry_client_id = _auto_client_id(
            "ent",
            user_id,
            symbol,
            market,
            interval,
            mode,
            int(df["open_time_ms"].iloc[-1]),
            round(entry_price, 6),
        )
        if _AUTO_LIVE_TRADING_ENABLED:
            try:
                fill = await _auto_place_market_order(
                    market=market,
                    symbol=symbol,
                    side=trade_side,
                    qty=qty,
                    api_key=api_key,
                    api_secret=api_secret,
                    reduce_only=False,
                    fallback_price=entry_price,
                    client_order_id=entry_client_id,
                )
                qty_exec = float(fill.get("executed_qty", 0.0) or 0.0)
                if market == "spot" and trade_side == "BUY":
                    qty_exec = max(0.0, qty_exec - float(fill.get("base_fee_qty", 0.0) or 0.0))
                entry_exec = float(fill.get("avg_price", 0.0) or entry_price)
                entry_exec_info = {
                    "order_id": str(fill.get("order_id", "") or ""),
                    "qty": round(qty_exec, 8),
                    "avg_price": round(entry_exec, 8),
                    "ts_ms": now_ms,
                    "source": str(source),
                    "client_order_id": entry_client_id,
                }
                if qty_exec <= 0 or entry_exec <= 0:
                    await _auto_audit_log(
                        user_id=user_id,
                        event="ENTRY_EMPTY_FILL",
                        level="ERROR",
                        symbol=symbol,
                        market=market,
                        mode=mode,
                        side=trade_side,
                        status="OPEN",
                        qty=qty,
                        detail="entry order not filled",
                        order_id=str(fill.get("order_id", "") or ""),
                        client_order_id=entry_client_id,
                        payload={"source": source},
                    )
                    raise HTTPException(status_code=400, detail="entry order not filled")
                await _auto_audit_log(
                    user_id=user_id,
                    event="ENTRY_FILLED",
                    symbol=symbol,
                    market=market,
                    mode=mode,
                    side=trade_side,
                    status="OPEN",
                    qty=qty_exec,
                    price=entry_exec,
                    order_id=str(fill.get("order_id", "") or ""),
                    client_order_id=entry_client_id,
                    payload={"requested_qty": round(qty, 8), "source": source},
                )
            except HTTPException as e:
                await _auto_audit_log(
                    user_id=user_id,
                    event="ENTRY_REJECTED",
                    level="ERROR",
                    symbol=symbol,
                    market=market,
                    mode=mode,
                    side=trade_side,
                    status="OPEN",
                    qty=qty,
                    price=entry_price,
                    detail=str(getattr(e, "detail", "") or "order rejected"),
                    client_order_id=entry_client_id,
                    payload={"source": source},
                )
                await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
                return {
                    "ok": True,
                    "action": "ORDER_REJECTED",
                    "detail": str(getattr(e, "detail", "") or "order rejected"),
                    "signal": signal_info,
                }

        if tp_cfg_pct > 0:
            if trade_side == "SELL":
                tp_manual = entry_exec * (1.0 - tp_cfg_pct)
            else:
                tp_manual = entry_exec * (1.0 + tp_cfg_pct)
            tp1_price = float(tp_manual)
            tp2_price = float(tp_manual)
        else:
            tp1_price = float(plan["tp1_price"])
            tp2_price = float(plan["tp2_price"])
        if sl_cfg_pct > 0:
            if trade_side == "SELL":
                stop_loss_price = float(entry_exec * (1.0 + sl_cfg_pct))
            else:
                stop_loss_price = float(entry_exec * (1.0 - sl_cfg_pct))
        else:
            stop_loss_price = float(plan["stop_price"])

        reason_state = {
            "v": 3,
            "mode": mode,
            "side": trade_side,
            "tp1_price": round(tp1_price, 8),
            "tp2_price": round(tp2_price, 8),
            "tp1_hit": False,
            "init_qty": round(qty_exec, 8),
            "remaining_qty": round(qty_exec, 8),
            "realized_pnl_usdt": 0.0,
            "hold_max_ms": _AUTO_HOLD_MAX_MS,
            "buy_pct": round(buy_pct, 4),
            "sell_pct": round(sell_pct, 4),
            "confidence": round(confidence, 6),
            "regime": str(regime.regime),
            "decision_side": str(decision.get("side", "")),
            "decision_diff": round(float(decision.get("diff", 0.0)), 6),
            "tp_mode": "manual_pct" if tp_cfg_pct > 0 else "fib_auto",
            "tp_input_pct": round(tp_cfg_pct * 100.0, 6),
            "sl_mode": "manual_pct" if sl_cfg_pct > 0 else "fib_auto",
            "sl_input_pct": round(sl_cfg_pct * 100.0, 6),
            "swing_is_up": 1 if bool(swing_up) else 0,
            "swing_lo": round(float(swing.get("lo", 0.0) or 0.0), 8),
            "swing_hi": round(float(swing.get("hi", 0.0) or 0.0), 8),
            "entry_exec": entry_exec_info,
            "entry_client_id": entry_client_id,
            "protect_sl_order_id": "",
            "protect_sl_client_id": "",
            "protect_sl_type": "",
            "last_exec_error": "",
        }
        create_body = {
            "user_id": user_id,
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "mode": mode,
            "side": trade_side,
            "status": "OPEN",
            "entry_price": round(entry_exec, 8),
            "take_profit_price": round(tp2_price, 8),
            "stop_loss_price": round(stop_loss_price, 8),
            "qty": round(qty_exec, 8),
            "notional_usdt": round(max(0.0, qty_exec * entry_exec), 4),
            "close_price": None,
            "pnl_usdt": 0.0,
            "opened_ms": now_ms,
            "closed_ms": None,
            "signal_buy_pct": buy_pct,
            "signal_sell_pct": sell_pct,
            "signal_confidence": round(confidence, 6),
            "decision_diff": round(float(decision["diff"]), 6),
            "reason": _auto_reason_pack(reason_state),
        }

        protect_info: Dict[str, Any] = {}
        if _AUTO_LIVE_TRADING_ENABLED and qty_exec > 0 and stop_loss_price > 0:
            protect_cid = _auto_client_id("psl", user_id, symbol, market, interval, int(now_ms))
            close_side = "SELL" if trade_side == "BUY" else "BUY"
            try:
                protect_info = await _auto_place_protective_stop_order(
                    market=market,
                    symbol=symbol,
                    side=close_side,
                    stop_price=stop_loss_price,
                    qty=qty_exec,
                    api_key=api_key,
                    api_secret=api_secret,
                    client_order_id=protect_cid,
                )
                reason_state["protect_sl_order_id"] = str(protect_info.get("order_id", "") or "")
                reason_state["protect_sl_client_id"] = str(protect_info.get("client_order_id", "") or "")
                reason_state["protect_sl_type"] = str(protect_info.get("type", "") or "")
                reason_state["last_exec_error"] = ""
                create_body["reason"] = _auto_reason_pack(reason_state)
                await _auto_audit_log(
                    user_id=user_id,
                    event="ENTRY_PROTECT_CREATE_OK",
                    symbol=symbol,
                    market=market,
                    mode=mode,
                    side=close_side,
                    status="OPEN",
                    qty=qty_exec,
                    price=stop_loss_price,
                    order_id=str(protect_info.get("order_id", "") or ""),
                    client_order_id=str(protect_info.get("client_order_id", "") or ""),
                )
            except HTTPException as e:
                fail_detail = str(getattr(e, "detail", "") or "stop order failed")
                await _auto_audit_log(
                    user_id=user_id,
                    event="ENTRY_PROTECT_CREATE_FAIL",
                    level="ERROR",
                    symbol=symbol,
                    market=market,
                    mode=mode,
                    side=close_side,
                    status="OPEN",
                    qty=qty_exec,
                    price=stop_loss_price,
                    detail=fail_detail,
                    client_order_id=protect_cid,
                )
                # 보호주문 실패 시 포지션 노출을 피하기 위해 즉시 시장가 청산 시도
                try:
                    flat_cid = _auto_client_id("eflat", user_id, symbol, market, now_ms)
                    flat_fill = await _auto_place_market_order(
                        market=market,
                        symbol=symbol,
                        side=("SELL" if trade_side == "BUY" else "BUY"),
                        qty=qty_exec,
                        api_key=api_key,
                        api_secret=api_secret,
                        reduce_only=(market == "futures"),
                        fallback_price=entry_exec,
                        client_order_id=flat_cid,
                    )
                    await _auto_audit_log(
                        user_id=user_id,
                        event="ENTRY_EMERGENCY_FLAT_OK",
                        level="WARN",
                        symbol=symbol,
                        market=market,
                        mode=mode,
                        side=("SELL" if trade_side == "BUY" else "BUY"),
                        status="OPEN",
                        qty=float(flat_fill.get("executed_qty", qty_exec) or qty_exec),
                        price=float(flat_fill.get("avg_price", entry_exec) or entry_exec),
                        order_id=str(flat_fill.get("order_id", "") or ""),
                        client_order_id=flat_cid,
                        detail="protective order create failed",
                    )
                except Exception as flat_e:
                    rescue_detail = "protective order failed and emergency flatten failed"
                    reason_state["protect_sl_order_id"] = ""
                    reason_state["protect_sl_client_id"] = protect_cid
                    reason_state["protect_sl_type"] = ""
                    reason_state["last_exec_error"] = f"{rescue_detail}: {type(flat_e).__name__}"
                    reason_state["last_exec_error_ms"] = int(time() * 1000)
                    create_body["reason"] = _auto_reason_pack(reason_state)
                    await _auto_audit_log(
                        user_id=user_id,
                        event="ENTRY_EMERGENCY_FLAT_FAIL",
                        level="ERROR",
                        symbol=symbol,
                        market=market,
                        mode=mode,
                        side=("SELL" if trade_side == "BUY" else "BUY"),
                        status="OPEN",
                        qty=qty_exec,
                        price=entry_exec,
                        detail=f"{rescue_detail}: {type(flat_e).__name__}",
                        client_order_id=protect_cid,
                        payload={"protect_fail_detail": fail_detail},
                    )
                    rescued = await _auto_rescue_open_record(
                        user_id=user_id,
                        create_body=create_body,
                        reason_state=reason_state,
                        event_prefix="ENTRY",
                        symbol=symbol,
                        market=market,
                        mode=mode,
                        side=trade_side,
                        qty=qty_exec,
                        entry_price=entry_exec,
                        client_order_id=entry_client_id,
                        detail="entry protect/flat fail; open record rescued",
                        payload={"protect_fail_detail": fail_detail},
                    )
                    await _auto_update_config(user_id, cfg_id, {**cfg, "enabled": False, "last_run_ms": now_ms})
                    if isinstance(rescued, dict):
                        return {
                            "ok": True,
                            "action": "OPEN_RESCUED",
                            "detail": "보호주문/비상청산 실패로 복구 레코드 저장 후 자동매매를 중단했습니다.",
                            "record": rescued,
                            "signal": signal_info,
                        }
                await _auto_update_config(user_id, cfg_id, {**cfg, "enabled": False, "last_run_ms": now_ms})
                return {
                    "ok": True,
                    "action": "ORDER_REJECTED",
                    "detail": f"보호주문 실패: {str(getattr(e, 'detail', '') or 'stop order failed')}",
                    "signal": signal_info,
                }

        try:
            created = await _sb_request("POST", "/rest/v1/auto_trade_records", json_body=[create_body])
        except HTTPException as e:
            await _auto_audit_log(
                user_id=user_id,
                event="RECORD_SAVE_FAIL",
                level="ERROR",
                symbol=symbol,
                market=market,
                mode=mode,
                side=trade_side,
                status="OPEN",
                qty=qty_exec,
                price=entry_exec,
                detail=str(getattr(e, "detail", "") or "record insert failed"),
                client_order_id=entry_client_id,
                payload={"stop_price": stop_loss_price, "tp2_price": tp2_price},
            )
            if _AUTO_LIVE_TRADING_ENABLED and qty_exec > 0:
                try:
                    if protect_info:
                        await _auto_cancel_order(
                            market=market,
                            symbol=symbol,
                            api_key=api_key,
                            api_secret=api_secret,
                            order_id=str(protect_info.get("order_id", "") or ""),
                            client_order_id=str(protect_info.get("client_order_id", "") or ""),
                        )
                except Exception:
                    pass
                try:
                    # DB 저장 실패 시 미기록 포지션 방지를 위해 즉시 역포지션 청산
                    dbflat_cid = _auto_client_id("dbflat", user_id, symbol, market, now_ms)
                    dbflat_fill = await _auto_place_market_order(
                        market=market,
                        symbol=symbol,
                        side=("SELL" if trade_side == "BUY" else "BUY"),
                        qty=qty_exec,
                        api_key=api_key,
                        api_secret=api_secret,
                        reduce_only=(market == "futures"),
                        fallback_price=entry_exec,
                        client_order_id=dbflat_cid,
                    )
                    await _auto_audit_log(
                        user_id=user_id,
                        event="DB_FAIL_FLAT_OK",
                        level="WARN",
                        symbol=symbol,
                        market=market,
                        mode=mode,
                        side=("SELL" if trade_side == "BUY" else "BUY"),
                        status="OPEN",
                        qty=float(dbflat_fill.get("executed_qty", qty_exec) or qty_exec),
                        price=float(dbflat_fill.get("avg_price", entry_exec) or entry_exec),
                        order_id=str(dbflat_fill.get("order_id", "") or ""),
                        client_order_id=dbflat_cid,
                    )
                except Exception as flat_e:
                    reason_state["last_exec_error"] = f"record save failed and flatten failed: {type(flat_e).__name__}"
                    reason_state["last_exec_error_ms"] = int(time() * 1000)
                    create_body["reason"] = _auto_reason_pack(reason_state)
                    await _auto_audit_log(
                        user_id=user_id,
                        event="DB_FAIL_FLAT_FAIL",
                        level="ERROR",
                        symbol=symbol,
                        market=market,
                        mode=mode,
                        side=("SELL" if trade_side == "BUY" else "BUY"),
                        status="OPEN",
                        qty=qty_exec,
                        price=entry_exec,
                        detail=f"record save failed and flatten failed: {type(flat_e).__name__}",
                        client_order_id=entry_client_id,
                        payload={"record_save_detail": str(getattr(e, "detail", "") or "record insert failed")},
                    )
                    rescued = await _auto_rescue_open_record(
                        user_id=user_id,
                        create_body=create_body,
                        reason_state=reason_state,
                        event_prefix="DBFAIL",
                        symbol=symbol,
                        market=market,
                        mode=mode,
                        side=trade_side,
                        qty=qty_exec,
                        entry_price=entry_exec,
                        client_order_id=entry_client_id,
                        detail="record save/flatten fail; open record rescued",
                        payload={"record_save_detail": str(getattr(e, "detail", "") or "record insert failed")},
                    )
                    await _auto_update_config(user_id, cfg_id, {**cfg, "enabled": False, "last_run_ms": now_ms})
                    if isinstance(rescued, dict):
                        return {
                            "ok": True,
                            "action": "OPEN_RESCUED",
                            "detail": "기록저장/비상청산 실패로 복구 레코드 저장 후 자동매매를 중단했습니다.",
                            "record": rescued,
                            "signal": signal_info,
                        }
            await _auto_update_config(user_id, cfg_id, {**cfg, "enabled": False, "last_run_ms": now_ms})
            if _auto_is_missing_table_error(e, "auto_trade_records"):
                raise _auto_table_missing_error("auto_trade_records")
            raise
        await _auto_update_config(user_id, cfg_id, {**cfg, "last_run_ms": now_ms})
        rec = created[0] if isinstance(created, list) and created else create_body
        await _auto_audit_log(
            user_id=user_id,
            event="RECORD_OPEN_SAVED",
            record_id=_auto_int_or_none(rec.get("id")),
            symbol=symbol,
            market=market,
            mode=mode,
            side=trade_side,
            status="OPEN",
            qty=qty_exec,
            price=entry_exec,
            order_id=str(entry_exec_info.get("order_id", "") or ""),
            client_order_id=entry_client_id,
            payload={
                "protect_order_id": str(protect_info.get("order_id", "") or ""),
                "protect_client_order_id": str(protect_info.get("client_order_id", "") or ""),
            },
        )
        return {
            "ok": True,
            "action": "OPENED",
            "record": rec,
            "signal": {
                "buy_pct": buy_pct,
                "sell_pct": sell_pct,
                "confidence": confidence,
                "regime": regime.regime,
                "side": trade_side,
                "diff": decision["diff"],
            },
            "plan": {
                "entry_price": round(entry_exec, 8),
                "stop_price": round(stop_loss_price, 8),
                "tp1_price": round(tp1_price, 8),
                "tp2_price": round(tp2_price, 8),
            },
        }


async def _auto_trade_bg_loop(app_obj: FastAPI) -> None:
    await asyncio.sleep(2.0)
    while True:
        user_id = 0
        try:
            public_user = await _sim_get_or_create_public_user()
            user_id = int(public_user.get("id") or 0)
            await _auto_trade_tick_core(user_id, force_run=False, source="scheduler")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _AUTO_LOG.exception("auto trade scheduler tick failed")
            if user_id > 0:
                try:
                    await _auto_audit_log(
                        user_id=user_id,
                        event="SCHEDULER_TICK_FAIL",
                        level="ERROR",
                        status="SYSTEM",
                        detail=f"{type(e).__name__}: {str(e)[:200]}",
                    )
                except Exception:
                    pass
        await asyncio.sleep(_AUTO_TICK_INTERVAL_S)


@app.post("/api/auto_trade/tick")
async def api_auto_trade_tick(
    request: Request,
    payload: Dict[str, Any] = Body(default={}),
):
    _require_cfg_unlock(request)
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    force_run = bool(payload.get("force", False))
    return await _auto_trade_tick_core(user_id, force_run=force_run, source="api")


@app.get("/api/auto_trade/records")
async def api_auto_trade_records(
    symbol: str = Query("", max_length=20),
    mode: str = Query("", max_length=20),
    status_filter: str = Query("", alias="status", max_length=20),
    limit: int = Query(20, ge=1, le=200),
    page: int = Query(1, ge=1, le=10000),
    sync_updates: bool = Query(False, alias="sync"),
):
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    if sync_updates:
        await _auto_refresh_open_records(user_id, sync_updates=True)
    per_page = max(1, min(int(limit), 200))
    fetch_n = per_page + 1
    offset = (int(page) - 1) * per_page
    params = {
        "select": "id,user_id,symbol,market,interval,mode,side,status,entry_price,take_profit_price,stop_loss_price,qty,notional_usdt,close_price,pnl_usdt,opened_ms,closed_ms,signal_buy_pct,signal_sell_pct,signal_confidence,decision_diff,reason",
        "user_id": f"eq.{user_id}",
        "order": "opened_ms.desc",
        "limit": str(fetch_n),
        "offset": str(offset),
    }
    symbol_u = str(symbol or "").strip().upper()
    if symbol_u:
        params["symbol"] = f"eq.{symbol_u}"
    mode_v = str(mode or "").strip().lower()
    if mode_v:
        params["mode"] = f"eq.{mode_v}"
    status_v = str(status_filter or "").strip().upper()
    if status_v in _AUTO_ALLOWED_RECORD_STATUS:
        params["status"] = f"eq.{status_v}"
    try:
        rows = await _sb_request("GET", "/rest/v1/auto_trade_records", params=params)
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_records"):
            raise _auto_table_missing_error("auto_trade_records")
        raise
    items = rows if isinstance(rows, list) else []
    has_next = len(items) > per_page
    page_rows = items[:per_page]
    return {"ok": True, "records": page_rows, "page": int(page), "limit": per_page, "has_next": has_next}


@app.get("/api/auto_trade/audit")
async def api_auto_trade_audit(
    request: Request,
    event: str = Query("", max_length=80),
    level: str = Query("", max_length=16),
    symbol: str = Query("", max_length=20),
    record_id: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    page: int = Query(1, ge=1, le=10000),
):
    _require_cfg_unlock(request)
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    per_page = max(1, min(int(limit), 200))
    fetch_n = per_page + 1
    offset = (int(page) - 1) * per_page
    params = {
        "select": "id,user_id,record_id,event,level,symbol,market,mode,side,status,qty,price,pnl_usdt,order_id,client_order_id,detail,payload,created_ms",
        "user_id": f"eq.{user_id}",
        "order": "created_ms.desc",
        "limit": str(fetch_n),
        "offset": str(offset),
    }
    event_v = str(event or "").strip().upper()
    if event_v:
        params["event"] = f"eq.{event_v}"
    level_v = str(level or "").strip().upper()
    if level_v:
        params["level"] = f"eq.{level_v}"
    symbol_v = str(symbol or "").strip().upper()
    if symbol_v:
        params["symbol"] = f"eq.{symbol_v}"
    if int(record_id) > 0:
        params["record_id"] = f"eq.{int(record_id)}"
    try:
        rows = await _sb_request("GET", "/rest/v1/auto_trade_exec_audit", params=params)
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_exec_audit"):
            raise _auto_table_missing_error("auto_trade_exec_audit")
        raise
    items = rows if isinstance(rows, list) else []
    has_next = len(items) > per_page
    page_rows = items[:per_page]
    return {"ok": True, "rows": page_rows, "page": int(page), "limit": per_page, "has_next": has_next}


@app.get("/api/auto_trade/stats")
async def api_auto_trade_stats(
    sync_updates: bool = Query(False, alias="sync"),
):
    public_user = await _sim_get_or_create_public_user()
    user_id = int(public_user.get("id"))
    if sync_updates:
        await _auto_refresh_open_records(user_id, sync_updates=True)
    try:
        rows = await _sb_request(
            "GET",
            "/rest/v1/auto_trade_records",
            params={
                "select": "id,symbol,status,pnl_usdt,notional_usdt,opened_ms,closed_ms",
                "user_id": f"eq.{user_id}",
                "order": "opened_ms.desc",
                "limit": "5000",
            },
        )
    except HTTPException as e:
        if _auto_is_missing_table_error(e, "auto_trade_records"):
            raise _auto_table_missing_error("auto_trade_records")
        raise
    records = rows if isinstance(rows, list) else []
    total = len(records)
    open_cnt = 0
    tp_cnt = 0
    sl_cnt = 0
    fail_cnt = 0
    realized_pnl = 0.0
    realized_notional = 0.0
    now_ms = int(time() * 1000)
    day_start_ms = _kst_day_start_ms(now_ms)
    today_pnl = 0.0
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for r in records:
        status = str(r.get("status", "")).upper()
        sym = str(r.get("symbol", "")).upper()
        pnl = float(r.get("pnl_usdt", 0.0) or 0.0)
        notional = float(r.get("notional_usdt", 0.0) or 0.0)
        closed_ms = int(r.get("closed_ms", 0) or 0)
        if status == "OPEN":
            open_cnt += 1
        elif status == "TP":
            tp_cnt += 1
        elif status == "SL":
            sl_cnt += 1
        elif status == "CLOSED_FAIL":
            fail_cnt += 1
        if status in {"TP", "SL", "CLOSED_FAIL"}:
            realized_pnl += pnl
            if math.isfinite(notional) and notional > 0:
                realized_notional += notional
            if closed_ms >= day_start_ms:
                today_pnl += pnl

        if not sym:
            continue
        if sym not in by_symbol:
            by_symbol[sym] = {
                "symbol": sym,
                "total": 0,
                "open": 0,
                "tp": 0,
                "sl": 0,
                "fail": 0,
                "realized_pnl_usdt": 0.0,
                "win_rate": 0.0,
            }
        a = by_symbol[sym]
        a["total"] += 1
        if status == "OPEN":
            a["open"] += 1
        elif status == "TP":
            a["tp"] += 1
        elif status == "SL":
            a["sl"] += 1
        elif status == "CLOSED_FAIL":
            a["fail"] += 1
        if status in {"TP", "SL", "CLOSED_FAIL"}:
            a["realized_pnl_usdt"] += pnl

    by_symbol_items = list(by_symbol.values())
    for a in by_symbol_items:
        done = int(a["tp"]) + int(a["sl"]) + int(a["fail"])
        a["realized_pnl_usdt"] = round(float(a["realized_pnl_usdt"]), 4)
        a["win_rate"] = round((float(a["tp"]) / done) * 100.0, 1) if done > 0 else 0.0
    by_symbol_items.sort(key=lambda x: (int(x["total"]), str(x["symbol"])), reverse=True)
    done_total = tp_cnt + sl_cnt + fail_cnt
    realized_pnl_pct = (realized_pnl / realized_notional) * 100.0 if realized_notional > 0 else 0.0
    return {
        "ok": True,
        "stats": {
            "total": total,
            "open": open_cnt,
            "tp": tp_cnt,
            "sl": sl_cnt,
            "fail": fail_cnt,
            "win_rate": round((float(tp_cnt) / done_total) * 100.0, 1) if done_total > 0 else 0.0,
            "realized_pnl_usdt": round(realized_pnl, 4),
            "realized_basis_notional_usdt": round(realized_notional, 4),
            "realized_pnl_pct": round(realized_pnl_pct, 2),
            "today_realized_pnl_usdt": round(today_pnl, 4),
        },
        "by_symbol": by_symbol_items,
    }


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
    if _auth_enabled():
        sid = websocket.cookies.get(_AUTH_COOKIE_NAME)
        if not _auth_valid_session(sid):
            await websocket.close(code=4401)
            return
    await websocket.accept()
    query = parse_qs(websocket.url.query)
    symbol = str(query.get("symbol", ["BTCUSDT"])[0]).upper()
    market = str(query.get("market", ["spot"])[0]).lower()
    interval = str(query.get("interval", ["5m"])[0])
    mode = str(query.get("mode", ["single"])[0]).lower()
    limit = int(str(query.get("limit", ["500"])[0]))
    last_marker: tuple | None = None
    last_sent_ts = 0.0
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
                marker = await _ws_analysis_marker(symbol=symbol, market=market, mode=mode, interval=interval)
                cache_key = f"{mode}:{symbol}:{market}:{interval}:{limit}"
                now_ts = time()
                marker_changed = marker is None or marker != last_marker
                force_refresh = (now_ts - last_sent_ts) >= _WS_ANALYSIS_FORCE_REFRESH_S
                if not marker_changed and not force_refresh:
                    await asyncio.sleep(_WS_ANALYSIS_POLL_S)
                    continue
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
                if marker is not None:
                    last_marker = marker
                last_sent_ts = now_ts
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
