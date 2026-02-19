from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Tuple

import httpx

BINANCE_SPOT_BASE_URL = str(os.getenv("BINANCE_SPOT_BASE_URL", "https://api.binance.com")).strip().rstrip("/")
BINANCE_FUTURES_BASE_URL = str(os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")).strip().rstrip("/")
BINANCE_SPOT_FALLBACK_BASE_URL = str(os.getenv("BINANCE_SPOT_FALLBACK_BASE_URL", "https://api.binance.us")).strip().rstrip("/")
BINANCE_FUTURES_FALLBACK_BASE_URLS = [
    x.strip().rstrip("/")
    for x in str(
        os.getenv(
            "BINANCE_FUTURES_FALLBACK_BASE_URLS",
            "https://fapi1.binance.com,https://fapi2.binance.com,https://fapi3.binance.com",
        )
    ).split(",")
    if x.strip()
]
ALLOWED_INTERVALS = {"5m", "1h", "4h"}
ALLOWED_MARKETS = {"spot", "futures"}


@dataclass(frozen=True)
class Kline:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int

    @property
    def open_time_iso(self) -> str:
        return (
            datetime.fromtimestamp(self.open_time_ms / 1000, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )


class BinanceClient:
    def __init__(self, *, base_url: str, market: str, fallback_base_urls: List[str] | None = None) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=10.0)
        self._market = market
        self._fallback_base_urls = [u for u in (fallback_base_urls or []) if u and u != str(self._client.base_url).rstrip("/")]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def klines(self, *, symbol: str, interval: str, limit: int) -> List[Kline]:
        symbol = symbol.upper().strip()
        if interval not in ALLOWED_INTERVALS:
            raise ValueError(f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
        if limit < 50 or limit > 1500:
            raise ValueError("limit must be between 50 and 1500")

        path = "/fapi/v1/klines" if self._market == "futures" else "/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        resp = await self._client.get(path, params=params)
        raw: List[List[Any]] | None = None
        if resp.status_code < 400:
            raw = resp.json()
        else:
            retryable = resp.status_code in {418, 429, 451, 403}
            if retryable and self._fallback_base_urls:
                for fb_url in self._fallback_base_urls:
                    try:
                        async with httpx.AsyncClient(base_url=fb_url, timeout=10.0) as fb:
                            fb_resp = await fb.get(path, params=params)
                        if fb_resp.status_code < 400:
                            raw = fb_resp.json()
                            break
                    except Exception:
                        continue
            if raw is None:
                resp.raise_for_status()
                raw = []

        out: List[Kline] = []
        for row in raw:
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


class _TTLCache:
    def __init__(self, ttl_seconds: float = 10.0) -> None:
        self._ttl = ttl_seconds
        self._store: Dict[Tuple[str, str, str, int], Tuple[float, List[Kline]]] = {}

    def get(self, key: Tuple[str, str, str, int], now_ts: float) -> List[Kline] | None:
        item = self._store.get(key)
        if item is None:
            return None
        ts, value = item
        if now_ts - ts > self._ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: Tuple[str, str, str, int], now_ts: float, value: List[Kline]) -> None:
        self._store[key] = (now_ts, value)


class CachedBinanceClient:
    def __init__(self, *, ttl_seconds: float = 10.0) -> None:
        spot_fallbacks: List[str] = [BINANCE_SPOT_FALLBACK_BASE_URL] if BINANCE_SPOT_FALLBACK_BASE_URL else []
        futures_fallbacks: List[str] = BINANCE_FUTURES_FALLBACK_BASE_URLS[:]
        self._spot = BinanceClient(base_url=BINANCE_SPOT_BASE_URL, market="spot", fallback_base_urls=spot_fallbacks)
        self._futures = BinanceClient(base_url=BINANCE_FUTURES_BASE_URL, market="futures", fallback_base_urls=futures_fallbacks)
        self._cache = _TTLCache(ttl_seconds=ttl_seconds)

    async def aclose(self) -> None:
        await self._spot.aclose()
        await self._futures.aclose()

    async def klines(self, *, symbol: str, interval: str, limit: int, now_ts: float, market: str = "spot") -> List[Kline]:
        market = market.lower().strip()
        if market not in ALLOWED_MARKETS:
            raise ValueError(f"market must be one of {sorted(ALLOWED_MARKETS)}")
        key = (market, symbol.upper().strip(), interval, int(limit))
        cached = self._cache.get(key, now_ts)
        if cached is not None:
            return cached
        client = self._futures if market == "futures" else self._spot
        value = await client.klines(symbol=symbol, interval=interval, limit=limit)
        self._cache.set(key, now_ts, value)
        return value
