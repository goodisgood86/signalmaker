from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import httpx

BINANCE_SPOT_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
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
    def __init__(self, *, base_url: str) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=10.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def klines(self, *, symbol: str, interval: str, limit: int) -> List[Kline]:
        symbol = symbol.upper().strip()
        if interval not in ALLOWED_INTERVALS:
            raise ValueError(f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
        if limit < 50 or limit > 1500:
            raise ValueError("limit must be between 50 and 1500")

        # base_url에 따라 spot(/api/v3/klines), futures(/fapi/v1/klines) 분기
        path = "/fapi/v1/klines" if "fapi.binance.com" in str(self._client.base_url) else "/api/v3/klines"
        resp = await self._client.get(path, params={"symbol": symbol, "interval": interval, "limit": limit})
        resp.raise_for_status()
        raw: List[List[Any]] = resp.json()

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
        self._spot = BinanceClient(base_url=BINANCE_SPOT_BASE_URL)
        self._futures = BinanceClient(base_url=BINANCE_FUTURES_BASE_URL)
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
