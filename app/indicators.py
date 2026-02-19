from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Indicators:
    ema20: float
    ema50: float
    rsi14: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_mid: float
    bb_upper: float
    bb_lower: float
    atr14: float
    stoch_k: float
    stoch_d: float


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    k_smooth: int = 3,
    d_period: int = 3,
):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    k = 100 * (close - lowest_low) / denom
    k = k.fillna(50.0)
    k_s = k.rolling(k_smooth).mean().fillna(k)
    d = k_s.rolling(d_period).mean().fillna(k_s)
    return k_s, d


def compute_indicators(df: pd.DataFrame) -> Optional[Indicators]:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None
    if len(df) < 60:
        return None

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    rsi14 = _rsi(close, 14)
    macd_line, macd_signal, macd_hist = _macd(close)
    bb_mid, bb_upper, bb_lower = _bollinger(close, 20, 2.0)
    atr14 = _atr(high, low, close, 14)
    stoch_k, stoch_d = _stoch(high, low, close, 14, 3, 3)

    last = df.index[-1]
    def _last_val(s: pd.Series) -> float:
        v = float(s.loc[last])
        if np.isnan(v) or np.isinf(v):
            return float(s.dropna().iloc[-1])
        return v

    return Indicators(
        ema20=_last_val(ema20),
        ema50=_last_val(ema50),
        rsi14=_last_val(rsi14),
        macd=_last_val(macd_line),
        macd_signal=_last_val(macd_signal),
        macd_hist=_last_val(macd_hist),
        bb_mid=_last_val(bb_mid.bfill().ffill()),
        bb_upper=_last_val(bb_upper.bfill().ffill()),
        bb_lower=_last_val(bb_lower.bfill().ffill()),
        atr14=_last_val(atr14),
        stoch_k=_last_val(stoch_k),
        stoch_d=_last_val(stoch_d),
    )
