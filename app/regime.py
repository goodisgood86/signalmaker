from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    adx14: float
    atr_pct: float
    bb_width_pct: float
    ema_gap_ratio: float


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    atr = _atr(high, low, close, period).replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100.0
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(20.0)


def _percentile_of_last(series: pd.Series, lookback: int = 252) -> float:
    s = series.dropna()
    if len(s) == 0:
        return 0.5
    window = s.iloc[-lookback:] if len(s) > lookback else s
    last = float(window.iloc[-1])
    pct = float((window <= last).mean())
    return max(0.0, min(1.0, pct))


def classify_regime(df: pd.DataFrame) -> RegimeResult:
    if len(df) < 60:
        return RegimeResult(
            regime="RANGE",
            adx14=20.0,
            atr_pct=0.5,
            bb_width_pct=0.5,
            ema_gap_ratio=0.0,
        )

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    atr14 = _atr(high, low, close, 14)
    adx14 = _adx(high, low, close, 14)

    bb_mid = close.rolling(20).mean().replace(0.0, np.nan)
    bb_std = close.rolling(20).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid

    adx_last = float(adx14.dropna().iloc[-1]) if len(adx14.dropna()) else 20.0
    atr_pct = _percentile_of_last(atr14, lookback=252)
    bb_width_pct = _percentile_of_last(bb_width, lookback=252)
    close_last = float(close.iloc[-1]) if float(close.iloc[-1]) != 0.0 else 1.0
    ema_gap_ratio = abs(float(ema20.iloc[-1]) - float(ema50.iloc[-1])) / abs(close_last)

    if atr_pct >= 0.85 or bb_width_pct >= 0.85:
        regime = "HIGH_VOL"
    elif adx_last >= 25.0 and ema_gap_ratio >= 0.003:
        regime = "TREND"
    else:
        regime = "RANGE"

    return RegimeResult(
        regime=regime,
        adx14=round(adx_last, 3),
        atr_pct=round(atr_pct, 3),
        bb_width_pct=round(bb_width_pct, 3),
        ema_gap_ratio=round(ema_gap_ratio, 6),
    )
