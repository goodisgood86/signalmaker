from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def find_pivots(df: pd.DataFrame, *, left: int = 3, right: int = 3) -> Dict[str, List[int]]:
    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)
    n = len(df)

    pivot_highs: List[int] = []
    pivot_lows: List[int] = []

    if n < (left + right + 1):
        return {"pivot_highs": pivot_highs, "pivot_lows": pivot_lows}

    for i in range(left, n - right):
        h = float(high.iloc[i])
        l = float(low.iloc[i])

        hi_window = high.iloc[i - left : i + right + 1]
        lo_window = low.iloc[i - left : i + right + 1]

        if h == float(hi_window.max()):
            pivot_highs.append(i)
        if l == float(lo_window.min()):
            pivot_lows.append(i)

    return {"pivot_highs": pivot_highs, "pivot_lows": pivot_lows}


def select_avwap_anchors(df: pd.DataFrame, pivots: Dict[str, List[int]]) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []

    lows = pivots.get("pivot_lows", [])
    highs = pivots.get("pivot_highs", [])

    if lows:
        out.append({"kind": "pivot_low", "index": int(lows[-1])})
    if highs:
        out.append({"kind": "pivot_high", "index": int(highs[-1])})

    return out


def compute_avwap(df: pd.DataFrame, *, anchor_index: int) -> Optional[float]:
    if len(df) == 0 or anchor_index < 0 or anchor_index >= len(df):
        return None

    seg = df.iloc[anchor_index:].copy()
    if len(seg) == 0:
        return None

    high = seg["high"].astype(float)
    low = seg["low"].astype(float)
    close = seg["close"].astype(float)
    volume = seg["volume"].astype(float)

    tp = (high + low + close) / 3.0
    pv = (tp * volume).cumsum()
    vv = volume.cumsum().replace(0.0, pd.NA)
    avwap = pv / vv
    s = avwap.dropna()
    if len(s) == 0:
        return None
    return float(s.iloc[-1])

