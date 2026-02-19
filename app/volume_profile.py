from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def compute_volume_profile(
    df: pd.DataFrame,
    *,
    lookback: int = 300,
    bins: int = 48,
) -> Dict[str, object]:
    if len(df) == 0:
        return {"lookback": 0, "bins": bins, "poc": None, "hvn": [], "lvn": [], "histogram": []}

    seg = df.iloc[-lookback:].copy()
    close = seg["close"].astype(float)
    volume = seg["volume"].astype(float)

    if len(close) < 10:
        return {"lookback": int(len(seg)), "bins": bins, "poc": None, "hvn": [], "lvn": [], "histogram": []}

    p_min = float(close.min())
    p_max = float(close.max())
    if p_max <= p_min:
        p = round(p_min, 6)
        return {
            "lookback": int(len(seg)),
            "bins": bins,
            "poc": p,
            "hvn": [p],
            "lvn": [],
            "histogram": [{"price": p, "volume": float(volume.sum())}],
        }

    edges = np.linspace(p_min, p_max, bins + 1)
    hist = np.zeros(bins, dtype=float)
    idx = np.digitize(close.values, edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    for i, v in zip(idx, volume.values):
        hist[int(i)] += float(v)

    centers = (edges[:-1] + edges[1:]) / 2.0
    max_i = int(np.argmax(hist))
    poc = round(float(centers[max_i]), 6)

    non_zero = hist[hist > 0]
    if len(non_zero) == 0:
        hvn_threshold = 0.0
        lvn_threshold = 0.0
    else:
        hvn_threshold = float(np.quantile(non_zero, 0.8))
        lvn_threshold = float(np.quantile(non_zero, 0.2))

    hvn_idx = np.where(hist >= hvn_threshold)[0]
    lvn_idx = np.where((hist > 0) & (hist <= lvn_threshold))[0]

    hvn_prices: List[float] = [round(float(centers[i]), 6) for i in hvn_idx[:3]]
    lvn_prices: List[float] = [round(float(centers[i]), 6) for i in lvn_idx[:3]]

    histogram = [
        {"price": round(float(centers[i]), 6), "volume": round(float(hist[i]), 6)}
        for i in range(bins)
        if hist[i] > 0
    ]

    return {
        "lookback": int(len(seg)),
        "bins": int(bins),
        "poc": poc,
        "hvn": hvn_prices,
        "lvn": lvn_prices,
        "histogram": histogram,
    }

