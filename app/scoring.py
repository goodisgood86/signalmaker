from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import math

from .indicators import Indicators


@dataclass(frozen=True)
class ScoreResult:
    buy_pct: float
    sell_pct: float
    confidence: float
    reasons: List[str]
    feature_contrib: Optional[Dict[str, float]] = None
    feature_raw: Optional[Dict[str, float]] = None
    feature_weights: Optional[Dict[str, float]] = None
    feature_detail: Optional[Dict[str, Dict[str, float]]] = None
    score_x: Optional[float] = None


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def score_signal(*, close: float, prev_close: float, ind: Indicators) -> ScoreResult:
    reasons: List[str] = []

    x = 0.0

    # Trend bias (EMA)
    if ind.ema20 > ind.ema50:
        x += 0.7
        reasons.append("추세: EMA20 > EMA50 (상승 우위)")
    else:
        x -= 0.7
        reasons.append("추세: EMA20 <= EMA50 (하락/약세 우위)")

    # RSI mean-reversion
    if ind.rsi14 < 30:
        x += 0.9
        reasons.append("RSI14 < 30 (과매도)")
    elif ind.rsi14 > 70:
        x -= 0.9
        reasons.append("RSI14 > 70 (과매수)")
    elif ind.rsi14 < 45:
        x += 0.2
        reasons.append("RSI14 < 45 (약한 과매도 구간)")
    elif ind.rsi14 > 55:
        x -= 0.2
        reasons.append("RSI14 > 55 (약한 과매수 구간)")

    # MACD histogram direction
    if ind.macd_hist > 0:
        x += 0.35
        reasons.append("MACD 히스토그램 > 0 (상승 모멘텀)")
    else:
        x -= 0.35
        reasons.append("MACD 히스토그램 <= 0 (하락 모멘텀)")

    # Bollinger position
    band_width = max(ind.bb_upper - ind.bb_lower, 1e-12)
    pos = (close - ind.bb_lower) / band_width  # 0=lower, 1=upper
    if pos < 0.15:
        x += 0.55
        reasons.append("볼린저: 하단 근접")
    elif pos > 0.85:
        x -= 0.55
        reasons.append("볼린저: 상단 근접")

    # Short-term price impulse (simple)
    if prev_close > 0:
        ret = (close / prev_close) - 1.0
        if ret > 0.004:
            x += 0.15
            reasons.append("단기: 전봉 대비 강세")
        elif ret < -0.004:
            x -= 0.15
            reasons.append("단기: 전봉 대비 약세")

    # 추세+모멘텀 같은 방향 합의 시 신호를 조금 더 확실하게 반영
    if ind.ema20 > ind.ema50 and ind.macd_hist > 0:
        x += 0.25
        reasons.append("합의 보너스: 추세+모멘텀 동시 강세")
    elif ind.ema20 <= ind.ema50 and ind.macd_hist <= 0:
        x -= 0.25
        reasons.append("합의 보너스: 추세+모멘텀 동시 약세")

    x *= 1.18

    buy = _sigmoid(x) * 100.0
    sell = _sigmoid(-x) * 100.0

    # Confidence: far from 50/50 => higher
    confidence = min(1.0, abs(buy - 50.0) / 50.0)

    return ScoreResult(
        buy_pct=round(buy, 2),
        sell_pct=round(sell, 2),
        confidence=round(confidence, 3),
        reasons=reasons,
        feature_contrib=None,
        feature_raw=None,
        feature_weights=None,
        feature_detail=None,
        score_x=round(x, 6),
    )


def score_signal_trendy(
    *,
    close: float,
    prev_close: float,
    ind: Indicators,
    regime: str,
    avwap_levels: List[Dict[str, float]],
    volume_profile: Dict[str, object],
) -> ScoreResult:
    reasons: List[str] = []

    # trend component
    trend = 0.0
    detail_trend: Dict[str, float] = {}
    if ind.ema20 > ind.ema50:
        v = 0.7
        trend += v
        detail_trend["ema_cross"] = v
        reasons.append("추세: EMA20 > EMA50")
    else:
        v = -0.7
        trend += v
        detail_trend["ema_cross"] = v
        reasons.append("추세: EMA20 <= EMA50")
    if ind.macd_hist > 0:
        v = 0.3
        trend += v
        detail_trend["macd_hist"] = v
        reasons.append("모멘텀: MACD hist > 0")
    else:
        v = -0.3
        trend += v
        detail_trend["macd_hist"] = v
        reasons.append("모멘텀: MACD hist <= 0")

    # mean reversion component
    mean_rev = 0.0
    detail_mean: Dict[str, float] = {}
    if ind.rsi14 < 30:
        v = 0.8
        mean_rev += v
        detail_mean["rsi"] = v
        reasons.append("RSI 과매도")
    elif ind.rsi14 > 70:
        v = -0.8
        mean_rev += v
        detail_mean["rsi"] = v
        reasons.append("RSI 과매수")
    else:
        detail_mean["rsi"] = 0.0
    if ind.stoch_k < 20:
        v = 0.25
        mean_rev += v
        detail_mean["stoch_k"] = v
        reasons.append("Stoch 저점권")
    elif ind.stoch_k > 80:
        v = -0.25
        mean_rev += v
        detail_mean["stoch_k"] = v
        reasons.append("Stoch 고점권")
    else:
        detail_mean["stoch_k"] = 0.0
    band_width = max(ind.bb_upper - ind.bb_lower, 1e-12)
    bb_pos = (close - ind.bb_lower) / band_width
    if bb_pos < 0.15:
        v = 0.35
        mean_rev += v
        detail_mean["bollinger_pos"] = v
        reasons.append("볼린저 하단 근접")
    elif bb_pos > 0.85:
        v = -0.35
        mean_rev += v
        detail_mean["bollinger_pos"] = v
        reasons.append("볼린저 상단 근접")
    else:
        detail_mean["bollinger_pos"] = 0.0

    # short impulse component
    impulse = 0.0
    detail_impulse: Dict[str, float] = {"bar_return": 0.0}
    if prev_close > 0:
        ret = (close / prev_close) - 1.0
        if ret > 0.004:
            v = 0.15
            impulse += v
            detail_impulse["bar_return"] = v
            reasons.append("단기 강세")
        elif ret < -0.004:
            v = -0.15
            impulse += v
            detail_impulse["bar_return"] = v
            reasons.append("단기 약세")

    # level component (AVWAP + VP)
    level = 0.0
    detail_level: Dict[str, float] = {"avwap_pos": 0.0, "vp_poc_pos": 0.0}
    if close > 0 and avwap_levels:
        nearest_avwap = min(
            avwap_levels,
            key=lambda x: abs(close - float(x.get("value", close))),
        )
        avwap_value = float(nearest_avwap.get("value", close))
        d = abs(close - avwap_value) / close
        if d <= 0.01:
            if close >= avwap_value:
                v = 0.2
                level += v
                detail_level["avwap_pos"] = v
                reasons.append("레벨: AVWAP 상단 유지")
            else:
                v = -0.2
                level += v
                detail_level["avwap_pos"] = v
                reasons.append("레벨: AVWAP 하단 위치")

    poc = volume_profile.get("poc") if isinstance(volume_profile, dict) else None
    if poc is not None and close > 0:
        poc_v = float(poc)
        d = abs(close - poc_v) / close
        if d <= 0.015:
            if close >= poc_v:
                v = 0.2
                level += v
                detail_level["vp_poc_pos"] = v
                reasons.append("레벨: POC 상단")
            else:
                v = -0.2
                level += v
                detail_level["vp_poc_pos"] = v
                reasons.append("레벨: POC 하단")

    regime = (regime or "RANGE").upper()
    weights = {
        "TREND": {"trend": 1.35, "mean_rev": 0.65, "impulse": 1.0, "level": 0.9},
        "RANGE": {"trend": 0.75, "mean_rev": 1.35, "impulse": 0.8, "level": 1.1},
        "HIGH_VOL": {"trend": 0.8, "mean_rev": 0.7, "impulse": 0.6, "level": 0.8},
    }.get(regime, {"trend": 1.0, "mean_rev": 1.0, "impulse": 1.0, "level": 1.0})

    x = (
        weights["trend"] * trend
        + weights["mean_rev"] * mean_rev
        + weights["impulse"] * impulse
        + weights["level"] * level
    )

    # 구성요소가 같은 방향으로 겹치면 중립으로 머무르지 않도록 방향성 가중
    comps = [trend, mean_rev, impulse, level]
    pos_cnt = sum(1 for v in comps if v > 0.05)
    neg_cnt = sum(1 for v in comps if v < -0.05)
    if pos_cnt >= 3 and neg_cnt == 0:
        x += 0.22
        reasons.append("합의 보너스: 다중 지표 강세 정렬")
    elif neg_cnt >= 3 and pos_cnt == 0:
        x -= 0.22
        reasons.append("합의 보너스: 다중 지표 약세 정렬")

    x *= 1.14 if regime != "HIGH_VOL" else 1.06

    if regime == "HIGH_VOL" and close > 0:
        atr_ratio = ind.atr14 / close
        if atr_ratio > 0.02:
            x -= 0.3
            reasons.append("고변동 패널티")
            detail_impulse["high_vol_penalty"] = -0.3
        else:
            detail_impulse["high_vol_penalty"] = 0.0

    buy = _sigmoid(x) * 100.0
    sell = _sigmoid(-x) * 100.0
    confidence = min(1.0, abs(buy - 50.0) / 50.0)
    if regime == "HIGH_VOL":
        confidence *= 0.9

    feature_contrib = {
        "trend": round(weights["trend"] * trend, 4),
        "mean_reversion": round(weights["mean_rev"] * mean_rev, 4),
        "impulse": round(weights["impulse"] * impulse, 4),
        "levels": round(weights["level"] * level, 4),
    }

    reasons.insert(0, f"레짐: {regime} 가중치 적용")

    return ScoreResult(
        buy_pct=round(buy, 2),
        sell_pct=round(sell, 2),
        confidence=round(confidence, 3),
        reasons=reasons,
        feature_contrib=feature_contrib,
        feature_raw={
            "trend": round(trend, 4),
            "mean_reversion": round(mean_rev, 4),
            "impulse": round(impulse, 4),
            "levels": round(level, 4),
        },
        feature_weights={
            "trend": round(weights["trend"], 4),
            "mean_reversion": round(weights["mean_rev"], 4),
            "impulse": round(weights["impulse"], 4),
            "levels": round(weights["level"], 4),
        },
        feature_detail={
            "trend": {k: round(v, 4) for k, v in detail_trend.items()},
            "mean_reversion": {k: round(v, 4) for k, v in detail_mean.items()},
            "impulse": {k: round(v, 4) for k, v in detail_impulse.items()},
            "levels": {k: round(v, 4) for k, v in detail_level.items()},
        },
        score_x=round(x, 6),
    )
