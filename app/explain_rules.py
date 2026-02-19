from __future__ import annotations

from typing import Any, Dict, List


def _pct_band(pct: float, side: str) -> Dict[str, str]:
    v = float(pct)
    if v >= 70:
        return {
            "label": "높음",
            "meaning": f"{side} 우위 시나리오를 강하게 보는 구간",
            "action_hint": "단, 단일 신호가 아니라 레짐/레벨과 함께 확인",
        }
    if v >= 55:
        return {
            "label": "약우위",
            "meaning": f"{side} 쪽으로 기울었지만 확신은 중간",
            "action_hint": "분할 접근 또는 추가 확인 신호 대기",
        }
    if v >= 45:
        return {
            "label": "중립",
            "meaning": "방향성 우위가 약한 균형 구간",
            "action_hint": "무리한 진입보다 관망/조건 충족 대기",
        }
    if v >= 30:
        return {
            "label": "약열세",
            "meaning": f"{side} 확률이 낮아진 구간",
            "action_hint": "반대 시나리오 우선 점검",
        }
    return {
        "label": "낮음",
        "meaning": f"{side} 시나리오의 우선순위가 낮은 구간",
        "action_hint": "역방향 근거가 충분한지 우선 확인",
    }


def _confidence_band(confidence: float) -> Dict[str, str]:
    v = float(confidence)
    if v >= 0.75:
        return {"label": "높음", "meaning": "신호 합의도가 높아 방향 판단이 비교적 명확"}
    if v >= 0.55:
        return {"label": "보통", "meaning": "신호는 있으나 충돌 요소도 존재"}
    if v >= 0.35:
        return {"label": "낮음", "meaning": "신호 합의가 약해 흔들릴 가능성 존재"}
    return {"label": "매우 낮음", "meaning": "우위 신호가 거의 없어 노이즈 구간 가능성 큼"}


def _regime_text(regime: str) -> Dict[str, str]:
    r = str(regime or "RANGE").upper()
    if r == "TREND":
        return {
            "name": "TREND",
            "summary": "추세장: 한 방향으로 흐름이 이어지기 쉬운 구간",
            "how_to_read": "추세추종(EMA/MACD/돌파) 신호 우선, 역추세는 보수적으로",
        }
    if r == "HIGH_VOL":
        return {
            "name": "HIGH_VOL",
            "summary": "고변동장: 급등락 폭이 커 손익 편차가 큰 구간",
            "how_to_read": "신호 강도 대비 포지션 크기 축소, 손절/리스크 관리 우선",
        }
    return {
        "name": "RANGE",
        "summary": "횡보장: 박스권 왕복이 잦은 구간",
        "how_to_read": "평균회귀(RSI/Stoch/밴드) 신호 우선, 추세추종은 선별",
    }


def _indicator_rows(
    *,
    close: float,
    indicators: Dict[str, Any],
    levels: Dict[str, Any],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    ema20 = float(indicators.get("ema20", 0.0))
    ema50 = float(indicators.get("ema50", 0.0))
    ema_state = "강세" if ema20 > ema50 else "약세"
    rows.append(
        {
            "name": "EMA20/EMA50",
            "state": ema_state,
            "value": f"EMA20={ema20:.2f}, EMA50={ema50:.2f}",
            "favor": "매수 우위" if ema_state == "강세" else "매도 우위",
            "meaning": "단기 평균이 장기 평균 위면 상승 추세, 아래면 하락 추세",
        }
    )

    rsi = float(indicators.get("rsi14", 50.0))
    if rsi >= 70:
        rsi_state, rsi_favor = "과매수", "매도 우위"
    elif rsi <= 30:
        rsi_state, rsi_favor = "과매도", "매수 우위"
    else:
        rsi_state, rsi_favor = "중립", "중립"
    rows.append(
        {
            "name": "RSI14",
            "state": rsi_state,
            "value": f"RSI={rsi:.1f}",
            "favor": rsi_favor,
            "meaning": "과열/침체 강도를 보는 오실레이터",
        }
    )

    macd_hist = float(indicators.get("macd_hist", 0.0))
    rows.append(
        {
            "name": "MACD 히스토그램",
            "state": "상승 모멘텀" if macd_hist > 0 else "하락 모멘텀",
            "value": f"hist={macd_hist:.4f}",
            "favor": "매수 우위" if macd_hist > 0 else "매도 우위",
            "meaning": "0선 위/아래로 모멘텀 방향을 판단",
        }
    )

    bb_u = float(indicators.get("bb_upper", close))
    bb_l = float(indicators.get("bb_lower", close))
    bb_w = max(bb_u - bb_l, 1e-12)
    bb_pos = (close - bb_l) / bb_w
    if bb_pos > 0.85:
        bb_state, bb_favor = "상단 근접", "매도 우위"
    elif bb_pos < 0.15:
        bb_state, bb_favor = "하단 근접", "매수 우위"
    else:
        bb_state, bb_favor = "중앙권", "중립"
    rows.append(
        {
            "name": "볼린저 밴드",
            "state": bb_state,
            "value": f"밴드 위치={bb_pos:.2f}",
            "favor": bb_favor,
            "meaning": "평균 대비 과열/이격 상태 확인",
        }
    )

    stoch_k = float(indicators.get("stoch_k", 50.0))
    if stoch_k >= 80:
        sk_state, sk_favor = "고점권", "매도 우위"
    elif stoch_k <= 20:
        sk_state, sk_favor = "저점권", "매수 우위"
    else:
        sk_state, sk_favor = "중립", "중립"
    rows.append(
        {
            "name": "Stochastic %K",
            "state": sk_state,
            "value": f"%K={stoch_k:.1f}",
            "favor": sk_favor,
            "meaning": "단기 과매수/과매도 민감 체크",
        }
    )

    avwaps = levels.get("avwap", []) if isinstance(levels, dict) else []
    if close > 0 and isinstance(avwaps, list) and avwaps:
        nearest = min(avwaps, key=lambda x: abs(float(x.get("value", close)) - close))
        av = float(nearest.get("value", close))
        rows.append(
            {
                "name": "AVWAP(근접선)",
                "state": "상단 유지" if close >= av else "하단 위치",
                "value": f"가격={close:.2f}, AVWAP={av:.2f}",
                "favor": "매수 우위" if close >= av else "매도 우위",
                "meaning": "거래량 가중 평균가 위/아래로 수급 우위 판단",
            }
        )

    vp = levels.get("volume_profile") if isinstance(levels, dict) else None
    poc = vp.get("poc") if isinstance(vp, dict) else None
    if poc is not None and close > 0:
        poc_v = float(poc)
        rows.append(
            {
                "name": "VP(POC)",
                "state": "POC 위" if close >= poc_v else "POC 아래",
                "value": f"가격={close:.2f}, POC={poc_v:.2f}",
                "favor": "매수 우위" if close >= poc_v else "매도 우위",
                "meaning": "거래가 가장 많이 쌓인 기준 가격대와의 상대 위치",
            }
        )

    return rows


def build_single_explain(
    *,
    buy_pct: float,
    sell_pct: float,
    confidence: float,
    regime: str,
    close: float,
    indicators: Dict[str, Any],
    levels: Dict[str, Any],
    regime_features: Dict[str, Any],
    feature_contrib: Dict[str, Any] | None,
    feature_raw: Dict[str, Any] | None,
    feature_weights: Dict[str, Any] | None,
    feature_detail: Dict[str, Any] | None,
    score_x: float | None,
) -> Dict[str, Any]:
    buy_band = _pct_band(buy_pct, "매수")
    sell_band = _pct_band(sell_pct, "매도")
    conf_band = _confidence_band(confidence)
    regime_info = _regime_text(regime)
    indicator_rows = _indicator_rows(close=close, indicators=indicators, levels=levels)

    diff = float(buy_pct) - float(sell_pct)
    if diff >= 7:
        favor = "매수 우위"
    elif diff <= -7:
        favor = "매도 우위"
    else:
        favor = "중립/관망"

    contrib = feature_contrib or {}
    raw = feature_raw or {}
    weights = feature_weights or {}
    detail = feature_detail or {}
    top_driver = sorted(contrib.items(), key=lambda x: abs(float(x[1])), reverse=True)
    top_driver_txt = ", ".join([f"{k}:{float(v):+.2f}" for k, v in top_driver[:2]]) if top_driver else "없음"

    breakdown: List[Dict[str, Any]] = []
    for key, label in (
        ("trend", "추세"),
        ("mean_reversion", "평균회귀"),
        ("impulse", "단기모멘텀"),
        ("levels", "레벨(AVWAP/POC)"),
    ):
        r = raw.get(key)
        w = weights.get(key)
        c = contrib.get(key)
        if r is None and w is None and c is None:
            continue
        detail_map = detail.get(key, {}) if isinstance(detail.get(key, {}), dict) else {}
        if key == "trend":
            detail_order = [("ema_cross", "EMA교차"), ("macd_hist", "MACD hist")]
            used_indicators = ["EMA20/EMA50", "MACD histogram"]
        elif key == "mean_reversion":
            detail_order = [("rsi", "RSI"), ("stoch_k", "Stoch K"), ("bollinger_pos", "볼린저 위치")]
            used_indicators = ["RSI14", "Stochastic %K", "Bollinger position"]
        elif key == "impulse":
            detail_order = [("bar_return", "전봉수익률"), ("high_vol_penalty", "고변동 패널티")]
            used_indicators = ["전봉 수익률(close/prev_close-1)", "ATR/가격(고변동 패널티)"]
        else:
            detail_order = [("avwap_pos", "AVWAP 위치"), ("vp_poc_pos", "VP POC 위치")]
            used_indicators = ["AVWAP 근접/상하", "VP POC 근접/상하"]
        detail_lines = []
        for dk, dn in detail_order:
            if dk in detail_map:
                detail_lines.append(f"{dn}:{float(detail_map.get(dk, 0.0)):+.3f}")
        breakdown.append(
            {
                "key": key,
                "label": label,
                "raw": None if r is None else round(float(r), 4),
                "weight": None if w is None else round(float(w), 4),
                "contribution": None if c is None else round(float(c), 4),
                "detail": detail_lines,
                "used_indicators": used_indicators,
            }
        )

    return {
        "probability": {
            "definition": "현재 규칙 기반 신호를 0~100%로 스케일한 상대 우위 점수",
            "buy": {"pct": round(float(buy_pct), 2), **buy_band},
            "sell": {"pct": round(float(sell_pct), 2), **sell_band},
        },
        "calc_easy": [
            "1) trend = EMA(20/50 교차 점수 ±0.7) + MACD hist 부호 점수 ±0.3",
            "2) mean_reversion = RSI(±0.8) + StochK(±0.25) + Bollinger 위치(±0.35)",
            "3) impulse = 전봉수익률 점수(±0.15), HIGH_VOL이면 ATR/가격 조건 패널티(-0.3) 추가 가능",
            "4) levels = AVWAP 위치 점수(±0.2) + VP POC 위치 점수(±0.2)",
            "5) 최종합: x = w_trend*trend + w_mean*mean_reversion + w_impulse*impulse + w_levels*levels",
            "6) 확률변환: buy = sigmoid(x)*100, sell = 100-buy, confidence = |buy-50|/50",
        ],
        "simple_guide": {
            "buy_sell": "매수%/매도%는 '오를 가능성 vs 내릴 가능성'의 상대 점수입니다.",
            "buy_sell_rule": "70%↑ 강한 우위, 55~69% 약한 우위, 45~54% 중립, 44%↓ 열세",
            "confidence": "신뢰도는 '지표들이 같은 방향을 얼마나 같이 가리키는지'입니다.",
            "confidence_rule": "75%↑ 높음, 55~74% 보통, 35~54% 낮음, 34%↓ 매우 낮음",
        },
        "confidence": {
            "value": round(float(confidence), 3),
            **conf_band,
            "why": "매수/매도 격차와 신호 일치도를 반영",
        },
        "regime": {
            **regime_info,
            "features": {
                "adx14": regime_features.get("adx14"),
                "atr_pct": regime_features.get("atr_pct"),
                "bb_width_pct": regime_features.get("bb_width_pct"),
                "ema_gap_ratio": regime_features.get("ema_gap_ratio"),
            },
        },
        "decision": {
            "favor": favor,
            "meaning": f"현재는 {favor} 해석이 우세. 다만 레짐과 레벨(AVWAP/POC) 일치 여부를 함께 확인",
            "drivers": f"주요 기여도: {top_driver_txt}",
        },
        "calc_breakdown": {
            "formula": "x = Σ(raw_i × weight_i), buy=σ(x)×100, sell=100-buy",
            "x": None if score_x is None else round(float(score_x), 6),
            "rows": breakdown,
        },
        "indicators": indicator_rows,
    }


def build_mtf_explain(
    *,
    buy_pct: float,
    sell_pct: float,
    confidence: float,
    tf_weights: Dict[str, float],
    tf_regimes: Dict[str, str],
    tf_buy_prob: Dict[str, float] | None = None,
    filter_shift: float | None = None,
    agreement_shift: float | None = None,
    raw_buy: float | None = None,
) -> Dict[str, Any]:
    buy_band = _pct_band(buy_pct, "매수")
    sell_band = _pct_band(sell_pct, "매도")
    conf_band = _confidence_band(confidence)
    r4h = tf_regimes.get("4h", "RANGE")
    tf_buy_prob = tf_buy_prob or {}

    diff = float(buy_pct) - float(sell_pct)
    if diff >= 7:
        favor = "매수 우위"
    elif diff <= -7:
        favor = "매도 우위"
    else:
        favor = "중립/관망"

    rows: List[Dict[str, Any]] = []
    for tf in ("4h", "1h", "5m"):
        b = float(tf_buy_prob.get(tf, 0.5))
        w = float(tf_weights.get(tf, 0.0))
        rows.append(
            {
                "key": tf,
                "label": f"{tf} 신호",
                "raw": round(b, 4),
                "weight": round(w, 4),
                "contribution": round(b * w, 4),
                "used_indicators": [f"{tf} 내부 모델 결과(buy_raw)"],
                "detail": [f"regime={tf_regimes.get(tf, '-')}"],
            }
        )

    if filter_shift is not None:
        rows.append(
            {
                "key": "4h_filter_shift",
                "label": "4h 방향 보정",
                "raw": round(float(filter_shift), 4),
                "weight": 1.0,
                "contribution": round(float(filter_shift), 4),
                "used_indicators": ["(buy_4h - 0.5) * 0.28"],
                "detail": ["상위 4h 방향을 최종값에 소폭 반영"],
            }
        )
    if agreement_shift is not None:
        rows.append(
            {
                "key": "tf_agreement_shift",
                "label": "TF 합의 보정",
                "raw": round(float(agreement_shift), 4),
                "weight": 1.0,
                "contribution": round(float(agreement_shift), 4),
                "used_indicators": ["((4h-0.5)+(1h-0.5)+(5m-0.5))/3 기반 보정"],
                "detail": ["세 타임프레임이 같은 방향이면 신호를 더 확실히 반영"],
            }
        )

    return {
        "probability": {
            "definition": "4h(필터)+1h(셋업)+5m(트리거) 결합 점수를 확률형으로 표현",
            "buy": {"pct": round(float(buy_pct), 2), **buy_band},
            "sell": {"pct": round(float(sell_pct), 2), **sell_band},
        },
        "calc_easy": [
            "1) 각 TF 내부 계산은 동일: trend(EMA/MACD) + mean_reversion(RSI/Stoch/BB) + impulse(전봉수익률) + levels(AVWAP/POC)",
            "2) 각 TF에서 x 계산 후 buy_raw 산출: buy_raw = sigmoid(x)",
            "3) TF 결합: base = 0.20*4h + 0.35*1h + 0.45*5m",
            "4) 상위 필터 보정: raw = base + (4h-0.5)*0.28",
            "5) TF 합의 보정: raw += clamp(avg(4h/1h/5m 편차) * 0.22, -0.06, +0.06)",
            "6) 확률 보정(calibration): buy = calibrated(raw)*100, sell = 100-buy",
            "7) 신뢰도: TF 방향 합의도가 높을수록 증가",
        ],
        "simple_guide": {
            "buy_sell": "최종 매수%/매도%는 4h·1h·5m를 합쳐 만든 종합 점수입니다.",
            "buy_sell_rule": "70%↑ 강한 우위, 55~69% 약한 우위, 45~54% 중립, 44%↓ 열세",
            "confidence": "신뢰도는 타임프레임들이 같은 방향인지 보여줍니다.",
            "confidence_rule": "75%↑ 높음, 55~74% 보통, 35~54% 낮음, 34%↓ 매우 낮음",
        },
        "confidence": {
            "value": round(float(confidence), 3),
            **conf_band,
            "why": "타임프레임 간 방향 합의도가 높을수록 상승",
        },
        "decision": {
            "favor": favor,
            "meaning": "상위(4h) 방향과 하위(1h/5m) 타이밍 합의 정도를 함께 본 결론",
            "drivers": "4h/1h/5m 가중합 + 4h 필터 보정 + TF 합의 보정",
        },
        "calc_breakdown": {
            "formula": "base=0.20*4h+0.35*1h+0.45*5m, raw=base+(4h-0.5)*0.28+agree_shift",
            "x": None if raw_buy is None else round(float(raw_buy), 6),
            "rows": rows,
        },
        "regime": {
            **_regime_text(r4h),
            "summary": f"상위 4h 레짐({r4h})을 필터로 사용",
        },
        "mtf": {
            "weights": tf_weights,
            "how_to_read": "4h는 방향 필터, 1h는 진입 준비, 5m는 타이밍 트리거",
        },
    }
