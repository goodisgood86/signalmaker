# TRENDY_TA_SPEC

## 1. 문서 목적
본 문서는 현물(Spot) 기준 암호화폐 기술적 분석 웹앱의 “트렌디한 분석” 3축을 정의한다.
- 레짐(국면) 분류: `TREND / RANGE / HIGH_VOL`
- 레벨 분석: `Anchored VWAP(AVWAP) + Volume Profile`
- 확률 보정: 백테스트 기반 `Platt` 또는 `Isotonic` calibration

대상 타임프레임은 `5m / 1h / 4h`이며, 최종 출력은 `buy_pct`, `sell_pct`를 확률처럼 해석 가능하게 만드는 것을 목표로 한다.

---

## 2. 현재 구현 상태 점검 (이미 있음 / 없음)

### 2.1 이미 있음
- 데이터
  - `/Users/mark/Desktop/자료/app/coin/app/binance.py`
  - Binance OHLCV 조회 + interval 제한(`5m/1h/4h`) + TTL 캐시
- 지표
  - `/Users/mark/Desktop/자료/app/coin/app/indicators.py`
  - EMA20/EMA50, RSI14, MACD, Bollinger, ATR14, Stochastic
- 스코어링
  - `/Users/mark/Desktop/자료/app/coin/app/scoring.py`
  - 룰 기반 선형 점수 -> sigmoid -> `buy_pct/sell_pct`
- API/UI
  - `/Users/mark/Desktop/자료/app/coin/app/main.py` (`/api/analysis`, `/api/klines`)
  - `/Users/mark/Desktop/자료/app/coin/static/app.js` (캔들 + 기본 분석 표시)

### 2.2 아직 없음
- 레짐 분류기(`TREND/RANGE/HIGH_VOL`) 자체
- AVWAP 계산 및 UI 렌더
- Volume Profile(POC/HVN/LVN) 계산 및 UI 렌더
- 멀티타임프레임 통합 확률 모델(현재는 단일 TF 분석)
- 확률 calibration(Platt/Isotonic) 및 보정 모델 로딩
- 백테스트 파이프라인(look-ahead 방지, 비용 반영, 보정 학습 데이터 생성)

---

## 3. buy_pct / sell_pct의 정의

### 3.1 권장 공식 정의 (Spot 기준)
- `buy_pct`:
  - 시점 `t`에서 롱 진입 시, 앞으로 `N`봉 내에 `+X% (TP)`가 `-Y% (SL)`보다 먼저 도달할 확률
- `sell_pct`:
  - 시점 `t`에서 “매수 회피/현금 유지 관점의 하방 우세 확률”로 정의
  - 즉, 같은 horizon에서 `-Y%`가 `+X%`보다 먼저 도달할 확률

Spot에서는 순수 숏 포지션이 일반적이지 않으므로, `sell_pct`는 “하락 리스크 우세 확률”로 문서/UI에 명시한다.

### 3.2 파라미터 예시
- 5m: `N=24`, `X=1.0%`, `Y=0.7%`
- 1h: `N=24`, `X=2.5%`, `Y=1.8%`
- 4h: `N=20`, `X=5.0%`, `Y=3.5%`

(최종 값은 백테스트 민감도 분석 후 고정)

---

## 4. 레짐 분류 설계

### 4.1 입력 지표
- `ADX(14)`
- `ATR percentile` (예: 최근 252봉 대비 ATR14 분위)
- `Bollinger Bandwidth` (`(upper-lower)/mid`)
- 보조: `EMA20-EMA50` 기울기/거리

### 4.2 임계값 예시
- `HIGH_VOL`
  - `atr_pct >= 0.85` 또는 `bb_width_pct >= 0.85`
- `TREND`
  - `ADX >= 25` AND `abs(ema20-ema50)/close >= 0.003`
- `RANGE`
  - 위 조건 미충족

### 4.3 의사코드
```text
input: adx, atr_pct, bb_width_pct, ema_gap_ratio

if atr_pct >= 0.85 or bb_width_pct >= 0.85:
    regime = HIGH_VOL
else if adx >= 25 and ema_gap_ratio >= 0.003:
    regime = TREND
else:
    regime = RANGE

return regime
```

### 4.4 레짐별 신호 우선순위/가중치

| Feature Group | TREND | RANGE | HIGH_VOL |
|---|---:|---:|---:|
| Trend-following (EMA, MACD>0, BOS) | 0.40 | 0.15 | 0.20 |
| Mean-reversion (RSI/Stoch, BB revert) | 0.10 | 0.40 | 0.10 |
| Level reaction (AVWAP/POC/HVN/LVN/Fib) | 0.20 | 0.25 | 0.20 |
| Volume confirmation (OBV/CMF, VP node) | 0.15 | 0.10 | 0.20 |
| Volatility risk control (ATR penalty) | 0.15 | 0.10 | 0.30 |

가중치는 초기값이며, 백테스트 결과로 교정한다.

---

## 5. AVWAP 설계

### 5.1 앵커 후보 (스윙 피벗 기반)
- 후보 생성: `pivot_high(left,right)`, `pivot_low(left,right)`
- 권장 기본값: `left=3`, `right=3`
- 앵커 선택 우선순위:
  1. 최근 확정 `swing low` (상승 추세 분석용)
  2. 최근 확정 `swing high` (하락 압력 분석용)
  3. 고변동 이벤트 봉(급등락 시작 봉)

### 5.2 계산법
- `typical_price = (high + low + close)/3`
- 앵커 시점 `a` 이후 누적:
  - `AVWAP_t = sum(tp_i * vol_i, i=a..t) / sum(vol_i, i=a..t)`

### 5.3 해석 규칙
- 가격 > AVWAP & AVWAP 기울기 상승: 매수 우위
- 가격이 AVWAP 재돌파 + 거래량 증가: 트리거 신뢰도 상승
- AVWAP 이탈 후 재진입 실패: 리스크 경고

### 5.4 UI 표현 방식
- 캔들 위에 AVWAP 라인 1~2개 표시(최근 swing low anchor, swing high anchor)
- 라벨 예시: `AVWAP(LowAnchor)`, `AVWAP(HighAnchor)`
- 현재가와 AVWAP 거리(%)를 우측 패널에 표시

---

## 6. Volume Profile 설계

### 6.1 가격 binning
- 구간: 최근 `lookback` 봉(예: 1h 차트 300봉)
- bin 개수: 기본 48 (변동성에 따라 32~96 동적 가능)
- 각 봉의 거래량을 가격대 bin으로 분배(단순: 종가 bin 적재, 개선: 고저 범위 비례 분배)

### 6.2 핵심 레벨 도출
- `POC`: 최대 거래량 bin 가격
- `HVN`: 상위 거래량 노드(POC 인접 고밀도 구간)
- `LVN`: 저거래량 노드(가격 통과 속도 빠른 구간)

### 6.3 해석 규칙
- TREND: POC/HVN 재테스트 후 추세 지속 확인
- RANGE: HVN 중심 회귀, LVN 이탈 시 브레이크 후보
- HIGH_VOL: LVN 돌파 시 미끄러짐 확대 주의

### 6.4 UI 표현 방식
- 가격축 우측 미니 히스토그램(가로 막대)
- POC 선 강조(굵은 선), HVN/LVN 점선 오버레이
- 토글: `VP 표시`, `POC/HVN/LVN만 표시`

---

## 7. 확률 보정(Calibration) 설계

### 7.1 목적
현재 `score -> sigmoid` 출력은 “확률처럼 보이는 점수”이므로, 실제 이벤트 빈도와 일치하도록 보정한다.

### 7.2 방식
- 입력: 원시 모델 출력 `p_raw` + 실제 라벨 `y` (TP first 여부)
- 후보:
  - Platt scaling (로지스틱 보정)
  - Isotonic regression (단조 비모수)

### 7.3 학습/적용
1. 백테스트 데이터 분할: Train / Validation / Test (시간순)
2. Validation으로 calibration 모델 학습
3. 추론 시: `p_cal = calibrator(p_raw)`
4. API에는 `buy_pct = 100*p_cal`, `sell_pct = 100*(1-p_cal)` 반환

### 7.4 선택 기준
- 데이터 양 충분 + 비선형 왜곡 크면 Isotonic
- 데이터 적거나 안정성 우선이면 Platt

---

## 8. 과최적화/미래참조 방지 원칙
- look-ahead 금지
  - 시점 `t` 의사결정은 `t`까지 확정 데이터만 사용
  - pivot 확정은 `right`봉 후에만 사용
- 시간순 검증
  - 랜덤 셔플 금지, walk-forward 또는 시계열 split 사용
- 비용 반영
  - 수수료/슬리피지/스프레드 가정 포함
- 파라미터 탐색 제한
  - 지나친 그리드서치 금지, 단순/해석 가능한 파라미터 우선
- 레짐별 샘플 불균형 점검
  - 특정 국면에만 과적합되지 않도록 성능 분해 리포트 필수

---

## 9. API 출력 스키마 제안 (트렌디 분석)
```json
{
  "symbol": "BTCUSDT",
  "spot": true,
  "timeframes": ["4h", "1h", "5m"],
  "asof_open_time_ms": 1735689600000,
  "regime": "TREND",
  "buy_pct": 63.4,
  "sell_pct": 36.6,
  "confidence": 0.71,
  "raw_score": 0.547,
  "calibrated": true,
  "calibration_method": "isotonic",
  "levels": {
    "avwap": [
      {"anchor": "pivot_low", "value": 102345.2},
      {"anchor": "pivot_high", "value": 104120.7}
    ],
    "volume_profile": {
      "lookback": 300,
      "poc": 103200.0,
      "hvn": [102900.0, 103800.0],
      "lvn": [102450.0]
    }
  },
  "reasons": [
    "TREND: ADX 강함 + EMA 갭 유지",
    "가격이 AVWAP 상단 유지",
    "POC 상단 체류"
  ]
}
```
