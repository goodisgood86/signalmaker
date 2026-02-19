# IMPLEMENTATION_TODOS

## 1. 전제
- 범위: 문서 기반 구현 계획 (불필요한 리팩토링 없음)
- 대상: Spot 시장, TF `5m/1h/4h`, 출력 `buy_pct/sell_pct`
- 목표: 레짐 + AVWAP + Volume Profile + Calibration 파이프라인 구축

---

## 2. 현재 구현 대비 갭

### 이미 구현
- `/Users/mark/Desktop/자료/app/coin/app/binance.py`: Binance OHLCV + 캐시
- `/Users/mark/Desktop/자료/app/coin/app/indicators.py`: 기본 지표
- `/Users/mark/Desktop/자료/app/coin/app/scoring.py`: 룰기반 스코어 -> sigmoid
- `/Users/mark/Desktop/자료/app/coin/app/main.py`: `/api/analysis`, `/api/klines`
- `/Users/mark/Desktop/자료/app/coin/static/app.js`: 기본 시각화

### 미구현
- 레짐 분류 함수/모듈
- AVWAP/Volume Profile 계산 모듈
- 멀티TF 결합 엔드포인트
- calibration 학습/적용 로직
- 백테스트 산출물 기반 확률 보정 데이터셋

---

## 3. P0 TODO (핵심 MVP)

### 3.1 레짐 분류기 추가
- 파일
  - 추가: `/Users/mark/Desktop/자료/app/coin/app/regime.py`
  - 수정: `/Users/mark/Desktop/자료/app/coin/app/indicators.py`
- 함수
  - `compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series`
  - `classify_regime(*, adx: float, atr_pct: float, bb_width_pct: float, ema_gap_ratio: float) -> str`
- 입출력 스키마
```json
{
  "input": {
    "adx": 27.1,
    "atr_pct": 0.62,
    "bb_width_pct": 0.58,
    "ema_gap_ratio": 0.0041
  },
  "output": {
    "regime": "TREND"
  }
}
```

### 3.2 AVWAP 계산 모듈 추가
- 파일
  - 추가: `/Users/mark/Desktop/자료/app/coin/app/levels.py`
- 함수
  - `find_pivots(df, left=3, right=3) -> dict`
  - `compute_avwap(df, anchor_index: int) -> pd.Series`
  - `select_avwap_anchors(df, pivots) -> list[int]`
- 입출력 스키마
```json
{
  "input": {
    "ohlcv": "DataFrame",
    "anchor_index": 420
  },
  "output": {
    "avwap_last": 103245.11,
    "anchor_time_ms": 1735680000000
  }
}
```

### 3.3 Volume Profile 계산 모듈 추가
- 파일
  - 추가: `/Users/mark/Desktop/자료/app/coin/app/volume_profile.py`
- 함수
  - `compute_volume_profile(df, lookback=300, bins=48) -> dict`
  - 반환: `poc`, `hvn`, `lvn`, `histogram`
- 입출력 스키마
```json
{
  "input": {
    "lookback": 300,
    "bins": 48
  },
  "output": {
    "poc": 103200.0,
    "hvn": [102900.0, 103800.0],
    "lvn": [102450.0],
    "histogram": [{"price": 103200.0, "volume": 18234.2}]
  }
}
```

### 3.4 멀티TF 분석 엔드포인트 추가
- 파일
  - 수정: `/Users/mark/Desktop/자료/app/coin/app/main.py`
  - 수정: `/Users/mark/Desktop/자료/app/coin/app/scoring.py`
- 엔드포인트
  - `GET /api/analysis_trendy?symbol=BTCUSDT&limit=500`
- 내부 함수
  - `score_signal_trendy(mtf_features: dict) -> ScoreResult`
- 응답 스키마
```json
{
  "symbol": "BTCUSDT",
  "spot": true,
  "regime": "RANGE",
  "buy_pct": 57.2,
  "sell_pct": 42.8,
  "confidence": 0.64,
  "levels": {
    "avwap": [{"anchor": "pivot_low", "value": 103245.1}],
    "volume_profile": {"poc": 103200.0, "hvn": [102900.0], "lvn": [102450.0]}
  },
  "reasons": ["RANGE 국면: 평균회귀 가중치 적용"]
}
```

### 3.5 UI 1차 반영
- 파일
  - 수정: `/Users/mark/Desktop/자료/app/coin/static/app.js`
  - (필요시) 수정: `/Users/mark/Desktop/자료/app/coin/static/index.html`
- 구현 항목
  - `/api/analysis_trendy` 호출
  - regime 배지 표시
  - AVWAP 라인/POC 라인 표시 토글

---

## 4. P1 TODO (백테스트 + 보정)

### 4.1 백테스트 MVP 구축 (look-ahead 방지 포함)
- 파일
  - 추가: `/Users/mark/Desktop/자료/app/coin/app/backtest.py`
  - 추가: `/Users/mark/Desktop/자료/app/coin/app/labels.py`
  - 추가: `/Users/mark/Desktop/자료/app/coin/app/metrics.py`
- 함수
  - `label_barrier(df, t, tp_pct, sl_pct, max_holding_bars) -> int`
  - `run_backtest(config: dict) -> dict`
- CLI
  - `python -m app.backtest --config configs/backtest_trendy.json`
- 출력 산출물
  - `artifacts/backtests/<run_id>_summary.json`
  - `artifacts/backtests/<run_id>_preds.csv`

### 4.2 Calibration 학습 파이프라인
- 파일
  - 추가: `/Users/mark/Desktop/자료/app/coin/app/calibration.py`
- 함수
  - `fit_platt(p_raw: np.ndarray, y: np.ndarray) -> object`
  - `fit_isotonic(p_raw: np.ndarray, y: np.ndarray) -> object`
  - `apply_calibration(model, p_raw: float) -> float`
- 모델 저장
  - `artifacts/models/calibrator_platt.pkl`
  - `artifacts/models/calibrator_isotonic.pkl`

### 4.3 API에 보정 적용
- 파일
  - 수정: `/Users/mark/Desktop/자료/app/coin/app/main.py`
- 엔드포인트
  - `GET /api/analysis_trendy` 응답에 `calibrated`, `calibration_method` 추가
- 응답 스키마 추가 필드
```json
{
  "calibrated": true,
  "calibration_method": "platt"
}
```

---

## 5. P2 TODO (고도화)

### 5.1 레짐/레벨 고도화
- 파일
  - 수정: `/Users/mark/Desktop/자료/app/coin/app/regime.py`
  - 수정: `/Users/mark/Desktop/자료/app/coin/app/volume_profile.py`
- 항목
  - 레짐 히스테리시스(빈번 전환 방지)
  - VP 분배를 종가 단일 할당 -> 고저 구간 비례 분배로 개선

### 5.2 MTF 동적 가중치/설명가능성 강화
- 파일
  - 수정: `/Users/mark/Desktop/자료/app/coin/app/scoring.py`
- 함수
  - `build_reason_codes(...) -> list[str]`
  - `score_with_regime_weights(...) -> dict`
- 출력 스키마 확장
```json
{
  "reason_codes": ["TREND_EMA_ALIGN", "AVWAP_SUPPORT", "POC_ABOVE"],
  "feature_contrib": {"trend": 0.22, "levels": 0.18, "volatility": -0.07}
}
```

### 5.3 검증 자동화
- 파일
  - 추가: `/Users/mark/Desktop/자료/app/coin/scripts/run_walkforward.sh`
- 항목
  - 기간별 성능/캘리브레이션 드리프트 리포트 자동 생성

---

## 6. 과최적화/미래참조 방지 체크리스트
- `pivot right-bars` 확정 전 신호 사용 금지
- 학습/검증/테스트 시간 분리 고정
- 동일 구간으로 임계값/성능 동시 최적화 금지
- 수수료/슬리피지 반영 없는 결과는 참고용으로만 표시
- TF별/레짐별 표본 수 최소 기준 미달 시 “신뢰도 낮음” 표기

---

## 7. 구현 순서 제안
1. P0-레짐 + AVWAP + VP 계산 유닛 완성
2. P0-`/api/analysis_trendy` + UI 노출
3. P1-백테스트 + calibration 적용
4. P2-고도화 및 자동 검증
