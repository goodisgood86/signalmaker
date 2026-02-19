# SIGNAL_MODEL

## 목차
1. “매수%/매도%” 의미 정의
2. 멀티 타임프레임 결합 설계 (4h/1h/5m)
3. 컨플루언스 점수화 모델
4. 신뢰도(confidence) 산출 제안
5. 현재 구조 매핑
6. 다음 작업 TODO

## 1) “매수%/매도%” 의미 정의
현재 구현은 규칙 점수 `x`를 sigmoid로 변환한 상대 확률 유사 값이다. 앞으로는 아래 중 하나로 명시적 의미를 고정해야 한다.

### 정의안 A (권장)
- 매수%: “진입 후 N봉 이내 TP(+X%)를 SL(-Y%)보다 먼저 달성할 확률”
- 매도%: 반대 방향 확률
- 장점: 실거래 의사결정(손익비)과 직접 연결
- 단점: X/Y/N 설정에 성능 민감

### 정의안 B
- 매수%: “N봉 후 수익률 > 0일 확률”
- 매도%: “N봉 후 수익률 < 0일 확률”
- 장점: 라벨 단순
- 단점: 중간 경로(최대역행, 손절) 미반영

### 정의안 C
- 매수%: “기대수익 E[R] > 0일 확률”
- 장점: 비용/리스크 반영 가능
- 단점: 모델링 복잡, 데이터 요구 증가

권장 채택: 단기 트레이딩 앱 성격상 A를 공식 정의로 채택.

---

## 2) 멀티 타임프레임 결합 설계 (4h/1h/5m)

### 역할 분리
- 4h (필터): 레짐/대추세 판정
- 1h (셋업): 눌림/돌파/레벨 근접 등 진입 준비 상태
- 5m (트리거): 실제 진입 타이밍

### 예시 규칙
1. 4h 필터
- `EMA50 기울기 > 0`, `가격 > EMA50`, `ADX > 22`면 롱만 허용
- 반대면 숏만 허용
- ADX < 18이면 레인지 모드

2. 1h 셋업
- 롱 모드에서: 1h RSI가 40~55 눌림 + MACD hist 재상향 + 피보 0.382~0.618 근접
- 숏 모드에서 반대로 적용

3. 5m 트리거
- VWAP 재돌파 + 직전 스윙 고점(BOS) 돌파 + 거래량 증가
- 손절: 최근 5m pivot low/high 또는 ATR 기반

---

## 3) 컨플루언스 점수화 모델

### 기본 형태
- 전체 점수: `S = Σ(w_i * f_i)`
- 확률 변환: `P_buy = sigmoid(alpha * S + b)`
- `P_sell = 1 - P_buy` 또는 별도 대칭 점수

### feature 그룹
- Trend: EMA 정렬, ADX, Ichimoku 위치
- Momentum: RSI, MACD, Stoch
- Volatility: ATR percentile, Bollinger bandwidth
- Volume: OBV/CMF/MFI, VWAP 거리
- Structure: BOS/CHOCH, SR/피보 근접도, 다이버전스

### 레짐 필터 아이디어
- Regime ∈ {Trend, Range, HighVol}
- 레짐별 `w_i`를 다르게 적용
  - Trend: 추세/모멘텀 가중치↑, 역추세 오실레이터 가중치↓
  - Range: mean-reversion 가중치↑
  - HighVol: 신호 자체보다 포지션 축소/패널티 강화

### 초기 가중치 예시
- Trend(0.30), Momentum(0.25), Volatility(0.15), Volume(0.15), Structure(0.15)
- 백테스트로 교정(calibration) 필요

---

## 4) 신뢰도(confidence) 산출 제안

현재 confidence는 `|buy-50|` 기반으로 단순하다. 개선식 제안:

```text
confidence = clip01(
  c1 * normalized_confluence_count
+ c2 * regime_fit_score
+ c3 * signal_stability
- c4 * volatility_overheat_penalty
- c5 * feature_conflict_penalty
)
```

구성요소 예시:
- `normalized_confluence_count`: 활성 신호 수 / 전체 후보 수
- `regime_fit_score`: 현재 신호가 레짐과 일치하는 정도
- `signal_stability`: 최근 k봉 연속 일관성
- `volatility_overheat_penalty`: ATR percentile 과열 구간 패널티
- `feature_conflict_penalty`: 추세/모멘텀 신호 충돌 패널티

---

## 5) 현재 구조 매핑

### 현재 구현됨
- `app/scoring.py`
  - 선형 합산 + sigmoid
  - 사용 feature: EMA/RSI/MACD hist/BB 위치/전봉 수익률
- `app/main.py`
  - 단일 interval 분석 API (`/api/analysis`)
- `static/app.js`
  - 선택 interval 결과 표시(멀티TF 결합 없음)

### 미구현
- 멀티TF 동시 조회 및 결합 점수
- 레짐 분류기(ADX/변동성 기반)
- confidence 고도화
- 확률 보정(Platt/Isotonic 등)

---

## 6) 다음 작업 TODO

### P0
- `score_signal_mtf()` 설계/구현: 4h/1h/5m 입력을 받아 단일 `buy/sell/confidence` 출력
- 레짐 분류기 추가(최소 ADX + BB폭 + ATR percentile)
- API 확장: `/api/analysis_mtf?symbol=BTCUSDT`

### P1
- feature conflict 감점과 signal stability 보너스 추가
- 확률 캘리브레이션(검증셋 기준 reliability curve)

### P2
- 단순 룰에서 ML 하이브리드(룰+로지스틱) 전환 실험
