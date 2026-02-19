# BACKTESTING

## 목차
1. 백테스트 핵심 원칙
2. 라벨링 정의 예시 3가지
3. 추천 성능 지표
4. 프로젝트 맞춤 백테스트 MVP 설계
5. 현재 구조 매핑
6. 다음 작업 TODO

## 1) 백테스트 핵심 원칙

### 1.1 Look-ahead 방지
- 신호 시점 `t`에서 사용 가능한 데이터는 `t`까지로 제한
- pivot 기반 신호는 `right bars` 이후 확정됨을 반영(신호 지연 모델링)
- 지표 계산 시 미래 봉 참조 금지

### 1.2 비용 반영
- 수수료: 진입/청산 모두 반영(예: taker 0.04% 가정)
- 슬리피지: 변동성/유동성 기반 고정 또는 ATR 비례
- 스프레드 대체 비용 포함(현물/선물 모드별 분리 가능)

### 1.3 데이터 품질
- 결측 봉 처리 규칙 명시: drop/forward-fill/재조회
- 중복 timestamp 제거
- 타임존/정렬(오름차순) 강제

### 1.4 체결 규칙
- 봉내 TP/SL 동시 터치 시 우선순위 규칙 고정(보수적: SL 우선)
- 시장가/지정가 체결 가정 분리

---

## 2) 라벨링 정의 예시 3가지

### A. Barrier Label (권장)
- 정의: N봉 내 +X%(TP) 또는 -Y%(SL) 중 먼저 닿는 이벤트로 성공/실패 라벨
- 장점: 리스크/리워드 직접 반영
- 단점: 파라미터(X,Y,N) 최적화 과적합 위험

### B. Horizon Return Label
- 정의: N봉 후 수익률 부호(>0 성공)
- 장점: 단순, 빠름
- 단점: 경로 의존 리스크 미반영

### C. Triple-Barrier + 시간 만료
- 정의: TP/SL/시간 만료 3중 장벽으로 다중 클래스 라벨
- 장점: 정보량 높고 실전 유사
- 단점: 구현 복잡, 해석 난이도 증가

---

## 3) 추천 성능 지표
- 승률(Win rate)
- 기대값(Expectancy): `E = win_rate*avg_win - loss_rate*avg_loss`
- Profit Factor
- MDD (최대낙폭)
- 샤프/소르티노
- 거래 빈도(일/주/월)
- 평균 보유시간
- 수수료 차감 후 순수익률
- 레짐별 성능 분해(Trend/Range/HighVol)

---

## 4) 프로젝트 맞춤 백테스트 MVP 설계

### 4.1 목표
현 구조(바이낸스 OHLCV + 룰 점수)를 유지하면서, 시그널 정의안 A를 검증 가능한 최소 파이프라인 제공.

### 4.2 입력 JSON 예시
```json
{
  "symbol": "BTCUSDT",
  "interval": "5m",
  "from": "2024-01-01T00:00:00Z",
  "to": "2025-12-31T23:59:59Z",
  "strategy": {
    "score_buy_threshold": 0.58,
    "score_sell_threshold": 0.58,
    "tp_pct": 0.012,
    "sl_pct": 0.008,
    "max_holding_bars": 24
  },
  "cost": {
    "fee_bps": 4,
    "slippage_bps": 2
  }
}
```

### 4.3 출력 JSON 예시
```json
{
  "summary": {
    "trades": 412,
    "win_rate": 0.47,
    "expectancy": 0.0018,
    "profit_factor": 1.19,
    "max_drawdown": 0.142,
    "sharpe": 1.08
  },
  "by_regime": {
    "trend": {"trades": 220, "expectancy": 0.0031},
    "range": {"trades": 150, "expectancy": -0.0004},
    "high_vol": {"trades": 42, "expectancy": -0.0012}
  },
  "trades_path": "artifacts/backtests/20260217_BTCUSDT_5m_trades.csv",
  "equity_path": "artifacts/backtests/20260217_BTCUSDT_5m_equity.csv"
}
```

### 4.4 저장 형식
- 요약: `artifacts/backtests/<run_id>_summary.json`
- 트레이드 로그: `artifacts/backtests/<run_id>_trades.csv`
- 에쿼티 커브: `artifacts/backtests/<run_id>_equity.csv`

### 4.5 실행 커맨드 제안
```bash
python -m app.backtest --config configs/backtest_mvp.json
```

### 4.6 최소 모듈 구성 제안
- `app/backtest.py`: 실행 엔트리
- `app/labels.py`: barrier label
- `app/execution.py`: 체결/수수료/슬리피지
- `app/metrics.py`: 성능지표 계산

---

## 5) 현재 구조 매핑

### 현재 구현됨
- 실시간/준실시간 분석 API와 지표 계산 파이프라인 존재
- 룰 기반 스코어 -> 확률 유사 출력 존재

### 미구현
- 백테스트 엔진 전반(라벨, 체결, 지표 집계, 산출물 저장)
- 기간 분할 검증(Train/Validation/Test)
- 파라미터 스윕/워크포워드

---

## 6) 다음 작업 TODO

### P0
- 백테스트 MVP 스캐폴드 구현(`app/backtest.py` 등)
- Barrier Label(A안) + 비용 반영 체결 로직 완성
- 요약/트레이드/에쿼티 산출물 저장

### P1
- 레짐별 성능 리포트 및 파라미터 민감도 분석
- 멀티 타임프레임(4h/1h/5m) 백테스트 지원

### P2
- 워크포워드 자동화 및 확률 캘리브레이션 평가
