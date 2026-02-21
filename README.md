# coin (Binance TA MVP)

바이낸스 캔들(OHLCV)을 가져와 기술적 지표 기반으로 `매수 %`, `매도 %`를 계산해 보여주는 간단한 웹 MVP입니다.

## 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

브라우저에서 `http://127.0.0.1:8000` 접속.

## 운영 환경변수 (Supabase)

```bash
cp configs/ops.env.example configs/ops.env
# configs/ops.env 에 실제 값 입력
source configs/ops.env
```

### 접근 키 잠금(권장)

접근 키를 설정하면, 키를 통과한 세션만 사이트/API를 사용할 수 있습니다.

1) 키 해시 생성
```bash
python3 - <<'PY'
import hashlib
raw = input("access key: ").strip()
print(hashlib.sha256(raw.encode("utf-8")).hexdigest())
PY
```

2) `configs/ops.env`에 설정
```bash
export APP_ACCESS_KEY_HASH='위에서 생성한_해시'
```

3) 서버 재시작 후 `/static/auth.html`에서 키 입력

### 자동매매 설정 잠금(서버 검증)

프론트 코드가 아닌 서버에서 비밀번호를 검증합니다.

1) 잠금 비밀번호 해시 생성
```bash
python3 - <<'PY'
import hashlib
raw = input("config unlock key: ").strip()
print(hashlib.sha256(raw.encode("utf-8")).hexdigest())
PY
```

2) `configs/ops.env`에 설정
```bash
export APP_CONFIG_UNLOCK_KEY_HASH='위에서_생성한_해시'
```

3) 바이낸스 API Secret 암호화 키 설정(필수)
```bash
export AUTO_TRADE_SECRET_ENC_KEY='충분히_긴_랜덤_문자열'
```

4) 자동매매 백그라운드 실행/실거래 옵션
```bash
# 서버에서 자동매매 tick 주기 실행(기본: 1)
export AUTO_TRADE_BG_ENABLED=1
# tick 주기 초(기본: 15초, 최소 5초)
export AUTO_TRADE_TICK_INTERVAL_S=15
# 실제 바이낸스 주문 실행(기본: 1)
export AUTO_TRADE_LIVE_ENABLED=1
# 플로우 스코어 가중치(기본 0.25, 허용 0.20~0.30)
export AUTO_FLOW_SCORE_WEIGHT=0.25
```

## API

- `GET /api/analysis?symbol=BTCUSDT&interval=5m&limit=500`
- `GET /api/analysis_trendy?symbol=BTCUSDT&interval=5m&limit=500`
- `GET /api/analysis_trendy_mtf?symbol=BTCUSDT&limit=500`
- `GET /api/klines?symbol=BTCUSDT&interval=5m&limit=500`
- `GET /api/auto_trade/config`
- `POST /api/auto_trade/config`
- `GET /api/auto_trade/config_lock/status`
- `POST /api/auto_trade/config_lock/unlock`
- `POST /api/auto_trade/config_lock/lock`
- `GET /api/auto_trade/binance/link`
- `GET /api/auto_trade/binance/collateral` (Spot 보유코인 환산 + Funding/Alpha 포함)
- `POST /api/auto_trade/binance/link` (API Key/Secret 1회 입력으로 spot+futures 동시 연동)
- `DELETE /api/auto_trade/binance/link`
- `POST /api/auto_trade/tick`
- `GET /api/auto_trade/records`
- `GET /api/auto_trade/audit` (실주문 감사로그 조회)
- `GET /api/auto_trade/stats`

자동매매 기능을 쓰려면 아래 스키마를 먼저 적용하세요.

- `docs/supabase_auto_trade_schema.sql`

## 백테스트 + 보정

```bash
source .venv/bin/activate
python -m app.backtest --config configs/backtest_trendy.json
```

생성 파일:
- `artifacts/backtests/*_summary.json`
- `artifacts/backtests/*_preds.csv`
- `artifacts/models/isotonic_calibrator.json` 또는 `artifacts/models/platt_calibrator.json`
