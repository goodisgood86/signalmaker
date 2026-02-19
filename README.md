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

## API

- `GET /api/analysis?symbol=BTCUSDT&interval=5m&limit=500`
- `GET /api/analysis_trendy?symbol=BTCUSDT&interval=5m&limit=500`
- `GET /api/analysis_trendy_mtf?symbol=BTCUSDT&limit=500`
- `GET /api/klines?symbol=BTCUSDT&interval=5m&limit=500`

## 백테스트 + 보정

```bash
source .venv/bin/activate
python -m app.backtest --config configs/backtest_trendy.json
```

생성 파일:
- `artifacts/backtests/*_summary.json`
- `artifacts/backtests/*_preds.csv`
- `artifacts/models/isotonic_calibrator.json` 또는 `artifacts/models/platt_calibrator.json`
