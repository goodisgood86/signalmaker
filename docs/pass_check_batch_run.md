PASS 검증 누적 배치 실행 순서

1) 서버 실행 (SUPABASE env 포함)

```bash
SUPABASE_URL='https://lmjqhtfavmtqitbimgqm.supabase.co' \
SUPABASE_SERVICE_ROLE_KEY='YOUR_KEY' \
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

2) 초기 적재 (기본 3d부터 권장, 순차 실행)

```bash
curl -X POST http://127.0.0.1:8000/api/pass_check_batch \
  -H 'Content-Type: application/json' \
  -d '{"market":"spot","interval":"5m","periods":["3d"],"symbols":["BTCUSDT"]}'

curl -X POST http://127.0.0.1:8000/api/pass_check_batch \
  -H 'Content-Type: application/json' \
  -d '{"market":"spot","interval":"5m","periods":["3d"],"symbols":["ETHUSDT"]}'

curl -X POST http://127.0.0.1:8000/api/pass_check_batch \
  -H 'Content-Type: application/json' \
  -d '{"market":"futures","interval":"5m","periods":["3d"],"symbols":["CROSSUSDT"]}'
```

3) 주간 증분 (기존 값 이후만 append)

```bash
curl -X POST http://127.0.0.1:8000/api/pass_check_batch \
  -H 'Content-Type: application/json' \
  -d '{"market":"spot","interval":"5m","periods":["24h","3d","7d"]}'

curl -X POST http://127.0.0.1:8000/api/pass_check_batch \
  -H 'Content-Type: application/json' \
  -d '{"market":"futures","interval":"5m","periods":["24h","3d","7d"],"symbols":["CROSSUSDT"]}'
```
