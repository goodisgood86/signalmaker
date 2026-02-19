# PASS 검증 DB 운영 (백업/배치)

## 1) 환경변수

```bash
cp configs/ops.env.example configs/ops.env
# configs/ops.env에 실제 값 입력 후
source configs/ops.env
```

## 1-1) Supabase 스키마 (1회)

아래 SQL 2개를 Supabase SQL Editor에서 실행해야 합니다.

- `docs/supabase_pass_check_incremental_schema.sql`
- `docs/supabase_pass_check_summary_schema.sql`

## 2) 배치 실행 (증분 append)

```bash
./scripts/pass_check_batch_runner.py \
  --base-url "$APP_BASE_URL" \
  --interval 5m \
  --periods 24h,3d,7d
```

- 기본 심볼
  - spot: `BTCUSDT,ETHUSDT,XRPUSDT,DOGEUSDT,SUIUSDT,SOLUSDT`
  - futures: `CROSSUSDT`
- 이미 적재된 신호는 `pass_check_progress.last_signal_time_ms` 기준으로 건너뜁니다.

## 3) DB 백업

```bash
./scripts/pass_check_backup.py --out-dir backups/pass_check
```

- 백업 대상 테이블
  - `pass_check_events`
  - `pass_check_progress`
- 생성 파일
  - `backups/pass_check/<timestamp>/pass_check_events.json`
  - `backups/pass_check/<timestamp>/pass_check_progress.json`
  - `backups/pass_check/<timestamp>/manifest.json`
  - `backups/pass_check/LATEST`

## 4) 주간 운영 예시

```bash
# 1) 백업
./scripts/pass_check_backup.py --out-dir backups/pass_check

# 2) 증분 배치
./scripts/pass_check_batch_runner.py --base-url "$APP_BASE_URL" --interval 5m --periods 24h,3d,7d
```

## 5) 자동 실행 (cron)

다음 2개가 주간 자동 실행으로 등록되어 있습니다.

```cron
50 2 * * 1 /bin/zsh /Users/mark/coin_app/scripts/pass_check_weekly_backup.sh >> /tmp/coin_passcheck_backup_cron.log 2>&1
10 3 * * 1 /bin/zsh /Users/mark/coin_app/scripts/pass_check_weekly_batch.sh >> /tmp/coin_passcheck_batch_cron.log 2>&1
```

- 월요일 02:50: DB 백업
- 월요일 03:10: PASS 증분 배치
