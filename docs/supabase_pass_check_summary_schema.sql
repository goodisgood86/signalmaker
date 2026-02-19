-- PASS 검증 조회 성능 개선용 요약 통계 테이블
create table if not exists public.pass_check_summary (
  id bigserial primary key,
  symbol text not null,
  market text not null,
  interval text not null,
  period text not null,
  pass_count integer not null default 0,
  executed_count integer not null default 0,
  no_entry_count integer not null default 0,
  tp1_hit_count integer not null default 0,
  sl_hit_count integer not null default 0,
  no_hit_count integer not null default 0,
  resolved_count integer not null default 0,
  latest_signal_time_ms bigint not null default 0,
  updated_ms bigint not null default 0
);

create unique index if not exists idx_pass_check_summary_uniq
  on public.pass_check_summary(symbol, market, interval, period);

create index if not exists idx_pass_check_summary_lookup
  on public.pass_check_summary(symbol, market, interval, period, updated_ms desc);
