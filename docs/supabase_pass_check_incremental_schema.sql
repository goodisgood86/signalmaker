-- PASS 검증 누적 저장용 스키마
-- 1) 초기 적재: /api/pass_check_batch 호출
-- 2) 주간 증분: 마지막 signal_time_ms 이후만 append

create table if not exists public.pass_check_events (
  id bigint generated always as identity primary key,
  symbol text not null,
  market text not null,
  interval text not null,
  period text not null, -- 24h / 3d / 7d
  signal_time_ms bigint not null,
  entry_time_ms bigint null,
  executed boolean not null default false,
  result text not null, -- NO_ENTRY / TP1_HIT / SL_HIT / NO_HIT
  side text not null default 'WAIT',
  entry_price double precision null,
  tp_price double precision null,
  stop_price double precision null,
  created_ms bigint not null
);

create unique index if not exists idx_pass_check_events_uniq
  on public.pass_check_events(symbol, market, interval, period, signal_time_ms);

create index if not exists idx_pass_check_events_lookup
  on public.pass_check_events(symbol, market, interval, period, signal_time_ms desc);

create table if not exists public.pass_check_progress (
  id bigint generated always as identity primary key,
  symbol text not null,
  market text not null,
  interval text not null,
  period text not null,
  last_signal_time_ms bigint not null default 0,
  updated_ms bigint not null
);

create unique index if not exists idx_pass_check_progress_uniq
  on public.pass_check_progress(symbol, market, interval, period);
