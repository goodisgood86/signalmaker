-- 기존 sim_trades 테이블에서 market/interval/side 저장 제거
-- Supabase SQL Editor에서 1회 실행

drop index if exists public.idx_sim_trades_symbol_market;

alter table if exists public.sim_trades
  drop column if exists market,
  drop column if exists interval,
  drop column if exists side;

create index if not exists idx_sim_trades_symbol on public.sim_trades(symbol);
