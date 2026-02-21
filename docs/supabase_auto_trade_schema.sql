-- Auto-trade settings (1 row per user)
create table if not exists public.auto_trade_settings (
  id bigint generated always as identity primary key,
  user_id bigint not null references public.sim_users(id) on delete cascade,
  enabled boolean not null default false,
  mode text not null default 'balanced',
  symbol text not null default 'ALL',
  market text not null default 'spot',
  interval text not null default '5m',
  futures_leverage integer not null default 3,
  order_size_usdt double precision not null default 120,
  take_profit_pct double precision not null default 1.8,
  stop_loss_pct double precision not null default 1.0,
  daily_max_loss_usdt double precision not null default 120,
  cooldown_min integer not null default 30,
  max_open_positions integer not null default 1,
  last_run_ms bigint not null default 0,
  created_ms bigint not null,
  updated_ms bigint not null,
  unique(user_id)
);

create index if not exists idx_auto_trade_settings_user on public.auto_trade_settings(user_id);

-- Existing table migration helpers
alter table public.auto_trade_settings
  add column if not exists futures_leverage integer not null default 3;

alter table public.auto_trade_settings
  alter column symbol set default 'ALL';

-- Binance API link (1 row per user)
create table if not exists public.auto_trade_binance_links (
  id bigint generated always as identity primary key,
  user_id bigint not null references public.sim_users(id) on delete cascade,
  market text not null default 'spot',
  api_key text not null,
  api_secret text not null, -- 암호화된 값(v1:... token) 저장
  status text not null default 'CONNECTED',
  linked_ms bigint not null,
  updated_ms bigint not null,
  unique(user_id)
);

create index if not exists idx_auto_trade_binance_links_user on public.auto_trade_binance_links(user_id);

-- Auto-trade records
create table if not exists public.auto_trade_records (
  id bigint generated always as identity primary key,
  user_id bigint not null references public.sim_users(id) on delete cascade,
  symbol text not null,
  market text not null default 'spot',
  interval text not null default '5m',
  mode text not null default 'balanced',
  side text not null default 'BUY',
  status text not null default 'OPEN',
  entry_price double precision not null,
  take_profit_price double precision not null,
  stop_loss_price double precision not null,
  qty double precision not null,
  notional_usdt double precision not null,
  close_price double precision null,
  pnl_usdt double precision null,
  opened_ms bigint not null,
  closed_ms bigint null,
  signal_buy_pct double precision null,
  signal_sell_pct double precision null,
  signal_confidence double precision null,
  decision_diff double precision null,
  reason text null
);

create index if not exists idx_auto_trade_records_user_opened on public.auto_trade_records(user_id, opened_ms desc);
create index if not exists idx_auto_trade_records_user_status on public.auto_trade_records(user_id, status);
create index if not exists idx_auto_trade_records_symbol on public.auto_trade_records(symbol);

-- Auto-trade execution audit log (order/DB mismatch trace)
create table if not exists public.auto_trade_exec_audit (
  id bigint generated always as identity primary key,
  user_id bigint not null references public.sim_users(id) on delete cascade,
  record_id bigint null references public.auto_trade_records(id) on delete set null,
  event text not null,
  level text not null default 'INFO',
  symbol text null,
  market text null,
  mode text null,
  side text null,
  status text null,
  qty double precision null,
  price double precision null,
  pnl_usdt double precision null,
  order_id text null,
  client_order_id text null,
  detail text null,
  payload text null,
  created_ms bigint not null
);

create index if not exists idx_auto_trade_exec_audit_user_created on public.auto_trade_exec_audit(user_id, created_ms desc);
create index if not exists idx_auto_trade_exec_audit_user_record on public.auto_trade_exec_audit(user_id, record_id, created_ms desc);
create index if not exists idx_auto_trade_exec_audit_user_event on public.auto_trade_exec_audit(user_id, event, created_ms desc);
