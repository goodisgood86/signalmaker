-- Simulation users
create table if not exists public.sim_users (
  id bigint generated always as identity primary key,
  nickname text not null unique,
  password_hash text not null,
  created_ms bigint not null
);

-- Simulation trade records
create table if not exists public.sim_trades (
  id bigint generated always as identity primary key,
  user_id bigint not null references public.sim_users(id) on delete cascade,
  nickname text not null,
  symbol text not null,
  entry_price double precision not null,
  take_profit double precision not null,
  stop_loss double precision not null,
  status text not null default 'OPEN',
  result_price double precision null,
  created_ms bigint not null,
  resolved_ms bigint null
);

create index if not exists idx_sim_trades_user_created on public.sim_trades(user_id, created_ms desc);
create index if not exists idx_sim_trades_symbol on public.sim_trades(symbol);
