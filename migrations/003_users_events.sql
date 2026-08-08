-- Users + events: real per-user free-tier counter and rating targets for
-- /v1/me and /v1/ratings. No HMAC signing / IP rate-limiting yet (deferred,
-- see api.py's module docstring) -- id is whatever X-Conductor-Id carries,
-- trusted as-is for now.
--
-- Fresh install: runs automatically via docker-entrypoint-initdb.d/.
-- Existing volume: run as a single transaction against the live DB:
--   docker compose exec -T db psql -U conductor -d conductor -1 -f - < migrations/003_users_events.sql

create extension if not exists "pgcrypto";

create table if not exists users (
    id              uuid primary key,
    created_at      timestamptz not null default now(),
    last_seen_at    timestamptz not null default now(),
    free_limit      int  not null default 50,
    free_used       int  not null default 0,
    experience      text,
    role            text,
    uninstalled_at  timestamptz
);

create table if not exists events (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references users(id),
    created_at  timestamptz not null default now(),
    tokens_in   int not null,
    tokens_out  int not null,
    rating      smallint,   -- 1 up | -1 down | null = no vote (client sends 0 to undo)
    rated_at    timestamptz
);

create index if not exists events_user_day on events (user_id, created_at);
