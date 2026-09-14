-- Migration 007: columns the admin dashboard (app/admin.py, ported from v1)
-- needs on events -- source/model/latency_ms/prompt/response/type/source_tier,
-- same as v1's server/migrations/000_base.sql + 001_kb.sql. v3 only ever calls
-- one model and has one event type, so those get simple defaults instead of
-- v1's multi-model/multi-type bookkeeping.
--
-- Fresh install: runs automatically via docker-entrypoint-initdb.d/.
-- Existing volume: run as a single transaction against the live DB:
--   docker compose exec -T db psql -U conductor -d conductor -1 -f - < migrations/007_admin_events.sql

alter table events add column if not exists source      text not null default 'free';
alter table events add column if not exists model       text;
alter table events add column if not exists latency_ms  int;
alter table events add column if not exists prompt      text;
alter table events add column if not exists response    text;
alter table events add column if not exists type        text not null default 'chat';
alter table events add column if not exists source_tier text;
