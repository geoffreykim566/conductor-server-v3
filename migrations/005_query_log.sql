-- Migration 005: persist lookup_concept queries for real-traffic KB gap
-- analysis (v3-kb-plan.md Phase 0 item 3). Result.trace previously lived
-- only in-memory per turn -- diagnosing a gap (e.g. the 2026-08-09 "how do
-- i make a beat" bug) required exec-ing into the running container to
-- root-cause it. This turns real traffic into a queryable backlog instead.
--
-- Fresh install: runs automatically via docker-entrypoint-initdb.d/.
-- Existing volume: run as a single transaction against the live DB:
--   docker compose exec -T db psql -U conductor -d conductor -1 -f - < migrations/005_query_log.sql

create table if not exists query_log (
    id          uuid primary key default gen_random_uuid(),
    created_at  timestamptz not null default now(),
    query       text not null,
    confidence  text not null,   -- short label: strong / moderate / weak / none
    top_results jsonb not null   -- [{kind, name, distance}, ...], up to 5
);

create index if not exists query_log_created_at on query_log (created_at);
