-- Migration 004: event_costs view for the daily-spend circuit breaker (budget.py).
--
-- v3 only ever calls one model (config.MODEL, currently claude-sonnet-4-6) --
-- unlike v1's model-aware event_costs (server/migrations/001_kb.sql), there's
-- no per-event model column to branch pricing on, so this hardcodes the
-- current model's rate. Update the constants below if config.MODEL's pricing
-- tier changes.
--
-- Fresh install: runs automatically via docker-entrypoint-initdb.d/.
-- Existing volume: run as a single transaction against the live DB:
--   docker compose exec -T db psql -U conductor -d conductor -1 -f - < migrations/004_event_costs.sql

create or replace view event_costs as
select e.*,
    round(e.tokens_in / 1e6 * 3.00, 6) + round(e.tokens_out / 1e6 * 15.00, 6) as usd_cost
from events e;
