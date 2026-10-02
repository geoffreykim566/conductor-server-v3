# core

Shared infrastructure that every other package imports. Nothing in here imports from `api/`, `pipeline/` or `tools/`.

## Files

| File | What |
|---|---|
| `config.py` | Every setting: env vars plus model names. Secrets only from env (`.env.local` locally) |
| `db.py` | asyncpg pool (pgvector + jsonb codecs) and every query outside the KB lookup and admin |
| `signing.py` | HMAC-signed conductor ids (`<uuid>.<sig>`) |
| `ratelimit.py` | slowapi limiter keyed by client IP |
| `budget.py` | Daily-spend circuit breaker |

## Quirks & why

- **Models:**
  - `MODEL` is the decider.
  - `WRITER_MODEL` is a cheaper Haiku, because the writer only phrases decided facts and never weighs evidence.
  - Changing the decider model needs a battery run first (the model switch has its own log).
- **`RESEARCH_CALL_TIMEOUT_S` = 150s.** 90s timed out every real research call, and a timed-out call can never earn the "research" tier.
- **`VOYAGE_MODEL`** must be the same at index and query time. After changing it, reseed (`python -m app.kb.load`).
- **Signing:** only `/v3/register` signs. A forged or bare uuid fails verification before touching the DB, and a valid token whose row was lost (DB reset) is upserted lazily (`api/deps.py`). No secret configured = registration returns 503 and every verify fails, closed.
- **Budget:** the console spend cap on the central key is the hard backstop, but tripping it takes the whole beta down until it resets. This breaker trips first, on trailing-24h `event_costs` > `DAILY_BUDGET_USD`, and recovers on its own. It's cached for 60s, so one aggregate query per minute.
- **Rate limit:** IPv6 is keyed per /64, because privacy extensions mint a fresh address per connection. A malformed IP is used as-is rather than raising.
- **`FREE_LIMIT`** is the default for new users. Raising one local user's count in dev: the `grant-free-messages` skill.
