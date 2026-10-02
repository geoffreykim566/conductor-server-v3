# Conductor server (v3)

The API the Conductor macOS client talks to. A user's message plus live Logic Pro context (screenshots, an Accessibility dump) goes through a tool-calling pipeline. The pipeline returns a reply and, when the user wants something done, a **card**: steps the client runs on their Mac when they press Run. FastAPI + Postgres/pgvector, deployed on Railway.

## Layout

| Path | What |
|---|---|
| `app/api/` | HTTP layer: routes, request validation, SSE, `/admin` dashboard |
| `app/pipeline/` | One turn: decider tool loop -> card -> writer reply. **Start here for behaviour.** |
| `app/tools/` | Tool schemas the decider sees, and their executors (lookup, actions, routes) |
| `app/prompts/` | Decider and writer system prompts |
| `app/research/` | `web_research`: nested web-search call |
| `app/kb/` | Embeddings and seeding the KB from `seed/` |
| `app/core/` | Config, DB, signing, rate limit, budget |
| `seed/` | KB content (`problems.json`) and approved navigation routes (`routes.json`) |
| `migrations/` | SQL schema, applied on a fresh DB volume |
| `tests/` | pytest unit tests (mocked model, no DB) |
| `evals/` | Graded scenario battery, latency A/B harness, manual turn driver |
| `test_runs/` | Gitignored output of battery/eval runs, one folder per date |

Each folder has its own README with the details and quirks for that part.

## Running locally

```bash
docker compose up -d --build                         # db on :5435, API on :8000
docker compose exec app python -m app.kb.load        # seed/reseed the KB (fresh volume, or after editing seed/)
docker compose exec app python -m pytest             # unit tests
docker compose exec app python -m evals.battery.runner > test_runs/$(date +%F)/run.log   # battery
```

`.env.local` holds the secrets (`CENTRAL_ANTHROPIC_KEY`, `VOYAGE_API_KEY`, `CONDUCTOR_ID_SECRET`, `ADMIN_PASSWORD`). Every setting is in `app/core/config.py`.

## Quirks

- **`app/` is baked into the image.** After editing anything under `app/` or `seed/`, run `docker compose up -d --build` or the container keeps running the old code. `tests/`, `evals/` and `test_runs/` are bind-mounted and live.
- **Migrations only run on a fresh volume** (they're mounted as `docker-entrypoint-initdb.d`). For an existing DB, apply the new file by hand: `docker compose exec -T db psql -U conductor -d conductor < migrations/00N_x.sql`. Prod gets the same by hand.
- **Second stack for a branch or worktree:** `docker compose -p <name> -f docker-compose.yml -f docker-compose.standalone.yml up -d --build`. It has no ports, bind-mounts the whole checkout with `PYTHONPATH=/srv` (so `app/` edits are live), and uses its own DB volume (seed it once). Drive it with `docker compose -p <name> -f docker-compose.yml -f docker-compose.standalone.yml exec -T app ...`.
- **Stale code before new theories.** If behaviour doesn't match the code, first check the image was rebuilt and the KB reseeded.
- **Log with `log.warning("[marker] ...")`, never `log.info`.** The root logger isn't configured, so info lines never reach `docker compose logs`; grep-able `[markers]` are the convention.
- **Stateless.** Nothing is stored between requests except users/events. The client round-trips the full history (`app/api/README.md`).

## Deploying

`main` auto-deploys to Railway. Work on a version branch (`v0xx`). Merge to `main` only through the `release-client` skill, and only when the user asks. The Dockerfile's `CMD` is the prod entrypoint (`uvicorn app.api.main:app`). Railway builds without `DEV=1`, so pytest isn't in the prod image.
