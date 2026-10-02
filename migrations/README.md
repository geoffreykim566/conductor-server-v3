# migrations

Plain SQL files, applied in filename order. There's no migration runner.

- **Local:** mounted as `docker-entrypoint-initdb.d`, so they run **only when the DB volume is first created**. On an existing volume, apply a new file by hand:
  `docker compose exec -T db psql -U conductor -d conductor < migrations/00N_name.sql`
- **Prod (Railway):** apply by hand the same way, against the prod DB, as part of the release. A missing migration fails silently at runtime (every lookup failed once because 008 hadn't been applied).
- **Release order: migrate -> deploy -> reseed.** New code queries the new columns, so the migration goes first; the reseed after the deploy is what fills them (e.g. `solutions.action`).

## Adding one

Name it `00N_what_it_does.sql`, the next number. Make it idempotent where you can (`add column if not exists`), since it may be re-applied by hand. Apply it to the local DB, then rebuild and reseed if the KB tables changed.
