@README.md

## Rules for this repo

- Run git and Python from `server-v3/`, never the parent `conductor/` folder.
- Code runs in Docker. Use `docker compose exec app ...`; there is no local venv.
- Tests go in `tests/` (pytest), never in `app/`. One-off experiments go in `evals/` or a scratch dir, never in `app/`.
- A file holds one concern you can say in one sentence. Split a file when it grows past ~250 lines or a second concern.
- Keep comments to one line of *why*. History and rationale go in the folder's README under "Quirks", and dated entries go in the feature log in the parent folder (`../README.md`, Logs).
- After editing `app/` or `seed/`, rebuild (`up -d --build`) before you test anything.
- Never merge or push to `main` without the user saying so: it deploys to prod.
