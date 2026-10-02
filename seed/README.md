# seed

Content, not code. Both files are baked into the image, so rebuild after editing.

| File | What | Read by |
|---|---|---|
| `problems.json` | KB: `solutions` (name, kind, aliases, content, action) and `problems` (name, aliases, note, linked solutions with `seed_weight` + `distinguisher`) | `app/kb/load.py` (then reseed) |
| `routes.json` | Approved navigation routes: name -> `desc`, `path`, optional `choice` / `picks` / `wait_for_run` / `toggle_ax_key` | `app/tools/routes.py` at import |

- **The schema and quirks** for each: `app/kb/README.md` (KB) and `app/tools/README.md` (routes, dropdown `choice` blocks).
- **After editing `problems.json`:** `docker compose up -d --build && docker compose exec app python -m app.kb.load`, then run the battery.
- **After editing `routes.json`:** rebuild. Verify any new or changed path live in Logic Pro before trusting it, and ask the user to confirm menu paths against the running app rather than asserting them.
