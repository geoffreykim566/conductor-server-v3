# seed

Content, not code. Both files are baked into the image, so rebuild after editing.

| File | What | Read by |
|---|---|---|
| `problems.json` | KB: `solutions` (name, kind, aliases, content, action) and `problems` (name, aliases, note, linked solutions with `seed_weight` + `distinguisher`) | `app/kb/` at import (rendered into the prompt) |
| `routes.json` | Approved navigation routes: name -> `desc`, `path`, optional `choice` / `picks` / `wait_for_run` / `toggle_ax_key` | `app/kb/` at import (paths in the prompt; runnable routes to `app/tools/`) |

- **The schema and quirks** for each: `app/kb/README.md` (KB) and `app/tools/README.md` (routes, dropdown `choice` blocks).
- **After editing `problems.json`:** rebuild (`docker compose up -d --build`), then run the core battery. There's no seeding step.
- **Menu paths go only in `routes.json`.** KB prose names what to do; the path comes from a route or a `reference` entry (`"reference": true`, `"text"`) in its `where` list.
- **After editing `routes.json`:** rebuild. Verify any new or changed path live in Logic Pro before trusting it, and ask the user to confirm menu paths against the running app rather than asserting them.
- **Dropdown `options` are copied from the running Logic** (the client's AX read or the user's screenshot), never guessed: guessed strings failed the client's choose and read-back.
