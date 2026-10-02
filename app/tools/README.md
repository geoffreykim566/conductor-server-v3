# tools

The decider's tools: their schemas (`TOOLS`) and executors. There are two kinds:

- **Knowledge:** `lookup_concept` (KB vector search), `web_research` (executor in `app/research/`), and `ask_clarifying_question` (no-op marker).
- **Actions:** `open_plugin`, `set_param`, `open_setting`. These don't *do* anything server-side. They validate and emit wire steps that the client runs when the user presses Run.

## Files

| File | What |
|---|---|
| `schemas.py` | The six tool definitions sent to the model, plus `TOOLS` and `ACTION_TOOLS` |
| `lookup.py` | `lookup_concept`: ranked search over problems + solutions; confidence labels; Voyage retry |
| `actions.py` | Executors for the three action tools; `action_calls` (a KB solution's stored action -> calls) |
| `choices.py` | Dropdown routes: match a value to an option, check the user named it |
| `routes.py` | `ROUTES` loaded from `seed/routes.json`; how a route is described to the model |
| `route_steps.py` | A route's `path` -> client wire steps (menu chains, shortcuts, clicks) |
| `turn_text.py` | ContextVars holding this turn's user text and the reply it answers |

## Wire steps the client runs

`{"ax_open_plugin": name, "track"?, "new"?}`, `{"ax_set_param": {plugin, param, value, track?}}`, `{"menu_path": [...]}`, `{"shortcut": "Cmd+F"}`, `{"click_text": ...}`, `{"click_value_of": label}`, `{"choose": option, "reopen": [steps], "shows"?}`. Any step may carry `expect`. The client's executor is in `client-v3/core/automation/`. Changing a shape here means changing it there too.

## Quirks & why

- **Actions are typed tools, not KB rows.** When they were KB rows ("open plugin" behind `lookup_concept`), the model had to find them first, and their names collided with plugin vocabulary ("plugin manager", "multipressor").
- **`set_param.value` is a string**, parsed server-side (`_parse_param_value`). A number-or-on/off union wasn't held by the model even under strict mode.
- **Routes are the only way to navigate.** The model names a route; the path that runs is always `seed/routes.json`'s. Model-written menu paths have been confidently wrong more than anything else in this codebase.
- **`lookup_concept` ranks problems and solutions in one pass.** Checking one type first starved better matches of the other. A solution hit always resolves to its bucket (the one it's weighted highest in), so distance noise can't hide its siblings.
- **The confidence bands (0.40 / 0.60)** are advisory text for the model, not a filter. A hard floor kept admitting a different wrong match on every rephrase. They were tuned at 8 problems and never recalibrated, so gibberish can still read "moderate".
- **The toggle gate:** a route with `toggle_ax_key` is refused when live state says it's already in its target state, because running it would flip it *away*.

### Dropdown routes (`choice` block in routes.json)
- `options` lists the dropdown's values. An empty list means not verified yet: the route opens the pane only.
- `ordered` allows `larger` / `smaller`.
- `model_may_pick` lists the values the model may choose on its own. **Any other value must appear in the user's message, or in the reply they're answering** (`TURN_OFFERED_TEXT`), checked in `choices.resolve_choice`.
- `shows` maps an option to the shorter text the control displays, for the client's read-back.
- A dropdown route either picks a value (`choose` step) or stops at the pane. It **never leaves a menu open**: an open menu swallowed the next step's keystrokes.
- **Pane-only routes:** the dropped dropdown row becomes the previous step's `expect`. The client then skips the menu or disclosure click when the row is already showing.
- **`picks`:** a route fixed to one option ("flex pitch") ends on a `choose` step, not a click. The client closes stray menus before every step except a choose, and a choose ledgers the old value for Revert.
- **`wait_for_run`:** the route's target depends on the user's selection, so its card never auto-runs.
- An ambiguous value ("flex on" fits four options) is refused with the list, so the reply asks which one.

## Adding things

- **A route:** add it to `seed/routes.json` (`desc`, `path`, optional `choice` / `picks` / `wait_for_run` / `toggle_ax_key`). Verify the path live in Logic first. Rebuild (it's read at import).
- **A KB solution that maps to an action:** set its `action` in `seed/problems.json` (`{"open_setting": "route name"}` or `{"open_plugin": {...}}`). `app.kb.load` rejects unknown routes.
- **A new action tool:** schema in `schemas.py` + `ACTION_TOOLS`, executor in `actions.py`, a case in `action_calls`, then the pipeline side (`app/pipeline/README.md`, Adding things), then the client executor.
