# tests/

pytest unit tests. Deterministic: the model is mocked, nothing hits the Anthropic,
Voyage or web-search APIs, and no test needs the database.

## Layout

| Path | What |
|---|---|
| `conftest.py` | Shared fakes: response builders, KB/route helpers, the `model` / `research` / `run` fixtures |
| `api/` | Request validation (`app.api.schemas`) |
| `core/` | Budget breaker, rate-limit keying, ID signing (`app.core`) |
| `pipeline/` | One turn end to end through `pipeline.respond()`, plus its helpers, one concern per file |
| `tools/` | Tool executors called directly (`app.tools`): action args, dropdown routes |
| `evals/` | The battery grader (`evals.battery.grader`) on hand-built traces |

## Running

Main stack (`tests/` is bind-mounted, `app/` is baked in: rebuild after editing `app/`):

```bash
docker compose exec app python -m pytest                                     # all
docker compose exec app python -m pytest tests/pipeline/test_resume.py       # one file
docker compose exec app python -m pytest tests/pipeline/test_resume.py::test_resume_keeps_replacements  # one test
docker compose exec app python -m pytest -k pick_or_ask                      # by name
```

Isolated stack (whole checkout bind-mounted, `app/` edits live):

```bash
docker compose -p <name> -f docker-compose.yml -f docker-compose.standalone.yml exec -T app python -m pytest -q
```

## Conventions

- pytest + pytest-asyncio in auto mode (`pyproject.toml`): write `async def test_...`, no marker needed.
- Never hit the API. Drive a turn with the `run` fixture:
  `result, decider_calls = await run([resp(...), ...], lookups=[...], messages=[...], **respond_kwargs)`.
  Decider calls take the scripted responses in order (last repeats); the writer call gets "writer text".
  `lookup_concept` is stubbed to return `lookups` in order; `web_research` is the `research` mock.
- Builders come from conftest: `from conftest import resp, text_block, tool_use, single, bucket, route, ...`.
  Keep `tests/conftest.py` the only conftest so that import stays unambiguous.
- Tunables: `patch.object(settings, "NAME", value)` on `app.pipeline.settings`. Never
  `from app.pipeline.settings import NAME` -- the code reads them at call time.
- Refusal and note wording: compare against `app.pipeline.notes.*`, never a copied string.
- Routes are the real `seed/routes.json`; use `route(name)` / `queue_setting(...)` for expected steps.
- Plain `unittest.mock.patch` / `patch.object` / `patch.dict`; no other mocking libs.
- Test basenames are unique across folders (no `__init__.py` files).
- Docstring: one line on what the test proves. History belongs in the feature log (`../../README.md`, Logs).

## Adding a test

1. Find the file whose one-sentence concern fits; otherwise add `tests/<area>/test_<concern>.py`
   with a one-line module docstring.
2. Pipeline behaviour: take the `run` fixture and script the decider with `resp(tool_use(...))` /
   `resp(text_block(...))`. Pure helpers: import and call them directly.
3. Assert with a message that shows the actual value (`assert x == y, x`).
4. Run the file, then the whole suite, in docker.
