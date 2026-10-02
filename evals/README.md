# evals

Model-in-the-loop evaluation. These hit the real API and the real KB, unlike `tests/`, which are mocked unit tests. Not part of the server image: `evals/` is bind-mounted into the container by `docker-compose.yml`. Output goes to `test_runs/<date>/` (gitignored).

## Files

| Path | What |
|---|---|
| `battery/runner.py` | Runs `scenarios/battery.json` through `respond()` and prints transcripts and verdicts |
| `battery/grader.py` | Turns a trace into an outcome and grades it against the scenario's `expect` block |
| `scenarios/battery.json` | The scenarios: `name`, `note`, `turns` (`text`, optional `ax_fixture` / `ax_state` / `expect`), `expect` |
| `latency_variants.py` | A/B harness: the same scenarios under several `pipeline.settings` overrides, interleaved |
| `drive_turn.py` | One turn at a time, with state kept in a file between calls: for reactive multi-turn probing |

## Running the battery

Rebuild and reseed first if `app/` or `seed/` changed, or you're grading the old code.

```bash
mkdir -p test_runs/$(date +%F)
docker compose exec -T app python -m evals.battery.runner > test_runs/$(date +%F)/run.log 2>&1   # all 113, ~35 min serial
docker compose exec -T app python -m evals.battery.runner name1 name2                            # some
docker compose exec -T app python -m evals.battery.runner --runs 3 name1                         # flakiness check
```

**Shard it 3-way in parallel** (~15 min). Rate limits are not a constraint:

```bash
for i in 0 1 2; do
  NAMES=$(python3 -c "import json;s=json.load(open('evals/scenarios/battery.json'));print(' '.join(x['name'] for x in s[$i::3]))")
  docker compose exec -T app python -m evals.battery.runner $NAMES > test_runs/$(date +%F)/run_shard$i.log 2>&1 &
done; wait
grep -c '\[PASS\]' test_runs/$(date +%F)/run_shard*.log; grep -h -B1 '\[FAIL\]' test_runs/$(date +%F)/run_shard*.log
```

Compare against a baseline run of the same scenarios, never against memory. Pass counts move by ±2 between identical runs, so read *which* assertions changed and read the answers themselves.

## Expect keys (`grader.grade`)

| Key | Passes when |
|---|---|
| `walkthrough_destination` | the route(s) on the card include it. `null` = nothing attached; a list = any of them, and `null` in the list means "nothing" is acceptable |
| `walkthrough_destination_absent` | none of these routes are on the card |
| `walkthrough_solution` | the KB solution behind the attached action (same forms as above) |
| `action_calls` | these action calls happened, in order, as a subsequence. `{"tool", "args": {k: substring}, "ok", "args_absent"}`. `null` = no action calls at all |
| `setting_calls` / `forbidden_setting_calls` | the same for `open_setting` calls (route + dropdown value) |
| `no_action_calls`, `no_refused_actions` | bools |
| `no_completion_claims` | the reply doesn't say a queued action already happened ("is set", "I've loaded"). Substring-brittle |
| `response_contains` / `_any` / `response_not_contains` | case-insensitive substrings of the reply |
| `first_tool` | name, or a list of acceptable names |
| anything else | compared for equality against the outcome: `match`, `resolved_problem`, `no_tool_calls`, `web_research_called`, `confidence_tier`, `sources_present`, `lookup_count`, `auto_run` |

A top-level `expect` grades the final turn. A per-turn `expect` grades that turn's own trace.

## Quirks & why

- **`expect` is written before running, never fitted to what happened.** A note that describes behaviour without asserting it is untested; add the assert.
- **The grader keeps its own `ACTION_TOOLS`** rather than importing `app.tools`, so it can grade any build.
- **`match` / `resolved_problem` grade the first lookup**, since later lookups by exact name are a legitimate second step.
- **Known grader issues:** `no_completion_claims` flags descriptive "is set", and `match` checks can fail correct final answers. Read the FAIL before believing it.
- **Code-guaranteed scenarios** (e.g. the toggle gate refusing an attach) pass whatever the model does. They're regression tests for that code path, not model judgment.
- **Latency A/B must record full answers.** Use the same prompts and interleave the variants. n=2 per cell can't judge quality; the battery can. Add a variant in `VARIANTS` as `{"SETTING_NAME": value}`.

## Adding a scenario

Add it to `scenarios/battery.json` with a `note` (what it tests and why) and an `expect` written from the desired behaviour. Run it alone 3× (`--runs 3`) to see whether it's stable before relying on it.
