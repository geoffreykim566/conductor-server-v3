# kb

The knowledge base lives **in the decider's prompt**. At import, `seed/problems.json` and `seed/routes.json` are loaded and the whole KB is rendered into the cached system prompt (`KB_TEXT`, ~20k tokens). The model reads all of it and records the entries it relied on with the `cite_kb` tool. Nothing is retrieved, embedded or stored in the DB.

## Files

| File | What |
|---|---|
| `data.py` | Loads the seed: `SOLUTIONS`, `PROBLEMS`, `ENTRY_NAMES` (cite_kb's enum), `BUCKETS_OF`, routes split into runnable `ROUTES` and `REFERENCES` |
| `paths.py` | A route's path as text (`route_path_text`) and a solution's `Where:` line |
| `render.py` | `KB_TEXT`: problems with ranked candidates, then solutions with action / content / Where |
| `cite.py` | `cite(entries)`: cite_kb's executor |

## The model

- A **problem** is a bucket ("song sounds slowed"). It has aliases and a note, and it links to candidate **solutions** with a `seed_weight` (a population prior) and a `distinguisher` (what evidence picks this one).
- A solution can sit in several buckets.
- A solution with a fix carries its `action`: `{"open_setting": "route"}`, `{"open_setting": {"name", "value"}}`, `{"open_plugin": {...}}`, or a list of these.
- **Menu paths live only in `routes.json`.** A solution's `Where:` line is rendered from its routes and `where` links. KB prose never spells out a path.

## How a citation works

`cite(entries)` resolves the **first** cited entry the way the old vector lookup resolved its top hit, so commit-to-one, backfill and the grader work unchanged:
- A problem, or a solution inside a bucket, returns that bucket with every candidate. A cited bucket wins over the solution's strongest one.
- A standalone solution returns alone.
- The result adds:
  - `cited`: the valid names.
  - `recommended`: the cited solutions with an action, which the pipeline queues if the model didn't call them.
  - A `bucket` on each candidate.
  - An `error` for unknown names.
- A bucket's own name counts as a pick only when that solution is its only cause.

## Quirks & why

- **Why there's no vector search.** Each lookup cost a full extra decider round trip (~2.5 s), and off-KB questions burned up to 4 of them. Correct and wrong matches also overlapped in distance, so no confidence cutoff could separate them. Putting the KB in the prompt took KB turns from a 9.2 s to a 7.0 s median, removed the 20 s+ retry tails, and cut ungrounded menu paths from 10 to 4. The cost: the larger prefix adds ~1 s to every decider call, direct commands included.
- **Citations are always `match_confidence: "strong"`.** The confidence bands are gone; the tier code still reads the field.
- **Reference entries** (`"reference": true` in routes.json) are paths outside Logic or never given steps. The model may state them, but they're never in open_setting's enum and can't be queued.
- **Keep the KB block byte-stable.** It's part of cache breakpoint 1. Anything per-turn goes elsewhere (`pipeline/context.py`).
- **Aliases still matter**, as the model's vocabulary for matching user wording. They cost tokens, though, and dropping them is an open idea.
- **DB tables** (`problems`, `solutions`, `problem_solutions`, `query_log`) and their migrations are left in place but unused. Drop them in a release migration.

## Editing the KB

Edit `seed/problems.json`, rebuild (it's read at import), and run the core battery. There's no seeding step. Every `open_setting` a solution names must be a real route.
