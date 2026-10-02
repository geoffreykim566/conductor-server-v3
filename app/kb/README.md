# kb

The knowledge base behind `lookup_concept`: Voyage embeddings, and loading `seed/problems.json` into Postgres (`problems`, `solutions`, `problem_solutions`). Lookup itself lives in `app/tools/lookup.py`.

## Files

| File | What |
|---|---|
| `embed.py` | `embed(texts, input_type)`: the Voyage call, with a minimum gap between calls |
| `load.py` | Seeds/reseeds the KB: validates routes, embeds, upserts by name, deletes orphans |

## Running

```bash
docker compose exec app python -m app.kb.load
```

Run it after any edit to `seed/problems.json` (after rebuilding, since `seed/` is baked into the image), and once on any fresh DB volume.

## The model

- A **problem** is a bucket ("song sounds slowed"). It has aliases and a note, and it links to candidate **solutions** with a `seed_weight` (a population prior) and a `distinguisher` (what evidence picks this one).
- A solution can sit in several buckets.
- A solution with a fix carries its `action` (`{"open_setting": "route"}` / `{"open_plugin": {...}}`, or a list of them). Paths are never stored here; they live once in `seed/routes.json`.

## Quirks & why

- **What gets embedded:**
  - Problems embed name + aliases.
  - *Destination* solutions embed name + curated aliases.
  - *Diagnosis* solutions embed their name only.

  This was settled by A/B:
  - Not embedding diagnoses at all lost every mechanism-named query to an adjacent bucket.
  - Embedding prose caused collisions ("wheres buffer size" matched a latency diagnosis whose cause text mentioned "buffer") and never caught anything the aliases didn't.
- **Aliases collide.** Before adding one, check it isn't the natural wording of a *different* bucket. Run the battery after any seed change.
- **Rate limits:** `embed.py` keeps a 3s floor between calls and `tools/lookup.py` retries with backoff. Real limits are high now (~4000 req/min). A 429 only shows as a `[voyage_429]` log line; don't blame rate limits without it.
- **Vector lookup is on its way out.** The `v042` branch moves the KB into the prompt and removes vector search. When that lands, this folder and `tools/lookup.py` shrink or go.
