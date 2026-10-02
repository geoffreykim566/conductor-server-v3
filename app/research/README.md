# research

`web_research(query)`: the executor for the `web_research` tool. It's one nested `RESEARCH_MODEL` + `web_search` call (up to 3 searches, up to 2 `pause_turn` continuations) that returns `findings`, `sources`, `source_tier`, and `_usage`.

| File | What |
|---|---|
| `search.py` | The call, timeouts, caps, timing logs (`[web_research_timing]`) |
| `parse.py` | Confidence line, findings text, and sources out of the response |

## How it's wired

- **The decider decides when to research.** The tool description is the only trigger. Firing it on every KB miss was rejected as the costlier, more eager option.
- **Research on a KB miss is the intended fallback** (user call), so a turn that escalates to research when nothing in the KB fits isn't graded as a failure.
- **`_usage`** is popped by `pipeline/dispatch.py`. It counts toward the turn's spend but never reaches the model.
- **The "research" tier:** a successful call earns it (`pipeline/confidence.py`), trusted like "strong". A timed-out or empty call doesn't. Sources reach the client as chips.
- **With `research_confirm`,** the turn parks for user approval before this runs (`pipeline/README.md`, Research approval).

## Quirks & known gaps

- **Results are capped** (6,000 chars of findings, 10 sources). The tool_result stays in round-tripped history, and an uncapped one 422'd every later request in that conversation.
- **Timeout is 150s** (`RESEARCH_CALL_TIMEOUT_S`). There's no retry on timeout, so one slow call just fails.
- **`source_tier` is computed but unused.** The fine-grained tier (confirmed / community / genre-inference) never reaches the confidence tier.
- **The `parse_confidence` regex misses bold `**CONFIDENCE:**`** and silently defaults to `commonly_believed`. Harmless while `source_tier` is unused.
- **Mixed turns:** a mixed KB + research turn can state web-sourced specifics under a "strong" tier inherited from an unrelated card. Not fixed.
