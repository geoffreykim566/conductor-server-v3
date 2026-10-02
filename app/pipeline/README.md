# pipeline

One turn of conversation. `respond(messages, ...)` takes the full history ending in the new user message and returns a `Result`: the reply text, the card's steps, the trace of tool calls, and the new history for the client to send back next turn. Nothing is stored server-side; history is the only state.

## Files

| File | What |
|---|---|
| `respond.py` | The turn loop (`run_turn`) and the auto-run decision (`respond`) |
| `dispatch.py` | Runs one tool call against the card: dup/endorse/replace/cap/commit-to-one rules, lookup auto-attach |
| `executors.py` | Tool name -> executor; status line while a tool runs |
| `turn_state.py` | `TurnState` (trace, card steps, spans, lookup count) and rebuilding it on resume |
| `cards.py` | Action identity keys, bucket candidates, span bookkeeping, `card_descriptions` |
| `context.py` | System prompt blocks, live-state block, screenshot splice, cache breakpoints |
| `heuristics.py` | Reads of the user's own text: question? undo? greeting/closer? forced first lookup? |
| `backfill.py` | Re-shows an earlier card on "show me that again" |
| `confidence.py` | Confidence tier from the trace; research sources |
| `writer.py` | Builds the writer's facts block and runs the writer |
| `model_io.py` | The only Anthropic API call site; response -> request-shape dicts |
| `transcript.py` | Turn boundary and pending tool calls in a message list |
| `settings.py` | Tunables (`MAX_ITERATIONS`, lookup cap, ...) |
| `notes.py` | Fixed text handed to the decider (refusals, nudges, card notes) |
| `result.py` | `Result` dataclass |

## How a turn flows

1. **Context.** The system prompt is the static decider prompt plus a live-state block (the client's AX dump, or the battery's `ax_fixture`). The screenshots are spliced into this turn's user message for each call only.
2. **Decider loop** (≤ `MAX_ITERATIONS`). The decider (`config.MODEL`) is called with the tools.
   - On the first call of a turn that needs grounding, `tool_choice: any` forces *some* tool.
   - Each tool call goes through `dispatch.handle_tool_use`, which updates the card.
   - The text the decider writes in every iteration is kept, not just in the last one.
3. **The turn ends** when the decider stops calling tools, calls `ask_clarifying_question`, or runs out of iterations. Running out gets one tools-off "answer now" call, so a turn is never empty.
4. **Writer.** A cheaper model (`config.WRITER_MODEL`) phrases the reply from a facts block built in code (`writer._facts`). Only the writer streams to the client.
5. **Auto-run.** `respond` decides whether the card may run without the user pressing Run.

## Conventions

- **Read tunables as `settings.X` at call time**, never `from app.pipeline.settings import X`. Tests and `evals/latency_variants.py` patch them on the module.
- **Call the model only through `model_io.call_model`**, by module attribute. Tests patch `model_io._client.messages.create`, and the latency harness swaps `model_io.call_model`.
- **Executors are looked up in `executors.EXECUTORS`** (tests `patch.dict` it). `research` and `tools` are reached as module attributes for the same reason.
- **Code decides, prompts don't.** Anything that has to hold every time is enforced here, not requested in a prompt: the lookup cap, commit-to-one, auto-run, "nothing has run yet", the confidence tier. Prompt-only versions of these held ~40–50% of the time.

## Quirks & why

### One card per turn
- Every action, whether the model called it or a lookup queued it, is a step on the same card, in call order. The card holds at most `MAX_ATTACHES_PER_TURN` steps.
- **Commit-to-one:** a second candidate from the *same* problem bucket is an alternative fix, not a second request, so it's refused with `FALLBACK_REFUSAL`. Before this, the last candidate silently won and contradicted the reply.
- **Auto-attach:** a lookup that lands on ONE solution at *strong* confidence queues that solution's action itself, because the model often explained the fix and never made the second call. The result records it as `on_card`; that is the only record, which is why `turn_state` and `backfill` read it.
- **Endorsement:** if the model then calls the same action (same target, by `endorse_key`), its version replaces the queued step and the step counts as model-called.
- **Re-visited setting:** a second `open_setting` for the same route replaces the first. The later call is the correction the prose follows.
- **Pick-or-ask:** a moderate single match, or a multi-candidate bucket, doesn't auto-attach. If the decider would end on prose with actionable candidates and an empty card, it's re-asked once with a tool call required (`PICK_NUDGE`). The confidence bands were tuned on a much smaller KB, so a correct hit often reads moderate, and a moderate *collision* ("frame rate" -> sample rate) is exactly why this doesn't auto-attach.

### Lookup cap
`LOOKUP_ATTEMPT_LIMIT` is enforced in code. Without it, an absent topic returned a *different* wrong match on every rephrasing, and the model retried until it ran out of iterations with no answer. `COUNT_ALL_LOOKUPS=False` (count only unproductive lookups) was measured and reverted: off-KB turns landed in a new bucket each time, so the cap never bound.

### Forced first call (`heuristics.needs_first_lookup`)
Without it, the model answered real questions from memory with zero grounding. Exemptions:
- **Re-asks**, which would starve backfill.
- **Closers** like "thanks": nothing to look up, and forcing a call makes the model invent a query.
- **Whole-project destructive requests**, which should be declined with no tool call.
- **Bare greetings**, matched against the *whole* message, because "hi" is inside "this".

`FORCE_FIRST_LOOKUP=False` forces "any tool" so a direct command can go straight to an action.

### Heuristics phrase lists
- They're hand-curated and deliberately narrow.
- Bare "again" and "forgot" are excluded ("thanks again, that fixed it").
- "cant find it" was removed after it matched 7 battery turns.
- The destructive list is project-scoped only, because "delete everything on this track" is undo-able.
- **Re-sweep the full battery after changing any list.** Checking only the scenario you were fixing has given false confidence more than once.

### Auto-run (`respond.respond`)
A card runs without Run only when all of these hold:
- the user's message is a direct instruction (not question-shaped, not bulk);
- no lookup happened this turn (a fix inferred from a diagnosis always waits);
- no route on the card has `wait_for_run`.

This exists because a "bypass control surfaces" card off a moderate match once auto-ran on a turn whose own reply said "I don't have a verified fix".

### Writer
- **Why there's a separate writer.** The decider leaked its own vocabulary into replies ("that result isn't relevant"), and a word-ban list couldn't close the leak. The writer only *phrases*: it never sees tools, results, or confidence.
- **The writer's facts block is built in code:**
  - The card's contents come from `card_descriptions(trace)`, not the decider's prose. On a lookup-queued card the decider may never mention the step.
  - "None of this has run yet." The writer once said "Loaded X" before anything ran, and the next turn believed it.
  - The undo-request decline. Typed undo is denied for now; the card's Revert button does it.
  - "You're looking at their screen." The writer never sees the screenshots and would otherwise say it can't.
  - Live state is appended last so nothing can drop it. It's framed as partial: authoritative for what it lists, silent on the rest.
- **The moderate-turn hedge is off** (`HEDGE_MODERATE_TURNS`). It fired on correct observations like "which tracks are muted".

### Confidence tier
The tier is computed from the trace, deterministically:
- a clarifying question -> generic;
- a card -> strong;
- successful research -> research;
- no tools called -> generic;
- otherwise strong / moderate / generic from the lookups.

A weak hit counts as no attempt. A strong *diagnosis* whose solutions carry no action doesn't count as strong. The client badge is disabled (see `app/api/README.md`); the tier is still used by the battery and the logs. **Known gap:** the tier certifies the destination, not every claim the prose adds on top of it.

### Backfill
If the user asks to see something again and the model answers from memory with no tool call, the earlier route is re-attached. It's gated on re-ask language in the *user's* message: matching the destination name in the reply misfired when a topic recurred for other reasons. Zero or several candidates are left alone. Hostile phrasings are covered in `tests/pipeline/test_backfill.py`.

### Context & caching
- **Three cache breakpoints:** the static prompt, the live-state block, and this turn's user message. Iterations 2+ read the prefix from cache (~9k -> <2.5k uncached tokens per call on a screenshot turn).
- **Live state is its own block.** While it was concatenated into the prompt, the static prompt never hit the cache across turns.
- **Screenshots and `cache_control` only ever go on the per-call copy** (`TurnContext.for_call`), never into `msgs`. `msgs` is the next turn's history, and the API validator rejects images and extra keys there.

### History round-trip
- `model_io.serialize_content` turns SDK blocks into plain dicts. Raw response objects carry fields the request schema rejects, which broke turn 2.
- Thinking blocks are dropped: the validator doesn't accept them, and this non-interleaved loop doesn't need them.

### Research approval
- With `research_confirm`, a `web_research` call parks the turn *before any tool in that batch runs*. The client then resumes with the whole batch, so every `tool_use` gets exactly one `tool_result`.
- On resume, `turn_state.restore_turn_state` replays the parked tail, so an earlier attach survives.
- On deny, the call gets `RESEARCH_DECLINED` in place of a result.

## Adding things

- **A tool:** add its schema and executor in `app/tools/`, register it in `executors.EXECUTORS`, and add a status line if it's slow. If it queues steps, add it to `tools.ACTION_TOOLS` plus an `endorse_key` / `card_descriptions` case.
- **A rule about the user's message:** add it to `heuristics.py` with a unit test, then sweep the battery.
- **A tunable:** add it to `settings.py` and read it as `settings.X`.
