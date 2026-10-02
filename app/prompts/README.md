# prompts

System prompts, one file per model role. Each file is a single string constant; `__init__.py` re-exports both.

| File | Constant | Used by |
|---|---|---|
| `decider.py` | `SYSTEM_PROMPT` | `pipeline/context.py`, as cache breakpoint 1 |
| `writer.py` | `WRITER_SYSTEM_PROMPT` | `pipeline/writer.py`, with the turn's facts appended |

## Quirks & why

- **The writer prompt never mentions tools, the KB, confidence, or any of the machinery.** It exists to stop the decider's vocabulary leaking into replies, which a word-ban list in the decider prompt couldn't fully close. The writer gets already-decided facts to phrase, never raw tool results.
- **Prompt-only rules are requests, not guarantees.** If something must hold every time (a cap, a refusal, "nothing has run yet"), enforce it in `pipeline/` code and keep the prompt line as guidance only. Prompt-only versions have held roughly 40–50% of the time.
- **The decider prompt is cached.** Keep it byte-identical across turns. Anything per-turn goes in its own system block (`pipeline/context.py`) so the cache still hits.

## Changing a prompt

Edit, rebuild, then run the full battery and read the answers, not just the pass count. Prompt changes move unrelated scenarios.
