"""v3 pipeline: query + context -> tool-calling loop -> response.

respond() takes the full running conversation (already ending with the newest
user turn) and returns the full conversation after this turn completes, so the
caller (the test harness) just appends the next user turn for the next call.
History is the only state — nothing is pinned server-side between turns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from anthropic import AsyncAnthropic

from app import tools
from app.config import CENTRAL_ANTHROPIC_KEY, MAX_TOKENS, MODEL
from app.prompt import SYSTEM_PROMPT

OnChunk = Callable[[str], Awaitable[None]]

MAX_ITERATIONS = 6

# Hard, code-enforced cap on lookup_concept calls within one turn — a prompt
# instruction to "stop retrying" is a request, not a guarantee. Found live
# 2026-07-30: a genuinely absent topic ("wheres the monitor button") kept
# returning a *different* wrong match on every retried phrasing, so the model
# never hit a clean, repeated "nothing here" signal and just kept trying new
# wording until it hit the overall iteration ceiling with no answer at all.
# This intervenes explicitly before that happens, distinct from the overall
# MAX_ITERATIONS safety net below (which covers any tool, not just this one).
LOOKUP_ATTEMPT_LIMIT = 4

_client = AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY)

_EXECUTORS = {
    "lookup_concept": lambda inp, fixture: tools.lookup_concept(inp["problem"]),
    "get_walkthrough": lambda inp, fixture: tools.get_walkthrough(inp["solution"], ax_fixture=fixture),
}


@dataclass
class Result:
    text: str
    walkthrough_steps: list | None
    trace: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)


# Bare "again" and "forgot" deliberately excluded, and "cant find it" removed
# after initially being added -- found via two rounds of Fable review,
# 2026-08-06. "again" collides with ordinary closing phrasing ("thanks again,
# that fixed it!"); "forgot" is the identical shape ("oh forgot to say -- that
# fixed it, thanks!"); "cant find it" (added when apostrophes were first
# stripped from both sides of the match, so "can't"/"cant" would both match)
# turned out to appear in 7 different battery turns once actually swept
# against the full battery, not just the one it was meant for -- a live
# comment here once claimed "only this gap's scenario matches" before that
# sweep was actually run; that was the exact circular-validation mistake this
# feature has repeatedly made. Every phrase kept here is a specific, low-
# frequency request to see something again, not a common word or filler.
_REASK_SIGNALS = (
    "remind me", "reminder", "one more time", "show me that",
    "show that again", "where was", "where is that", "closed the window",
    "lost the window",
)


def _backfill_walkthrough(messages: list[dict], text: str) -> dict | None:
    """Commit-to-one is scoped per turn, so a turn that attaches nothing normally
    leaves the user with prose only -- correct when the model is just answering a
    new question, but wrong when the user asked to *see* a destination again and
    the model answered from memory instead of re-calling get_walkthrough (found
    live 2026-08-05, multiturn_same_destination_revisited_new_turn: inconsistent,
    sometimes recalls the tool, sometimes doesn't). Deterministic backfill instead
    of another prompt instruction -- prompt-only attempts at behaviors like this
    have repeatedly not held (hedge-on-navigation, toggle-gate force flag).

    Gated on the CURRENT user turn's own message containing explicit re-ask
    language (_REASK_SIGNALS), not on the model's response text alone. First
    version matched purely on the destination name appearing anywhere in the
    response; found live (2026-08-06) that's unreliable -- a turn can legitimately
    discuss the same general topic (e.g. "sample rate") for an unrelated reason
    (pointing the user at a different, non-Logic screen) without ever being asked
    to reattach anything, and the destination's own name is exactly the word most
    likely to recur regardless of intent. Requiring zero tool calls (this turn did
    no fresh work at all -- pure memory recall) plus explicit re-ask language in
    what the user actually typed is a direct signal of intent instead of an
    inference from response-content overlap.

    Checking this against the battery's own scenario texts is a weak signal, not
    real validation -- it only proves the gate doesn't misfire on the handful of
    conversations it was tuned against, not on real phrasing in general (this was
    stated here once before without actually re-running the sweep after a signal
    list change, and turned out to be false at the time -- re-verify by sweeping
    the battery, don't just trust this comment, if this list changes again). Real
    coverage of hostile phrasing (e.g. "thanks again, that fixed it" on a closing
    turn) lives in test_backfill.py's unit tests instead, which don't depend on
    guessing what a live model happens to say.

    Scans prior turns' message history (not this turn's own -- there's nothing to
    find there if this turn made no tool calls) for successful get_walkthrough
    attaches, identified by result shape (attached+destination+steps together, not
    by tool name -- avoids needing to correlate tool_use_id across blocks). If the
    re-ask check passes, reattaches the one destination whose name appears in this
    turn's response text (now just disambiguating *which* prior destination among
    possibly several, not deciding *whether* to backfill at all); zero or multiple
    matches are left alone rather than guessed.
    """
    if not messages or messages[-1].get("role") != "user":
        return None
    user_text = messages[-1].get("content")
    if not isinstance(user_text, str):
        return None
    normalized = user_text.lower().replace("'", "")
    if not any(sig in normalized for sig in _REASK_SIGNALS):
        return None

    attached_by_destination: dict[str, list] = {}
    for msg in messages:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            try:
                result = json.loads(block["content"])
            except (TypeError, ValueError):
                continue
            if result.get("attached") and result.get("destination") and result.get("steps"):
                attached_by_destination[result["destination"]] = result["steps"]

    matches = [dest for dest in attached_by_destination if dest.lower() in text.lower()]
    if len(matches) != 1:
        return None
    return {"destination": matches[0], "steps": attached_by_destination[matches[0]]}


_HEDGE_PREFIX = (
    "Note: I couldn't verify this against confirmed Logic Pro documentation, so treat "
    "the following as general guidance rather than an exact path.\n\n"
)


def _needs_hedge(trace: list[dict], walkthrough_steps: list | None) -> bool:
    """Deterministic backstop -- prompt-only hedge instructions held ~40-50% of the time
    across live testing, 0% on broad/indirect questions specifically (2026-08-11).
    Fires when this turn made a real attempt (a tool call) but never reached strong
    grounding. Does NOT fire on zero tool calls -- can't distinguish a plain
    conversational close from the separate, not-yet-fixed tool-skip gap."""
    if walkthrough_steps is not None:
        return False
    if not trace:
        return False
    return not any(
        c["tool"] == "lookup_concept" and c["output"].get("match_confidence") == "strong"
        for c in trace
    )


def _is_grounded(trace: list[dict], walkthrough_steps: list | None) -> bool:
    """True once this turn can no longer end up needing a hedge, regardless of what
    happens later -- a walkthrough attach or a strong lookup hit, both permanent
    once trace/walkthrough_steps get set (trace only grows; commit-to-one means
    walkthrough_steps is set at most once). Used to decide, per iteration, whether
    that iteration's text is safe to stream live or needs buffering (see
    _call_model's on_chunk callers in respond()) -- distinct from _needs_hedge,
    which answers "does the turn need a hedge *right now*" and treats an
    empty/inconclusive trace as no (silently, not permanently) safe."""
    if walkthrough_steps is not None:
        return True
    return any(
        c["tool"] == "lookup_concept" and c["output"].get("match_confidence") == "strong"
        for c in trace
    )


class _ChunkBuffer:
    """Collects streamed text instead of forwarding it live, for the stretch of a
    turn where grounding isn't yet decided -- see the streaming-hedge gap logged
    2026-08-11/12: a real client sees raw model text the instant it's generated,
    before the turn's grounding outcome (and therefore whether to hedge) is known.
    Held chunks are released via flush() once that's decided, hedge-prefixed first
    if needed, so the hedge always precedes the text it's warning about instead of
    arriving after the client already rendered it live."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    async def collect(self, text: str) -> None:
        self.chunks.append(text)

    async def flush(self, on_chunk: OnChunk) -> None:
        for chunk in self.chunks:
            await on_chunk(chunk)


async def _finalize_answer(
    trace: list[dict],
    walkthrough_steps: list | None,
    text: str,
    buffer: "_ChunkBuffer | None",
    on_chunk: OnChunk | None,
) -> str:
    """Called once this turn's final answer text is fully known (either the
    no-more-tool-calls return or the MAX_ITERATIONS safety net). Applies the
    hedge to the returned text as before, and -- new -- if that final iteration's
    stream was buffered (grounding wasn't yet decided when it started), releases
    it now: hedge chunk first if needed, then the buffered text, so a streaming
    client sees the same hedge-before-answer ordering non-streaming callers get
    from Result.text alone."""
    hedge = _needs_hedge(trace, walkthrough_steps)
    if hedge:
        text = _HEDGE_PREFIX + text
    if buffer is not None:
        if hedge:
            await on_chunk(_HEDGE_PREFIX)
        await buffer.flush(on_chunk)
    return text


def _serialize_content(content: list) -> list[dict]:
    """SDK response content blocks -> plain request-shape dicts.

    resp.content items are response objects (TextBlock, ToolUseBlock, ...)
    carrying response-only fields (e.g. TextBlock.parsed_output) that the
    request-side schema doesn't accept. Appending them raw into msgs works
    fine *within* a single respond() call (never re-serialized before the
    next .create()), but Result.messages leaves the process as JSON for the
    client to round-trip as the next turn's history -- at that point the
    extra field is a literal dict key, and the API rejects it outright
    ("Extra inputs are not permitted") on the following request. Found live
    2026-08-08: turn 1 (fresh) succeeds, turn 2 ("show me that again",
    replaying turn 1's text block from history) 400s. Normalizing here,
    once, at the source keeps msgs safe for both same-turn reuse and
    cross-request replay.
    """
    out = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        else:
            out.append(b.model_dump())
    return out


def _usage_dict(u) -> dict:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


async def _call_model(system: list[dict], tools_param: list[dict] | None, msgs: list[dict], on_chunk: OnChunk | None):
    """One messages.create call, or the streamed equivalent when on_chunk is given.

    Streaming path forwards every text delta from every iteration (not just the
    final answer) — a tool-calling turn's own text_stream only ever carries its
    text blocks, never tool_use input, so a turn that also calls a tool doesn't
    accidentally leak partial JSON to the client. When on_chunk is None (every
    caller except api.py — the battery runner, drive_turn, and the mocked unit
    tests) this still calls _client.messages.create directly, unchanged, so
    existing mocks (which patch that exact method) keep working.
    """
    kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=msgs)
    if tools_param is not None:
        kwargs["tools"] = tools_param
    if on_chunk is None:
        return await _client.messages.create(**kwargs)
    async with _client.messages.stream(**kwargs) as stream:
        async for text in stream.text_stream:
            await on_chunk(text)
        return await stream.get_final_message()


async def respond(
    messages: list[dict],
    ax_fixture: dict | None = None,
    on_chunk: OnChunk | None = None,
) -> Result:
    msgs = list(messages)
    trace: list[dict] = []
    usage: list[dict] = []
    walkthrough_steps: list | None = None
    lookup_attempts = 0

    # Pushed, not pulled: live state (when known for this turn) is handed to the
    # model automatically rather than waiting on it to decide to call a tool for
    # it -- a decision point it's already been observed skipping under real
    # conditions (found live 2026-08-04, the monitor-button hallucination).
    system_text = SYSTEM_PROMPT
    if ax_fixture:
        state_lines = "\n".join(f"- {k}: {v}" for k, v in ax_fixture.items())
        system_text += (
            "\n\n## Live state for this turn\n\n"
            "Read directly from the running Logic Pro project via the Accessibility "
            "API -- ground truth, not something the user said or you inferred. This "
            "outranks stated claims, seed_weight, and anything read from a screenshot "
            "when they conflict.\n\n"
            f"{state_lines}"
        )
    # Cached as its own block: identical across every iteration of this turn's
    # loop (system+tools resent unchanged on each one -- measured 2026-08-05:
    # 66 calls, 207,869 uncached input tokens across a 24-scenario battery).
    # The safety-net call below appends its own instruction as a second,
    # uncached block so it still hits this same cache entry.
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

    for _ in range(MAX_ITERATIONS):
        # Grounding not yet decided -> this iteration might turn out to be the
        # final answer, so buffer it instead of streaming live (see
        # _ChunkBuffer). Once grounded, it can never become un-grounded again
        # this turn, so later iterations stream straight through unaffected.
        buffer = None
        iter_on_chunk = on_chunk
        if on_chunk is not None and not _is_grounded(trace, walkthrough_steps):
            buffer = _ChunkBuffer()
            iter_on_chunk = buffer.collect

        resp = await _call_model(system, tools.TOOLS, msgs, iter_on_chunk)
        usage.append(_usage_dict(resp.usage))
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text")
            msgs.append({"role": "assistant", "content": _serialize_content(resp.content)})
            if not trace:
                backfilled = _backfill_walkthrough(messages, text)
                if backfilled:
                    walkthrough_steps = backfilled["steps"]
                    trace.append({
                        "tool": "get_walkthrough",
                        "input": {"solution": None},
                        "output": {
                            "attached": True,
                            "destination": backfilled["destination"],
                            "steps": backfilled["steps"],
                            "backfilled": True,
                        },
                    })
            text = await _finalize_answer(trace, walkthrough_steps, text, buffer, on_chunk)
            return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs, usage=usage)

        # Not the final answer -- this iteration's text (if any) carries no hedge
        # risk on its own, so release any buffered chunks immediately rather than
        # holding them until the turn ends.
        if buffer is not None:
            await buffer.flush(on_chunk)

        msgs.append({"role": "assistant", "content": _serialize_content(resp.content)})
        tool_results = []
        for tu in tool_uses:
            if tu.name == "lookup_concept":
                lookup_attempts += 1
            if tu.name == "lookup_concept" and lookup_attempts > LOOKUP_ATTEMPT_LIMIT:
                result = {
                    "error": (
                        f"Too many lookup attempts ({lookup_attempts}) without a clear answer. "
                        "Stop searching now — answer from general Logic Pro knowledge if you're "
                        "genuinely confident, or tell the user you don't have a verified answer "
                        "for this. Do not call lookup_concept again this turn."
                    )
                }
            elif tu.name == "get_walkthrough" and walkthrough_steps is not None:
                # Commit-to-one: only the first successful attach in a turn
                # reaches the user (found live 2026-08-05 -- multiple attaches
                # in one turn meant the *last* one silently won, contradicting
                # whichever candidate the response text actually led with).
                result = {
                    "attached": False,
                    "reason": (
                        "a walkthrough was already attached this turn -- only one attaches per "
                        "turn. Mention a fallback candidate in prose instead, and only attach it "
                        "if the user comes back and says the first one didn't work."
                    ),
                }
            else:
                executor = _EXECUTORS.get(tu.name)
                result = {"error": f"unknown tool {tu.name!r}"} if executor is None else await executor(tu.input, ax_fixture)
            trace.append({"tool": tu.name, "input": tu.input, "output": result})
            if tu.name == "get_walkthrough" and result.get("attached"):
                walkthrough_steps = result.get("steps")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })
        msgs.append({"role": "user", "content": tool_results})

    # Safety net: never return a truly empty response, regardless of why the
    # loop didn't converge on its own. Force one final tools-off call so the
    # model must synthesize whatever it already learned into a real answer.
    buffer = None
    final_on_chunk = on_chunk
    if on_chunk is not None and not _is_grounded(trace, walkthrough_steps):
        buffer = _ChunkBuffer()
        final_on_chunk = buffer.collect
    final = await _call_model(
        system + [{
            "type": "text",
            "text": "\n\nAnswer now with your best available information — no more tool calls.",
        }],
        None,
        msgs,
        final_on_chunk,
    )
    usage.append(_usage_dict(final.usage))
    text = "".join(b.text for b in final.content if b.type == "text")
    msgs.append({"role": "assistant", "content": _serialize_content(final.content)})
    # No backfill check here (unlike the early-return path above): reaching this
    # point requires every one of MAX_ITERATIONS loop passes to have had at least
    # one tool call, each of which unconditionally appends to trace -- so trace
    # can never be empty here, and _backfill_walkthrough only ever fires on a
    # turn with zero tool calls. A backfill check here was dead code (found via
    # Fable review, 2026-08-06) and has been removed rather than left in place.
    text = await _finalize_answer(trace, walkthrough_steps, text, buffer, on_chunk)
    return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs, usage=usage)
