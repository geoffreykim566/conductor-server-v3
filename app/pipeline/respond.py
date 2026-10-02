"""The turn: history + this turn's context in, the reply, its card and the new
history out. Stateless: history is the only state. How a turn flows: README.md."""
import json
import logging

from app import tools
from app.pipeline import model_io, notes, settings
from app.pipeline.backfill import backfill_walkthrough
from app.pipeline.cards import needs_pick
from app.pipeline.confidence import collect_sources
from app.pipeline.context import TurnContext
from app.pipeline.dispatch import handle_tool_use
from app.pipeline.executors import status_for_tools
from app.pipeline.heuristics import auto_run_ok, force_first_call, last_user_text, offered_text
from app.pipeline.model_io import OnChunk, OnStatus
from app.pipeline.result import Result
from app.pipeline.transcript import last_user_text_index, pending_tool_uses
from app.pipeline.turn_state import TurnState, restore_turn_state
from app.pipeline.writer import finalize_answer

log = logging.getLogger(__name__)


async def respond(messages: list[dict], **kwargs) -> Result:
    """run_turn, plus whether the card may auto-run, decided once from the user's
    own message. Only a direct command with no citation behind it auto-runs; routes
    marked wait_for_run always wait (README: Auto-run)."""
    result = await run_turn(messages, **kwargs)
    inferred = result.card_from_lookup or any(
        c["tool"] == "cite_kb" and (c.get("output") or {}).get("cited") for c in result.trace)
    waits = any(c["tool"] == "open_setting" and tools.ROUTES.get((c.get("input") or {}).get("name"), {}).get("wait_for_run")
                for c in result.trace)
    result.auto_run = (not inferred) and not waits and auto_run_ok(messages, result.walkthrough_steps)
    return result


async def _call(ctx: TurnContext, msgs: list[dict], usage: list[dict],
                nudge: str | None = None, tool_choice: dict | None = None):
    system = ctx.system + ([{"type": "text", "text": "\n\n" + nudge}] if nudge else [])
    resp = await model_io.call_model(system, tools.TOOLS, ctx.for_call(msgs), None, tool_choice)
    usage.append(model_io.usage_dict(resp.usage))
    tool_uses = [b for b in resp.content if b.type == "tool_use"]
    return resp, tool_uses, "".join(b.text for b in resp.content if b.type == "text")


async def _decide(i: int, msgs: list[dict], state: TurnState, ctx: TurnContext,
                  force_first: bool, on_status: OnStatus | None, usage: list[dict]):
    """One decider call (non-streaming; only the writer streams), plus at most two
    re-asks: a go-ahead's first call that made no tool call (TOOL_NUDGE), and a
    reply about to end on prose while a cited bucket's candidates map to actions
    (pick-or-ask). Returns (response, tool_uses, this iteration's text)."""
    if on_status:
        await on_status("Thinking…")
    tool_choice = settings.FORCED_TOOL_CHOICE if i == 0 and force_first else None
    resp, tool_uses, iter_text = await _call(ctx, msgs, usage, tool_choice=tool_choice)
    if (i == 0 and force_first and not tool_uses and tool_choice is None
            and settings.RETRY_NO_TOOL_FIRST_CALL):
        resp, tool_uses, iter_text = await _call(ctx, msgs, usage, nudge=notes.TOOL_NUDGE)
    if not tool_uses and not state.pick_forced and needs_pick(state.trace, state.card_steps):
        state.pick_forced = True
        resp, tool_uses, iter_text = await _call(
            ctx, msgs, usage, nudge=notes.PICK_NUDGE, tool_choice=settings.FORCED_TOOL_CHOICE)
    if iter_text:
        state.text_parts.append(iter_text)
    return resp, tool_uses, iter_text


def _all_accepted(tool_results: list[dict]) -> bool:
    for tr in tool_results:
        out = json.loads(tr["content"])
        if "cited" in out:
            if "error" in out or not all(o["output"].get("attached") for o in out.get("on_card") or []):
                return False
        elif not out.get("attached"):
            return False
    return True


def _can_exit_early(tool_uses: list, tool_results: list[dict], iter_text: str, state: TurnState) -> bool:
    """Nothing left to decide: every call this iteration was an accepted action or
    citation, a citing iteration already wrote its explanation, and no pick is
    pending. Skips the prose-only follow-up call (~2.5 s). Any refusal or error
    keeps looping so the decider can react (README: Early exit)."""
    citing = any(tu.name == "cite_kb" for tu in tool_uses)
    return (settings.EARLY_EXIT_ON_ACTION and bool(tool_uses)
            and all(tu.name in tools.ACTION_TOOLS or tu.name == "cite_kb" for tu in tool_uses)
            and _all_accepted(tool_results)
            and bool(iter_text.strip() or not citing)
            and not needs_pick(state.trace, state.card_steps))


async def run_turn(
    messages: list[dict],
    ax_fixture: dict | None = None,
    on_chunk: OnChunk | None = None,
    on_status: OnStatus | None = None,
    screenshots_b64: list[str] | None = None,
    ax_state: str | None = None,
    research_confirm: bool = False,
    resume: str | None = None,
) -> Result:
    """research_confirm: park the turn (Result.pending_research_query) instead of
    running web_research, so the client can ask the user. resume: "allow_research"
    or "deny_research"; `messages` is then the parked transcript and this call
    executes its pending tool calls first (README: Research approval)."""
    msgs = list(messages)
    parked: list | None = None
    research_authorized = resume == "allow_research"
    research_denied = resume == "deny_research"
    if resume:
        # `messages` must end at this turn's user message, as on a fresh call.
        turn_start = last_user_text_index(msgs)
        if turn_start is None:
            raise ValueError("resume: no user turn in history")
        messages = msgs[: turn_start + 1]
        parked = pending_tool_uses(msgs)
        if not parked:
            raise ValueError("resume: history does not end in a pending tool call")
    tools.TURN_USER_TEXT.set(last_user_text(messages))
    tools.TURN_OFFERED_TEXT.set(offered_text(messages))

    ctx = TurnContext.build(msgs, screenshots_b64, ax_fixture, ax_state)
    usage: list[dict] = []
    state = restore_turn_state(msgs[len(messages):-1]) if resume else TurnState()
    force_first = force_first_call(messages)

    async def finish(is_clarifying_question: bool = False) -> Result:
        text, tier = await finalize_answer(
            state.trace, state.steps, "\n\n".join(state.text_parts), messages, on_chunk, usage,
            is_clarifying_question=is_clarifying_question, live_state=ctx.live_state,
            had_screenshots=ctx.had_screenshots, has_actions=bool(state.card_steps),
        )
        return Result(text=text, walkthrough_steps=state.steps,
                      trace=state.trace, messages=msgs, usage=usage, confidence_tier=tier,
                      sources=collect_sources(state.trace), card_from_lookup=bool(state.queued_by_lookup))

    for i in range(settings.MAX_ITERATIONS):
        if i == 0 and parked:
            # Resumed: the model already made this call; execute what it asked for.
            tool_uses, resp, iter_text = parked, None, ""
        else:
            resp, tool_uses, iter_text = await _decide(i, msgs, state, ctx, force_first, on_status, usage)

        if not tool_uses:
            msgs.append({"role": "assistant", "content": model_io.serialize_content(resp.content)})
            if not state.trace:
                backfilled = backfill_walkthrough(messages, "\n\n".join(state.text_parts))
                if backfilled:
                    state.card_steps = list(backfilled["steps"])
                    # A re-shown card waits for Run like any lookup-queued card.
                    state.queued_by_lookup = {"backfill": (0, len(state.card_steps))}
                    state.trace.append({
                        "tool": "open_setting",
                        "input": {"name": backfilled["destination"]},
                        "output": {
                            "attached": True,
                            "destination": backfilled["destination"],
                            "steps": backfilled["steps"],
                            "backfilled": True,
                        },
                    })
            return await finish()

        if resp is not None:
            msgs.append({"role": "assistant", "content": model_io.serialize_content(resp.content)})
        # Park before executing ANY of this batch, so every tool_use still gets
        # exactly one tool_result when the client resumes.
        if research_confirm and not research_authorized and not research_denied:
            pending = [tu for tu in tool_uses if tu.name == "web_research"]
            if pending:
                log.warning("[research_prompt] parking turn for user approval, query=%r", pending[0].input.get("query"))
                return Result(
                    text="", walkthrough_steps=None, trace=state.trace, messages=msgs, usage=usage,
                    confidence_tier="generic", pending_research_query=str(pending[0].input.get("query", "")),
                )
        if on_status:
            status_text = status_for_tools(tool_uses)
            if status_text:
                await on_status(status_text)
        tool_results = [
            await handle_tool_use(tu, state, ax_fixture, research_denied, usage) for tu in tool_uses
        ]
        msgs.append({"role": "user", "content": tool_results})

        if _can_exit_early(tool_uses, tool_results, iter_text, state):
            return await finish()

        # A clarifying question IS the final answer; it ends the turn.
        if any(tu.name == "ask_clarifying_question" for tu in tool_uses):
            # The question itself, when the decider didn't also write it as text.
            if not iter_text.strip():
                state.text_parts += [str(tu.input.get("question")) for tu in tool_uses
                                     if tu.name == "ask_clarifying_question" and tu.input.get("question")]
            return await finish(is_clarifying_question=True)

    # Out of iterations: one tools-off call so the turn never ends empty.
    final = await model_io.call_model(
        ctx.system + [{"type": "text", "text": notes.FINAL_ANSWER_NUDGE}],
        None,
        ctx.for_call(msgs),
        None,
    )
    usage.append(model_io.usage_dict(final.usage))
    final_text = "".join(b.text for b in final.content if b.type == "text")
    if final_text:
        state.text_parts.append(final_text)
    msgs.append({"role": "assistant", "content": model_io.serialize_content(final.content)})
    return await finish()
