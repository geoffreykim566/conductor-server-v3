"""Early exit (settings.EARLY_EXIT_ON_ACTION): an iteration of only accepted actions and/or citations ends the turn."""
from __future__ import annotations

from app.pipeline import notes, settings
from conftest import cite_kb, open_plugin, resp, route, text_block, tool_use


async def test_direct_action_exits(run) -> None:
    """A direct action that attached ends the turn without another decider call."""
    result, calls = await run([resp(open_plugin("Compressor", "a1")), resp(text_block("done"))], early_exit=True)
    assert len(calls) == 1, len(calls)
    assert result.walkthrough_steps == [{"ax_open_plugin": "Compressor"}], result.walkthrough_steps


async def test_refusal_keeps_looping(run) -> None:
    """A refused action keeps looping so the decider can react."""
    _, calls = await run([resp(tool_use("open_setting", {"name": "no such route"}, "a1")),
                          resp(text_block("done"))], early_exit=True)
    assert len(calls) == 2, len(calls)


async def test_citation_with_explanation_exits(run) -> None:
    """Explanation + citation (+ its auto-queued action) in one response: done."""
    result, calls = await run([resp(text_block("Raise the I/O buffer size."), cite_kb(["buffer size"], "l1")),
                               resp(text_block("more"))], early_exit=True)
    assert len(calls) == 1 and result.walkthrough_steps == route("buffer size"), (len(calls), result.walkthrough_steps)


async def test_citation_without_text_keeps_looping(run) -> None:
    """A citation with no explanation keeps looping so it gets written."""
    _, calls = await run([resp(cite_kb(["buffer size"], "l1")),
                          resp(text_block("Raise the I/O buffer size."))], early_exit=True)
    assert len(calls) == 2, len(calls)


async def test_empty_citation_with_text_exits(run) -> None:
    """Nothing in the KB, said so: an empty citation plus text ends the turn."""
    _, calls = await run([resp(text_block("I don't have a verified answer."), cite_kb([], "l1")),
                          resp(text_block("more"))], early_exit=True)
    assert len(calls) == 1, len(calls)


async def test_unknown_entry_keeps_looping(run) -> None:
    """A citation naming a non-entry comes back with an error and keeps looping."""
    _, calls = await run([resp(text_block("Here's why."), cite_kb(["not a real entry"], "l1")),
                          resp(text_block("more"))], early_exit=True)
    assert len(calls) == 2, len(calls)


async def test_unpicked_bucket_keeps_looping(run) -> None:
    """A cited bucket with actionable candidates and nothing picked is not done: the pick re-ask follows."""
    _, calls = await run([resp(text_block("Could be a few things."), cite_kb(["crackling during playback"], "l1")),
                          resp(text_block("prose")),
                          resp(tool_use("ask_clarifying_question", {"question": "When?"}, "q1"))], early_exit=True)
    assert len(calls) == 3 and calls[2].get("tool_choice") == settings.FORCED_TOOL_CHOICE, \
        [c.get("tool_choice") for c in calls]
    assert notes.PICK_NUDGE in calls[2]["system"][-1]["text"], calls[2]["system"][-1]
