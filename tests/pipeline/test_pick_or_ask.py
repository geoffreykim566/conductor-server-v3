"""The pick-or-ask rule: after citing a bucket whose candidates map to actions, the turn can't end on prose alone."""
from __future__ import annotations

from app.pipeline import notes, settings
from conftest import cite_kb, resp, route, text_block, tool_use

CRACKLING = "crackling during playback"


async def test_bucket_forces_one_reask(run) -> None:
    """A prose-only answer after a cited bucket gets one re-ask with the pick nudge, which picks."""
    result, calls = await run([
        resp(cite_kb([CRACKLING], "l1")),
        resp(text_block("It's CPU overload -- raise the buffer.")),
        resp(tool_use("open_setting", {"name": "buffer size"}, "w1")),
        resp(text_block("Raise the buffer size.")),
    ])
    assert calls[2].get("tool_choice") == settings.FORCED_TOOL_CHOICE, calls[2].get("tool_choice")
    assert notes.PICK_NUDGE in calls[2]["system"][-1]["text"], calls[2]["system"][-1]
    assert result.walkthrough_steps == route("buffer size"), result.walkthrough_steps


async def test_bucket_reasks_only_once(run) -> None:
    """A second prose-only answer ends the turn with no card."""
    result, calls = await run([
        resp(cite_kb([CRACKLING], "l1")),
        resp(text_block("prose")), resp(text_block("still prose")),
    ])
    assert result.walkthrough_steps is None and len(calls) == 3, (result.walkthrough_steps, len(calls))


async def test_clarifying_question_satisfies_pick(run) -> None:
    """A clarifying question counts as the pick-or-ask answer."""
    _, calls = await run([
        resp(cite_kb([CRACKLING], "l1")),
        resp(tool_use("ask_clarifying_question", {"question": "Does it crackle only with many plugins?"}, "q1")),
    ])
    assert len(calls) == 2, len(calls)


async def test_empty_citation_does_not_force_a_pick(run) -> None:
    """Citing nothing (no verified answer) never forces a pick -- the honest answer stands."""
    result, calls = await run([
        resp(cite_kb([], "l1")),
        resp(text_block("I don't have a verified answer for that.")),
    ])
    assert result.walkthrough_steps is None and len(calls) == 2, (result.walkthrough_steps, len(calls))
