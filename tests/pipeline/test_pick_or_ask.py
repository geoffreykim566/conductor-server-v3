"""The pick-or-ask rule: after a lookup whose candidates map to actions, the turn can't end on prose alone."""
from __future__ import annotations

from unittest.mock import patch

from app.pipeline import notes, settings
from conftest import bucket, resp, route, single, text_block, tool_use

CRACKLING = [bucket("crackling", {"CPU overload": {"open_setting": "buffer size"},
                                  "sample rate mismatch": {"open_setting": "sample rate"}})]


async def test_moderate_single_is_pick_or_ask(run) -> None:
    """A moderate single isn't auto-attached but gets pick-or-ask (flag restores prose-only)."""
    lookups = [single("recording settings", {"open_setting": "recording settings"}, conf="moderate — treat with skepticism")]
    result, calls = await run([resp(tool_use("lookup_concept", {"problem": "x"}, "l1")), resp(text_block("hm")),
                               resp(tool_use("open_setting", {"name": "recording settings"}, "w1")), resp(text_block("ok"))],
                              lookups=lookups)
    assert calls[2]["tool_choice"] == {"type": "any"}, calls[2].get("tool_choice")
    assert result.walkthrough_steps == route("recording settings") and result.card_from_lookup is False
    with patch.object(settings, "PICK_ON_MODERATE_SINGLE", False):
        result, calls = await run([resp(tool_use("lookup_concept", {"problem": "x"}, "l1")), resp(text_block("hm"))],
                                  lookups=lookups)
    assert result.walkthrough_steps is None and len(calls) == 2


async def test_bucket_forces_one_reask(run) -> None:
    """A prose-only answer after a bucket lookup gets one forced re-ask, which picks."""
    result, calls = await run([
        resp(tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        resp(text_block("It's CPU overload -- raise the buffer.")),
        resp(tool_use("open_setting", {"name": "buffer size"}, "w1")),
        resp(text_block("Raise the buffer size.")),
    ], lookups=CRACKLING)
    assert calls[2]["tool_choice"] == {"type": "any"} and notes.PICK_NUDGE in calls[2]["system"][-1]["text"], calls[2]
    assert result.walkthrough_steps == route("buffer size"), result.walkthrough_steps


async def test_bucket_reasks_only_once(run) -> None:
    """A second prose-only answer ends the turn with no card."""
    result, calls = await run([
        resp(tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        resp(text_block("prose")), resp(text_block("still prose")),
    ], lookups=CRACKLING)
    assert result.walkthrough_steps is None and len(calls) == 3, (result.walkthrough_steps, len(calls))


async def test_clarifying_question_satisfies_pick(run) -> None:
    """A clarifying question counts as the pick-or-ask answer."""
    _, calls = await run([
        resp(tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        resp(tool_use("ask_clarifying_question", {}, "q1")),
    ], lookups=CRACKLING)
    assert len(calls) == 2, len(calls)


async def test_weak_bucket_does_not_force_a_pick(run) -> None:
    """A weak match never forces a pick -- an honest 'no verified answer' stands."""
    weak = [{"match": "problem", "problem": "something else", "match_confidence": "weak — likely not relevant",
             "solutions": [{"name": "buffer size", "action": {"open_setting": "buffer size"}}]}]
    result, calls = await run([
        resp(tool_use("lookup_concept", {"problem": "phantom power"}, "l1")),
        resp(text_block("I don't have a verified answer for that.")),
    ], lookups=weak)
    assert result.walkthrough_steps is None and len(calls) == 2, (result.walkthrough_steps, len(calls))
