"""backfill_walkthrough: re-attaching a prior turn's card when the user re-asks for the same destination.
Hostile phrasing is tested directly because backfill has produced real false positives."""
from __future__ import annotations

from conftest import resp, text_block, tool_use


async def _turn(run, messages: list, responses: list):
    """One respond() call; thread result.messages into the next turn."""
    result, _ = await run(responses, messages=messages)
    return result


async def _sample_rate_turn(run):
    return await _turn(run, [{"role": "user", "content": "wheres sample rate"}], [
        resp(tool_use("open_setting", {"name": "sample rate"}, "call_1")),
        resp(text_block("File > Project Settings > Audio, click Sample Rate.")),
    ])


async def test_genuine_reask_backfills(run) -> None:
    """A zero-tool-call turn with explicit re-ask language reattaches turn 1's exact steps."""
    turn1 = await _sample_rate_turn(run)
    assert turn1.walkthrough_steps is not None, "turn 1 should have attached for real"

    messages = turn1.messages + [{"role": "user", "content": "remind me where that setting was again"}]
    turn2 = await _turn(run, messages, [
        resp(text_block("Same path: File > Project Settings > Audio, click the Sample Rate dropdown.")),
    ])
    assert turn2.walkthrough_steps == turn1.walkthrough_steps, \
        f"expected backfill to reattach turn 1's steps, got {turn2.walkthrough_steps}"
    backfilled_calls = [c for c in turn2.trace if c["output"].get("backfilled")]
    assert len(backfilled_calls) == 1, f"expected exactly one backfilled trace entry, got {turn2.trace}"


async def test_thanks_again_does_not_backfill(run) -> None:
    """A closing 'thanks again' turn that names the destination is not a re-ask."""
    turn1 = await _sample_rate_turn(run)
    assert turn1.walkthrough_steps is not None

    messages = turn1.messages + [{"role": "user", "content": "thanks again, that fixed it!"}]
    turn2 = await _turn(run, messages, [resp(text_block("Glad the sample rate fix worked!"))])
    assert turn2.walkthrough_steps is None, \
        f"a closing 'thanks again' turn should NOT backfill, got {turn2.walkthrough_steps}"
    assert not any(c["output"].get("backfilled") for c in turn2.trace), \
        f"expected no backfilled trace entry, got {turn2.trace}"


async def test_ambiguous_multiple_destinations_does_not_backfill(run) -> None:
    """A re-ask whose response names two prior destinations is left alone, not guessed."""
    turn1 = await _sample_rate_turn(run)
    messages = turn1.messages + [{"role": "user", "content": "wheres buffer size"}]
    turn2 = await _turn(run, messages, [
        resp(tool_use("open_setting", {"name": "buffer size"}, "call_2")),
        resp(text_block("Logic Pro > Settings > Audio, I/O Buffer Size.")),
    ])
    assert turn2.walkthrough_steps is not None, "turn 2 should have attached buffer size for real"

    messages = turn2.messages + [{"role": "user", "content": "remind me where those were again"}]
    turn3 = await _turn(run, messages, [
        resp(text_block("Sample rate and buffer size are both in Settings, Audio tab.")),
    ])
    assert turn3.walkthrough_steps is None, \
        f"ambiguous match across two prior destinations should NOT backfill, got {turn3.walkthrough_steps}"


async def test_tool_call_this_turn_does_not_backfill(run) -> None:
    """A turn with a real tool call never also backfills, regardless of re-ask language.
    The response names turn 1's destination on purpose so the zero-tool-calls gate,
    not the name-match check, is what this exercises."""
    turn1 = await _sample_rate_turn(run)

    messages = turn1.messages + [{"role": "user", "content": "remind me, does this also apply to buffer size?"}]
    turn2 = await _turn(run, messages, [
        resp(tool_use("open_setting", {"name": "buffer size"}, "call_2")),
        resp(text_block("Yes, sample rate applies too -- and here's buffer size: Logic Pro > Settings > Audio.")),
    ])
    backfilled_calls = [c for c in turn2.trace if c["output"].get("backfilled")]
    assert not backfilled_calls, f"a turn with a real tool call should never also backfill, got {turn2.trace}"
