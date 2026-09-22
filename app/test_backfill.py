"""Deterministic unit tests for pipeline.py's _backfill_walkthrough -- the
same-destination-revisit fix. Unlike test_multiattach_guard.py's guard (which
never actually misfired live), backfill has already produced two real false
positives this session (multiturn_evidence_arrives_later_turn on the first,
weaker gate; the "thanks again" risk found via Fable review before it ever
happened live). It gets dedicated tests covering hostile phrasing directly,
instead of hoping the right conversation shows up in a live battery run.

Run inside the app container:
    docker compose exec app python -m app.test_backfill
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import db, pipeline


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=10, output_tokens=10,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )


def _tool_use(name: str, input_: dict, id_: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


async def _run_turn(messages: list, responses: list) -> object:
    """Runs one respond() call against a scripted sequence of mocked model
    responses -- one respond() call per turn, same pattern run_graded_battery.py
    uses for multi-turn scenarios (thread result.messages into the next call)."""
    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    with patch.object(pipeline._client.messages, "create", new=AsyncMock(side_effect=fake_create)):
        return await pipeline.respond(messages)


async def test_genuine_reask_backfills() -> None:
    """Turn 1 attaches for real (a real open_setting call against the approved
    routes); turn 2 has zero tool calls and
    explicit re-ask language -- backfill should fire and reattach turn 1's
    exact steps."""
    turn1 = await _run_turn([{"role": "user", "content": "wheres sample rate"}], [
        SimpleNamespace(
            content=[_tool_use("open_setting", {"name": "sample rate"}, "call_1")],
            usage=_usage(),
        ),
        SimpleNamespace(content=[_text("File > Project Settings > Audio, click Sample Rate.")], usage=_usage()),
    ])
    assert turn1.walkthrough_steps is not None, "turn 1 should have attached for real"

    messages = turn1.messages + [{"role": "user", "content": "remind me where that setting was again"}]
    turn2 = await _run_turn(messages, [
        SimpleNamespace(content=[_text("Same path: File > Project Settings > Audio, click the Sample Rate dropdown.")], usage=_usage()),
    ])
    assert turn2.walkthrough_steps == turn1.walkthrough_steps, \
        f"expected backfill to reattach turn 1's steps, got {turn2.walkthrough_steps}"
    backfilled_calls = [c for c in turn2.trace if c["output"].get("backfilled")]
    assert len(backfilled_calls) == 1, f"expected exactly one backfilled trace entry, got {turn2.trace}"
    print("PASS: genuine re-ask correctly backfills the prior walkthrough.")


async def test_thanks_again_does_not_backfill() -> None:
    """The exact false-positive risk found via Fable review (2026-08-06): a
    closing/thank-you turn that happens to contain "again" but isn't asking to
    see anything again. Zero tool calls (normal for a closing turn) plus a
    response that happens to name the destination -- must NOT backfill."""
    turn1 = await _run_turn([{"role": "user", "content": "wheres sample rate"}], [
        SimpleNamespace(
            content=[_tool_use("open_setting", {"name": "sample rate"}, "call_1")],
            usage=_usage(),
        ),
        SimpleNamespace(content=[_text("File > Project Settings > Audio, click Sample Rate.")], usage=_usage()),
    ])
    assert turn1.walkthrough_steps is not None

    messages = turn1.messages + [{"role": "user", "content": "thanks again, that fixed it!"}]
    turn2 = await _run_turn(messages, [
        SimpleNamespace(content=[_text("Glad the sample rate fix worked!")], usage=_usage()),
    ])
    assert turn2.walkthrough_steps is None, \
        f"a closing 'thanks again' turn should NOT backfill, got {turn2.walkthrough_steps}"
    assert not any(c["output"].get("backfilled") for c in turn2.trace), \
        f"expected no backfilled trace entry, got {turn2.trace}"
    print("PASS: 'thanks again' closing turn correctly does NOT backfill.")


async def test_ambiguous_multiple_destinations_does_not_backfill() -> None:
    """Two different destinations attached across history; a re-ask turn whose
    response names both should be left alone rather than guessed which one the
    user means."""
    turn1 = await _run_turn([{"role": "user", "content": "wheres sample rate"}], [
        SimpleNamespace(
            content=[_tool_use("open_setting", {"name": "sample rate"}, "call_1")],
            usage=_usage(),
        ),
        SimpleNamespace(content=[_text("File > Project Settings > Audio, click Sample Rate.")], usage=_usage()),
    ])
    messages = turn1.messages + [{"role": "user", "content": "wheres buffer size"}]
    turn2 = await _run_turn(messages, [
        SimpleNamespace(
            content=[_tool_use("open_setting", {"name": "buffer size"}, "call_2")],
            usage=_usage(),
        ),
        SimpleNamespace(content=[_text("Logic Pro > Settings > Audio, I/O Buffer Size.")], usage=_usage()),
    ])
    assert turn2.walkthrough_steps is not None, "turn 2 should have attached buffer size for real"

    messages = turn2.messages + [{"role": "user", "content": "remind me where those were again"}]
    turn3 = await _run_turn(messages, [
        SimpleNamespace(content=[_text("Sample rate and buffer size are both in Settings, Audio tab.")], usage=_usage()),
    ])
    assert turn3.walkthrough_steps is None, \
        f"ambiguous match across two prior destinations should NOT backfill, got {turn3.walkthrough_steps}"
    print("PASS: ambiguous multi-destination re-ask correctly does NOT backfill.")


async def test_tool_call_this_turn_does_not_backfill() -> None:
    """Even with re-ask language in the user's message, a turn where the model
    actually calls a tool (fresh work, not pure memory recall) should not also
    backfill -- the zero-tool-calls gate must hold regardless of keyword match.

    The response text below deliberately names turn 1's own destination
    ("sample rate") in addition to turn 2's real one -- found via Fable review,
    2026-08-06: an earlier version of this test used a response that never named
    the old destination at all, so it passed even with the zero-tool-calls gate
    deleted entirely (the disambiguation check found no name match and bailed
    out for an unrelated reason, never actually exercising the gate this test
    claims to cover). With the old destination's name present, deleting the gate
    would make this test fail for real.
    """
    turn1 = await _run_turn([{"role": "user", "content": "wheres sample rate"}], [
        SimpleNamespace(
            content=[_tool_use("open_setting", {"name": "sample rate"}, "call_1")],
            usage=_usage(),
        ),
        SimpleNamespace(content=[_text("File > Project Settings > Audio, click Sample Rate.")], usage=_usage()),
    ])

    messages = turn1.messages + [{"role": "user", "content": "remind me, does this also apply to buffer size?"}]
    turn2 = await _run_turn(messages, [
        SimpleNamespace(
            content=[_tool_use("open_setting", {"name": "buffer size"}, "call_2")],
            usage=_usage(),
        ),
        SimpleNamespace(
            content=[_text("Yes, sample rate applies too -- and here's buffer size: Logic Pro > Settings > Audio.")],
            usage=_usage(),
        ),
    ])
    backfilled_calls = [c for c in turn2.trace if c["output"].get("backfilled")]
    assert not backfilled_calls, f"a turn with a real tool call should never also backfill, got {turn2.trace}"
    print("PASS: a turn with a real tool call never also backfills, regardless of re-ask language.")


async def main() -> None:
    await db.connect()
    try:
        await test_genuine_reask_backfills()
        await test_thanks_again_does_not_backfill()
        await test_ambiguous_multiple_destinations_does_not_backfill()
        await test_tool_call_this_turn_does_not_backfill()
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
