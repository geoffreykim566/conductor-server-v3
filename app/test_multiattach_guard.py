"""Deterministic regression tests for pipeline.py's commit-to-one multi-attach guard.

The guard (get_walkthrough refused after a successful attach already happened in the
same turn) has never actually fired in any real battery run -- the model has always
voluntarily called it only once, so the code path itself was unverified (found via
adversarial review, 2026-08-05: every scenario "fixed" by this guard passed because
of the prompt rule, not the guard). These force the exact trigger condition directly
by mocking the Anthropic client, instead of hoping a scenario provokes the model into
it, covering the two distinct shapes that hit the same guard condition
(pipeline.py: walkthrough_steps is not None):

  1. two get_walkthrough tool_use blocks in a single model response
  2. an attach in one loop iteration, a retry attempt in a later iteration

Run inside the app container:
    docker compose exec app python -m app.test_multiattach_guard
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


async def _run_with_responses(responses: list) -> object:
    """Drives pipeline.respond() through a scripted sequence of mocked model
    responses, one per _client.messages.create() call, in order."""
    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    await db.connect()
    try:
        with patch.object(pipeline._client.messages, "create", new=AsyncMock(side_effect=fake_create)):
            return await pipeline.respond([{"role": "user", "content": "test"}])
    finally:
        await db.disconnect()


def _assert_second_call_refused(result) -> None:
    walkthrough_calls = [c for c in result.trace if c["tool"] == "get_walkthrough"]
    assert len(walkthrough_calls) == 2, f"expected 2 get_walkthrough calls in trace, got {len(walkthrough_calls)}"
    assert walkthrough_calls[0]["output"]["attached"] is True, \
        f"first call should have attached: {walkthrough_calls[0]['output']}"
    assert walkthrough_calls[1]["output"]["attached"] is False, \
        f"second call should have been refused: {walkthrough_calls[1]['output']}"
    assert "already attached" in walkthrough_calls[1]["output"]["reason"], \
        f"refusal reason missing/wrong: {walkthrough_calls[1]['output'].get('reason')!r}"
    assert result.walkthrough_steps == walkthrough_calls[0]["output"]["steps"], \
        "walkthrough_steps should reflect the FIRST successful attach, not the refused second one"
    print(f"  first call:  {walkthrough_calls[0]['output']}")
    print(f"  second call: {walkthrough_calls[1]['output']}")


async def test_same_response_duplicate() -> None:
    # Both get_walkthrough tool_use blocks in one response -- the exact shape the
    # prompt instructs the model never to produce, forced directly here so the
    # guard is tested independently of whether the model would try it.
    responses = [
        SimpleNamespace(
            content=[
                _tool_use("get_walkthrough", {"solution": "sample rate"}, "call_1"),
                _tool_use("get_walkthrough", {"solution": "buffer size"}, "call_2"),
            ],
            usage=_usage(),
        ),
        SimpleNamespace(content=[_text("done")], usage=_usage()),
    ]
    result = await _run_with_responses(responses)
    _assert_second_call_refused(result)
    print("PASS: guard refused a second get_walkthrough call in the same response.")


async def test_later_iteration_duplicate() -> None:
    # Attach succeeds in loop iteration 1; a *different* model response, in a
    # later loop iteration, attempts a second get_walkthrough for a different
    # solution -- the realistic retry shape (e.g. after mentioning a fallback
    # candidate in prose) that hits the identical guard condition but was never
    # exercised by test_same_response_duplicate.
    responses = [
        SimpleNamespace(
            content=[_tool_use("get_walkthrough", {"solution": "sample rate"}, "call_1")],
            usage=_usage(),
        ),
        SimpleNamespace(
            content=[_tool_use("get_walkthrough", {"solution": "buffer size"}, "call_2")],
            usage=_usage(),
        ),
        SimpleNamespace(content=[_text("done")], usage=_usage()),
    ]
    result = await _run_with_responses(responses)
    _assert_second_call_refused(result)
    print("PASS: guard refused a get_walkthrough retry in a later loop iteration.")


async def test_actions_combine() -> None:
    # The compound-command shape (2026-09-19): two DIRECT ACTIONS in one turn
    # ("add an eq and set the low cut") are not fallback candidates for each
    # other -- they are both wanted, they run in attach order, and each gets its
    # own ledger entry. Commit-to-one deliberately does not apply to these; the
    # steps concatenate into one card. Guarded by _is_action_steps, not by
    # problem bucket: both template rows belong to no problem at all.
    responses = [
        SimpleNamespace(
            content=[
                _tool_use("get_walkthrough",
                          {"solution": "open plugin",
                           "args": {"plugin": "Channel EQ", "track": "Audio 1"}}, "call_1"),
                _tool_use("get_walkthrough",
                          {"solution": "set plugin parameter",
                           "args": {"plugin": "Channel EQ", "param": "Low Cut Frequency",
                                    "value": "80", "track": "Audio 1"}}, "call_2"),
            ],
            usage=_usage(),
        ),
        SimpleNamespace(content=[_text("done")], usage=_usage()),
    ]
    result = await _run_with_responses(responses)
    calls = [c for c in result.trace if c["tool"] == "get_walkthrough"]
    assert len(calls) == 2, f"expected 2 get_walkthrough calls, got {len(calls)}"
    for i, c in enumerate(calls):
        assert c["output"].get("attached") is True, f"call {i + 1} should have attached: {c['output']}"
    expected = list(calls[0]["output"]["steps"]) + list(calls[1]["output"]["steps"])
    assert result.walkthrough_steps == expected, (
        f"steps should be first + second in attach order, got {result.walkthrough_steps}")
    print(f"  combined steps: {result.walkthrough_steps}")
    print("PASS: two direct actions combined into one run, in attach order.")


async def test_action_cap() -> None:
    # Past the cap, a further action is refused rather than queued.
    action = lambda n: _tool_use(  # noqa: E731
        "get_walkthrough",
        {"solution": "open plugin", "args": {"plugin": "Compressor", "track": f"Audio {n}"}},
        f"call_{n}",
    )
    responses = [
        SimpleNamespace(content=[action(1), action(2), action(3), action(4)], usage=_usage()),
        SimpleNamespace(content=[_text("done")], usage=_usage()),
    ]
    result = await _run_with_responses(responses)
    calls = [c for c in result.trace if c["tool"] == "get_walkthrough"]
    attached = [c for c in calls if c["output"].get("attached")]
    assert len(attached) == pipeline.MAX_ATTACHES_PER_TURN, (
        f"expected {pipeline.MAX_ATTACHES_PER_TURN} attaches, got {len(attached)}")
    assert calls[-1]["output"]["attached"] is False, f"4th action should be refused: {calls[-1]['output']}"
    assert "limit for one turn" in calls[-1]["output"]["reason"], calls[-1]["output"]
    print(f"PASS: attaches capped at {pipeline.MAX_ATTACHES_PER_TURN} per turn.")


async def main() -> None:
    await test_same_response_duplicate()
    await test_later_iteration_duplicate()
    await test_actions_combine()
    await test_action_cap()


if __name__ == "__main__":
    asyncio.run(main())
