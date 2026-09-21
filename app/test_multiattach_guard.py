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

Since 2026-09-21 (one card per turn) the guard refuses only an ALTERNATIVE for
the same problem; a walkthrough for a different problem joins the card
(test_action_tools covers that). Neither solution here came from a
lookup_concept result, so neither has a known problem and both count as the
same one -- the conservative reading, and the one these tests pin.

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


async def main() -> None:
    await test_same_response_duplicate()
    await test_later_iteration_duplicate()


if __name__ == "__main__":
    asyncio.run(main())
