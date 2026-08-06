"""Deterministic regression test for pipeline.py's commit-to-one multi-attach guard.

The guard (get_walkthrough refused after a successful attach already happened in the
same turn) has never actually fired in any real battery run -- the model has always
voluntarily called it only once, so the code path itself was unverified (found via
adversarial review, 2026-08-05: every scenario "fixed" by this guard passed because
of the prompt rule, not the guard). This forces the exact trigger condition -- two
get_walkthrough tool_use blocks in a single model response -- by mocking the
Anthropic client directly, instead of hoping a scenario provokes the model into it.

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


async def main() -> None:
    # First model call: two get_walkthrough tool_use blocks in one response -- the
    # exact shape the prompt instructs the model never to produce, forced directly
    # here so the guard is tested independently of whether the model would try it.
    first_response = SimpleNamespace(
        content=[
            _tool_use("get_walkthrough", {"solution": "sample rate"}, "call_1"),
            _tool_use("get_walkthrough", {"solution": "buffer size"}, "call_2"),
        ],
        usage=_usage(),
    )
    # Second model call, after tool results come back: plain text, ends the turn.
    second_response = SimpleNamespace(content=[_text("done")], usage=_usage())

    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        return first_response if call_count == 1 else second_response

    await db.connect()
    try:
        with patch.object(pipeline._client.messages, "create", new=AsyncMock(side_effect=fake_create)):
            result = await pipeline.respond([{"role": "user", "content": "test"}])
    finally:
        await db.disconnect()

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

    print("PASS: multi-attach guard correctly refused the second get_walkthrough call in the same turn.")
    print(f"  first call:  {walkthrough_calls[0]['output']}")
    print(f"  second call: {walkthrough_calls[1]['output']}")


if __name__ == "__main__":
    asyncio.run(main())
