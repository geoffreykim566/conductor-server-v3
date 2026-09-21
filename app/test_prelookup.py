"""Server-side first lookup (pipeline.PRELOOKUP_FIRST_CALL, 2026-09-19).
No DB, no Voyage, no API: the lookup executor and the model client are both
mocked. Runnable as a script (the container has no pytest-asyncio):
    docker compose exec app python -m app.test_prelookup
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import pipeline

_FAKE_LOOKUP = {
    "match": "solution", "match_confidence": "strong", "problem": None,
    "solutions": [{"name": "sample rate", "has_path": True}],
}


def _usage():
    return SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def _text(t):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=t)], usage=_usage())


def _tool(name, inp, id_):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=name, input=inp, id=id_)], usage=_usage())


async def _run(messages, responses, *, prelookup=True, cap=4):
    calls: list[dict] = []
    executed: list[str] = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    async def fake_lookup(inp, fixture):
        executed.append(inp["problem"])
        return dict(_FAKE_LOOKUP)

    old = pipeline._EXECUTORS["lookup_concept"], pipeline.PRELOOKUP_FIRST_CALL, pipeline.LOOKUP_ATTEMPT_LIMIT
    pipeline._EXECUTORS["lookup_concept"] = fake_lookup
    pipeline.PRELOOKUP_FIRST_CALL, pipeline.LOOKUP_ATTEMPT_LIMIT = prelookup, cap
    try:
        with patch.object(pipeline._client.messages, "create", new=AsyncMock(side_effect=fake_create)):
            result = await pipeline.respond(messages)
    finally:
        pipeline._EXECUTORS["lookup_concept"], pipeline.PRELOOKUP_FIRST_CALL, pipeline.LOOKUP_ATTEMPT_LIMIT = old
    return result, calls, executed


async def test_prelookup_replaces_forced_first_call():
    q = "wheres sample rate"
    result, calls, executed = await _run([{"role": "user", "content": q}], [_text("Project Settings > Audio."), _text("writer")])
    assert executed == [q], executed
    first = calls[0]
    assert "tool_choice" not in first, "first decider call must not be forced anymore"
    m = first["messages"]
    assert m[-2]["role"] == "assistant" and m[-2]["content"][0]["type"] == "tool_use"
    assert m[-2]["content"][0]["name"] == "lookup_concept" and m[-2]["content"][0]["input"] == {"problem": q}
    assert m[-1]["role"] == "user" and m[-1]["content"][0]["type"] == "tool_result"
    assert m[-1]["content"][0]["tool_use_id"] == m[-2]["content"][0]["id"]
    assert json.loads(m[-1]["content"][0]["content"]) == _FAKE_LOOKUP
    assert result.trace[0] == {"tool": "lookup_concept", "input": {"problem": q}, "output": _FAKE_LOOKUP}
    # the pair survives into the client-bound history with the same shape
    assert result.messages[1]["content"][0]["type"] == "tool_use"
    assert result.messages[2]["content"][0]["type"] == "tool_result"
    assert result.confidence_tier != "generic", result.confidence_tier
    print("PASS: pre-lookup replaces the forced first call and lands in transcript + trace")


async def test_prelookup_counts_toward_cap():
    result, calls, executed = await _run(
        [{"role": "user", "content": "wheres sample rate"}],
        [_tool("lookup_concept", {"problem": "sample rate setting"}, "c1"), _text("done"), _text("writer")],
        cap=1,
    )
    assert executed == ["wheres sample rate"], "the refused second lookup must not execute"
    lookups = [c for c in result.trace if c["tool"] == "lookup_concept"]
    assert len(lookups) == 2
    assert lookups[1]["output"].get("error", "").startswith("Too many lookup attempts (2)"), lookups[1]
    print("PASS: pre-lookup counts as attempt 1 against LOOKUP_ATTEMPT_LIMIT")


async def test_carveouts_skip_prelookup():
    for text in ("thanks", "delete my entire project"):
        _, calls, executed = await _run([{"role": "user", "content": text}], [_text("ok"), _text("writer")])
        assert executed == [], (text, executed)
        assert "tool_choice" not in calls[0]
    print("PASS: closing / irreversible carve-outs skip the pre-lookup")


async def test_flag_off_restores_forced_call():
    result, calls, executed = await _run(
        [{"role": "user", "content": "wheres sample rate"}],
        [_tool("lookup_concept", {"problem": "sample rate"}, "c1"), _text("done"), _text("writer")],
        prelookup=False,
    )
    assert calls[0].get("tool_choice") == {"type": "tool", "name": "lookup_concept"}
    assert executed == ["sample rate"], executed
    assert result.trace[0]["input"] == {"problem": "sample rate"}
    print("PASS: PRELOOKUP_FIRST_CALL=False restores the forced first decider call")


async def main():
    for t in (test_prelookup_replaces_forced_first_call, test_prelookup_counts_toward_cap,
              test_carveouts_skip_prelookup, test_flag_off_restores_forced_call):
        await t()


if __name__ == "__main__":
    asyncio.run(main())
