"""Deterministic tests for the typed action tools (open_plugin / set_param,
2026-09-21) and the lookup-cap change that came with them.

Covers: action calls concatenating into one card in call order (same response
and across iterations), the per-turn cap, duplicate refusal, the first call being
tool_choice "any" rather than a forced lookup_concept, the cap counting only
unproductive lookups, the resume path restoring queued actions, the
executors' own argument validation, and (2026-09-21) one card per turn for
walkthroughs and actions alike plus the auto-run rule.

No DB, no Voyage: model responses are scripted, and lookup_concept /
get_walkthrough are stubbed in pipeline._EXECUTORS. Run inside the app container:
    docker compose exec app python -m app.test_action_tools
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import pipeline, tools


def _usage() -> SimpleNamespace:
    return SimpleNamespace(input_tokens=10, output_tokens=10,
                           cache_creation_input_tokens=0, cache_read_input_tokens=0)


def _tool_use(name: str, input_: dict, id_: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _resp(*blocks) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), usage=_usage())


async def _fake_walkthrough(inp: dict, fixture) -> dict:
    return {"attached": True, "destination": inp["solution"], "steps": [{"menu_path": ["File", inp["solution"]]}]}


def _fake_lookup(outputs: list[dict]):
    """lookup_concept stub returning `outputs` in order (last one repeats)."""
    n = 0

    async def fake(inp: dict, fixture) -> dict:
        nonlocal n
        out = outputs[min(n, len(outputs) - 1)]
        n += 1
        return out
    return fake


async def _run(responses: list, lookups: list[dict] | None = None, messages: list[dict] | None = None,
               **kwargs):
    """Drive respond() through scripted decider responses. Decider calls pass
    tools=; the writer call doesn't, so it gets a plain text reply and the
    decider script isn't consumed by it. Returns (result, decider_kwargs)."""
    decider_calls: list[dict] = []

    async def fake_create(**kw):
        if "tools" not in kw:
            return _resp(_text("writer text"))
        decider_calls.append(kw)
        return responses[min(len(decider_calls) - 1, len(responses) - 1)]

    stubs = {"get_walkthrough": _fake_walkthrough,
             "lookup_concept": _fake_lookup(lookups or [{"match": "none"}])}
    with patch.object(pipeline._client.messages, "create", new=AsyncMock(side_effect=fake_create)), \
         patch.dict(pipeline._EXECUTORS, stubs):
        result = await pipeline.respond(messages or [{"role": "user", "content": "test"}], **kwargs)
    return result, decider_calls


def _open(plugin: str, id_: str, **extra) -> SimpleNamespace:
    return _tool_use("open_plugin", {"plugin": plugin, **extra}, id_)


def _set(plugin: str, param: str, value: str, id_: str) -> SimpleNamespace:
    return _tool_use("set_param", {"plugin": plugin, "param": param, "value": value}, id_)


async def test_actions_combine_same_response() -> None:
    result, _ = await _run([
        _resp(_open("Channel EQ", "a1"), _set("Channel EQ", "Low Cut Frequency", "80hz", "a2")),
        _resp(_text("done")),
    ])
    assert result.walkthrough_steps == [
        {"ax_open_plugin": "Channel EQ"},
        {"ax_set_param": {"plugin": "Channel EQ", "param": "Low Cut Frequency", "value": "80"}},
    ], result.walkthrough_steps
    print("PASS: two actions in one response -> one card, call order, value '80hz' parsed to '80'.")


async def test_actions_combine_across_iterations() -> None:
    result, _ = await _run([
        _resp(_open("Compressor", "a1", track="Vocal")),
        _resp(_set("Compressor", "Threshold", "-18 dB", "a2")),
        _resp(_text("done")),
    ])
    assert result.walkthrough_steps == [
        {"ax_open_plugin": "Compressor", "track": "Vocal"},
        {"ax_set_param": {"plugin": "Compressor", "param": "Threshold", "value": "-18"}},
    ], result.walkthrough_steps
    print("PASS: actions in separate iterations concatenate; track carried on the wire step.")


async def test_action_cap() -> None:
    calls = [_open(f"Plugin {n}", f"a{n}") for n in range(1, pipeline.MAX_ATTACHES_PER_TURN + 2)]
    result, _ = await _run([_resp(*calls), _resp(_text("done"))])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_plugin"]
    assert sum(o["attached"] for o in outs) == pipeline.MAX_ATTACHES_PER_TURN, outs
    assert outs[-1] == {"attached": False, "reason": pipeline._ATTACH_CAP_REFUSAL}, outs[-1]
    assert len(result.walkthrough_steps) == pipeline.MAX_ATTACHES_PER_TURN
    print(f"PASS: action {pipeline.MAX_ATTACHES_PER_TURN + 1} refused at the cap.")


async def test_duplicate_action_refused() -> None:
    result, _ = await _run([_resp(_open("Compressor", "a1"), _open("Compressor", "a2")), _resp(_text("done"))])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_plugin"]
    assert outs[0]["attached"] and outs[1] == {"attached": False, "reason": pipeline._DUPLICATE_ATTACH_REFUSAL}, outs
    assert result.walkthrough_steps == [{"ax_open_plugin": "Compressor"}]
    print("PASS: an exact duplicate action is refused; it runs once.")


async def test_action_and_walkthrough_share_the_card() -> None:
    # One card per turn: a KB route and an action are both steps, call order.
    result, _ = await _run([
        _resp(_tool_use("get_walkthrough", {"solution": "buffer size"}, "w1"), _open("Compressor", "a1")),
        _resp(_text("done")),
    ])
    assert result.walkthrough_steps == [{"menu_path": ["File", "buffer size"]}, {"ax_open_plugin": "Compressor"}], \
        result.walkthrough_steps
    print("PASS: a walkthrough and an action share one card, in call order.")


async def test_fallback_refused_separate_request_joins() -> None:
    # Two solutions from the SAME problem -> the second is an alternative, refused.
    same = [{"match": "problem", "problem": "crackling", "match_confidence": "strong",
             "solutions": [{"name": "buffer size"}, {"name": "sample rate"}]}]
    result, _ = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        _resp(_tool_use("get_walkthrough", {"solution": "buffer size"}, "w1"),
              _tool_use("get_walkthrough", {"solution": "sample rate"}, "w2")),
        _resp(_text("done")),
    ], lookups=same)
    outs = [c["output"] for c in result.trace if c["tool"] == "get_walkthrough"]
    assert outs[0]["attached"] and outs[1] == {"attached": False, "reason": pipeline._FALLBACK_REFUSAL}, outs
    # Solutions from DIFFERENT problems -> two requests, both on the card.
    two = [{"match": "problem", "problem": "crackling", "match_confidence": "strong", "solutions": [{"name": "buffer size"}]},
           {"match": "problem", "problem": "freezing", "match_confidence": "strong", "solutions": [{"name": "freeze track"}]}]
    result, _ = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1"), _tool_use("lookup_concept", {"problem": "freeze"}, "l2")),
        _resp(_tool_use("get_walkthrough", {"solution": "buffer size"}, "w1"),
              _tool_use("get_walkthrough", {"solution": "freeze track"}, "w2")),
        _resp(_text("done")),
    ], lookups=two)
    assert result.walkthrough_steps == [{"menu_path": ["File", "buffer size"]}, {"menu_path": ["File", "freeze track"]}], \
        result.walkthrough_steps
    print("PASS: an alternative for the same problem is refused; a separate request joins the card.")


async def test_auto_run() -> None:
    card = [_resp(_open("Compressor", "a1")), _resp(_text("done"))]
    cases = {
        "put a compressor on this": True,
        "my vocal sounds muddy, fix it": True,
        "can you add a compressor to this track": True,
        "how do i add a compressor": False,
        "explain what a compressor does and put one on this": False,
        "channel eq?": False,
        "put a compressor on every track": False,
    }
    for text, want in cases.items():
        result, _ = await _run(card, messages=[{"role": "user", "content": text}])
        assert result.auto_run is want, (text, result.auto_run)
    result, _ = await _run([_resp(_text("hi there"))], messages=[{"role": "user", "content": "put a compressor on this"}])
    assert result.walkthrough_steps is None and result.auto_run is False
    print("PASS: auto_run follows the message (instruction vs question vs bulk) and needs a card.")


def test_is_question() -> None:
    questions = ["how do i open channel eq", "what does ratio do", "why is my mix harsh, add something",
                 "channel eq?", "show me how to open channel eq", "can i add a compressor here",
                 "can you explain what a compressor does", "hey, how do i mute a track", "wheres the buffer size",
                 "do i need a limiter", "is my vocal too loud"]
    instructions = ["open channel eq", "put valhalla on the vocal", "my vocal sounds muddy, fix it",
                    "can you add a compressor", "could you set the low cut to 80?", "please add an eq",
                    "yes please, do it for me", "do it", "set the threshold to -18"]
    for t in questions:
        assert pipeline._is_question(pipeline._last_user_text([{"role": "user", "content": t}])), t
    for t in instructions:
        assert not pipeline._is_question(pipeline._last_user_text([{"role": "user", "content": t}])), t
    print(f"PASS: _is_question on {len(questions)} questions and {len(instructions)} instructions.")


async def test_first_call_is_any() -> None:
    _, calls = await _run([_resp(_open("Compressor", "a1")), _resp(_text("done"))])
    assert calls[0]["tool_choice"] == {"type": "any"}, calls[0].get("tool_choice")
    assert "tool_choice" not in calls[1], calls[1].get("tool_choice")
    with patch.object(pipeline, "FORCE_FIRST_LOOKUP", True):
        _, calls = await _run([_resp(_tool_use("lookup_concept", {"problem": "x"}, "l1")), _resp(_text("done"))])
    assert calls[0]["tool_choice"] == {"type": "tool", "name": "lookup_concept"}, calls[0].get("tool_choice")
    print("PASS: first call is tool_choice any; FORCE_FIRST_LOOKUP restores the forced lookup.")


def _lookups(n: int, id_prefix: str = "l") -> list:
    return [_resp(_tool_use("lookup_concept", {"problem": f"q{i}"}, f"{id_prefix}{i}")) for i in range(n)]


def _lookup_outputs(result) -> list[dict]:
    return [c["output"] for c in result.trace if c["tool"] == "lookup_concept"]


async def test_cap_counts_only_unproductive() -> None:
    limit = pipeline.LOOKUP_ATTEMPT_LIMIT
    # limit + 1 unproductive (no match) lookups -> the last one is capped
    result, _ = await _run(_lookups(limit + 1) + [_resp(_text("done"))], lookups=[{"match": "none"}])
    outs = _lookup_outputs(result)
    assert "error" in outs[-1] and all("error" not in o for o in outs[:-1]), outs

    # limit + 1 lookups each returning a DIFFERENT real bucket -> none capped
    distinct = [{"match": "problem", "problem": f"bucket {i}", "match_confidence": "strong"} for i in range(limit + 1)]
    with patch.object(pipeline, "MAX_ITERATIONS", limit + 3):
        result, _ = await _run(_lookups(limit + 1) + [_resp(_text("done"))], lookups=distinct)
    assert all("error" not in o for o in _lookup_outputs(result)), _lookup_outputs(result)

    # the same bucket over and over is unproductive after the first
    same = [{"match": "problem", "problem": "bucket", "match_confidence": "strong"}]
    with patch.object(pipeline, "MAX_ITERATIONS", limit + 3):
        result, _ = await _run(_lookups(limit + 2) + [_resp(_text("done"))], lookups=same)
    outs = _lookup_outputs(result)
    assert "error" in outs[-1] and all("error" not in o for o in outs[:limit + 1]), outs

    # COUNT_ALL_LOOKUPS restores the old rule: distinct buckets still count
    with patch.object(pipeline, "COUNT_ALL_LOOKUPS", True), patch.object(pipeline, "MAX_ITERATIONS", limit + 3):
        result, _ = await _run(_lookups(limit + 1) + [_resp(_text("done"))], lookups=distinct)
    assert "error" in _lookup_outputs(result)[-1]
    print("PASS: cap counts no-match / weak / repeated-bucket lookups only; COUNT_ALL_LOOKUPS restores the old rule.")


async def test_resume_restores_actions() -> None:
    # A turn parked at a web_research prompt after an action was queued: the
    # resumed turn must still ship the action.
    step = {"ax_open_plugin": "Compressor"}
    parked = [
        {"role": "user", "content": "put a compressor on this and tell me about 1176s"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a1", "name": "open_plugin", "input": {"plugin": "Compressor"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a1",
             "content": json.dumps({"attached": True, "action": "open_plugin", "steps": [step]})}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "r1", "name": "web_research", "input": {"query": "1176"}}]},
    ]
    with patch.object(pipeline.research, "web_research", new=AsyncMock(return_value={"findings": "x", "sources": []})):
        result, _ = await _run([_resp(_text("done"))], messages=parked, resume="allow_research")
    assert result.walkthrough_steps == [step], result.walkthrough_steps
    print("PASS: a resumed turn keeps the action queued before it parked.")


def test_executor_validation() -> None:
    assert tools.queue_open_plugin({"plugin": "  "})["attached"] is False
    assert tools.queue_open_plugin({"plugin": "Channel EQ", "new_instance": True})["steps"] == [
        {"ax_open_plugin": "Channel EQ", "new": True}]
    assert tools.queue_open_plugin({"plugin": "Channel EQ", "new_instance": False})["steps"] == [
        {"ax_open_plugin": "Channel EQ"}]
    cases = {"80": "80", "80hz": "80", "-18 dB": "-18", "2.5": "2.5", "On": "on", "disabled": "off",
             -18: "-18", 4.0: "4", True: "on"}
    for raw, want in cases.items():
        got = tools.queue_set_param({"plugin": "P", "param": "X", "value": raw})
        assert got["attached"] and got["steps"][0]["ax_set_param"]["value"] == want, (raw, got)
    for bad in ("4:1", "loud", "", None):
        got = tools.queue_set_param({"plugin": "P", "param": "X", "value": bad})
        assert got["attached"] is False and "plain number" in got["reason"], (bad, got)
    print("PASS: executors validate names and parse values ('80hz'->'80', 'On'->'on'; '4:1' refused).")


async def main() -> None:
    test_executor_validation()
    test_is_question()
    await test_actions_combine_same_response()
    await test_actions_combine_across_iterations()
    await test_action_cap()
    await test_duplicate_action_refused()
    await test_action_and_walkthrough_share_the_card()
    await test_fallback_refused_separate_request_joins()
    await test_auto_run()
    await test_first_call_is_any()
    await test_cap_counts_only_unproductive()
    await test_resume_restores_actions()


if __name__ == "__main__":
    asyncio.run(main())
