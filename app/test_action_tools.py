"""Deterministic tests for the typed action tools (open_plugin / set_param,
2026-09-21) and the lookup-cap change that came with them.

Covers: action calls concatenating into one card in call order (same response
and across iterations), the per-turn cap, duplicate refusal, the first call being
tool_choice "any" rather than a forced lookup_concept, the cap counting only
unproductive lookups, the resume path restoring queued actions, the
executors' own argument validation, one card per turn, the auto-run rule, and
(routes, 2026-09-21) open_setting, a strong single lookup queuing its
solution's action, pick-or-ask after a bucket lookup, and refusing a second
candidate from the same bucket.

No DB, no Voyage: model responses are scripted and lookup_concept is stubbed
in pipeline._EXECUTORS (open_setting reads seed/routes.json). Run inside the app container:
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


def _route(name: str) -> list:
    return tools.queue_open_setting({"name": name})["steps"]


def _single(solution: str, action: dict, conf: str = "strong") -> dict:
    return {"match": "single", "problem": solution, "match_confidence": conf,
            "solutions": [{"name": solution, "action": action}]}


def _bucket(problem: str, cands: dict[str, dict]) -> dict:
    return {"match": "problem", "problem": problem, "match_confidence": "strong",
            "solutions": [{"name": n, "action": a} for n, a in cands.items()]}


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
               early_exit: bool = False, **kwargs):  # noqa: D401
    """Drive respond() through scripted decider responses. Decider calls pass
    tools=; the writer call doesn't, so it gets a plain text reply and the
    decider script isn't consumed by it. Returns (result, decider_kwargs).
    EARLY_EXIT_ON_ACTION is off unless asked for: these scripts keep the
    decider going after an action, which is what most tests here exercise."""
    decider_calls: list[dict] = []

    async def fake_create(**kw):
        if "tools" not in kw:
            return _resp(_text("writer text"))
        decider_calls.append(kw)
        return responses[min(len(decider_calls) - 1, len(responses) - 1)]

    stubs = {"lookup_concept": _fake_lookup(lookups or [{"match": "none"}])}
    with patch.object(pipeline._client.messages, "create", new=AsyncMock(side_effect=fake_create)), \
         patch.dict(pipeline._EXECUTORS, stubs), \
         patch.object(pipeline, "EARLY_EXIT_ON_ACTION", early_exit):
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


async def test_early_exit_direct_action() -> None:
    result, calls = await _run([_resp(_open("Compressor", "a1")), _resp(_text("done"))], early_exit=True)
    assert len(calls) == 1, len(calls)
    assert result.walkthrough_steps == [{"ax_open_plugin": "Compressor"}], result.walkthrough_steps
    print("PASS: early exit -- a direct action that attached ends the turn without another decider call.")


async def test_early_exit_not_on_refusal() -> None:
    _, calls = await _run([_resp(_tool_use("open_setting", {"name": "no such route"}, "a1")),
                           _resp(_text("done"))], early_exit=True)
    assert len(calls) == 2, len(calls)
    print("PASS: early exit -- a refused action keeps looping so the decider can react.")


async def test_early_exit_not_after_lookup() -> None:
    _, calls = await _run([_resp(_tool_use("lookup_concept", {"problem": "harsh"}, "l1")),
                           _resp(_open("Channel EQ", "a1")), _resp(_text("done"))],
                          lookups=[{"match": "none"}], early_exit=True)
    assert len(calls) == 3, len(calls)
    history = [
        {"role": "user", "content": "no sound"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "l0", "name": "lookup_concept",
                                           "input": {"problem": "no sound"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "l0", "content": "{}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "check the output"}]},
        {"role": "user", "content": "spotify went silent too"},
    ]
    _, calls = await _run([_resp(_tool_use("open_setting", {"name": "audio settings"}, "a1")),
                           _resp(_text("core audio"))], messages=history, early_exit=True)
    assert len(calls) == 2, len(calls)
    print("PASS: early exit -- not on a turn that looked something up, or right after one that did.")


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


async def test_route_and_action_share_the_card() -> None:
    result, _ = await _run([
        _resp(_tool_use("open_setting", {"name": "buffer size"}, "w1"), _open("Compressor", "a1")),
        _resp(_text("done")),
    ])
    assert result.walkthrough_steps == _route("buffer size") + [{"ax_open_plugin": "Compressor"}], result.walkthrough_steps
    assert result.card_from_lookup is False
    print("PASS: a route and a plugin action share one card, in call order.")


async def test_unknown_route_refused() -> None:
    result, _ = await _run([_resp(_tool_use("open_setting", {"name": "frame rate"}, "w1")), _resp(_text("done"))])
    out = next(c["output"] for c in result.trace if c["tool"] == "open_setting")
    assert out["attached"] is False and "approved route" in out["reason"], out
    assert result.walkthrough_steps is None
    print("PASS: open_setting refuses a name that isn't an approved route.")


async def test_auto_attach_strong_single() -> None:
    lookups = [_single("buffer size", {"open_setting": "buffer size"})]
    result, calls = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "buffer size"}, "l1")),
        _resp(_text("It's the I/O buffer size.")),
    ], lookups=lookups, messages=[{"role": "user", "content": "raise my buffer size"}])
    assert result.walkthrough_steps == _route("buffer size"), result.walkthrough_steps
    assert result.card_from_lookup is True and result.auto_run is False, (result.card_from_lookup, result.auto_run)
    look = next(c["output"] for c in result.trace if c["tool"] == "lookup_concept")
    assert look["on_card_note"] == pipeline._ON_CARD_NOTE + pipeline._PANE_ONLY_NOTE, look  # a dropdown route
    assert len(calls) == 2, "no extra decider call on an auto-attached turn"
    print("PASS: a strong single lookup queues its solution's action; a lookup-queued card never auto-runs.")


async def test_moderate_single_is_pick_or_ask() -> None:
    lookups = [_single("recording settings", {"open_setting": "recording settings"}, conf="moderate — treat with skepticism")]
    result, calls = await _run([_resp(_tool_use("lookup_concept", {"problem": "x"}, "l1")), _resp(_text("hm")),
                                _resp(_tool_use("open_setting", {"name": "recording settings"}, "w1")), _resp(_text("ok"))],
                               lookups=lookups)
    assert calls[2]["tool_choice"] == {"type": "any"}, calls[2].get("tool_choice")
    assert result.walkthrough_steps == _route("recording settings") and result.card_from_lookup is False
    with patch.object(pipeline, "PICK_ON_MODERATE_SINGLE", False):
        result, calls = await _run([_resp(_tool_use("lookup_concept", {"problem": "x"}, "l1")), _resp(_text("hm"))],
                                   lookups=lookups)
    assert result.walkthrough_steps is None and len(calls) == 2
    print("PASS: a moderate single isn't auto-attached but gets pick-or-ask (flag restores prose-only).")


async def test_route_toggle_gate() -> None:
    lookups = [_single("input monitoring toggle", {"open_setting": "input monitoring toggle"})]
    result, _ = await _run([_resp(_tool_use("lookup_concept", {"problem": "monitor button"}, "l1")), _resp(_text("ok"))],
                           lookups=lookups, ax_fixture={"input_monitoring_button_visible": True})
    look = next(c["output"] for c in result.trace if c["tool"] == "lookup_concept")
    assert result.walkthrough_steps is None and look["on_card_note"].startswith("This solution's action was NOT queued"), look
    print("PASS: the toggle gate on a route refuses an auto-attach that would switch it away.")


async def test_pick_or_ask() -> None:
    crackling = [_bucket("crackling", {"CPU overload": {"open_setting": "buffer size"},
                                       "sample rate mismatch": {"open_setting": "sample rate"}})]
    # Prose-only answer after the bucket -> one forced re-ask, which picks.
    result, calls = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        _resp(_text("It's CPU overload -- raise the buffer.")),
        _resp(_tool_use("open_setting", {"name": "buffer size"}, "w1")),
        _resp(_text("Raise the buffer size.")),
    ], lookups=crackling)
    assert calls[2]["tool_choice"] == {"type": "any"} and pipeline._PICK_NUDGE in calls[2]["system"][-1]["text"], calls[2]
    assert result.walkthrough_steps == _route("buffer size"), result.walkthrough_steps
    # Asked once only: a second prose-only answer ends the turn with no card.
    result, calls = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        _resp(_text("prose")), _resp(_text("still prose")),
    ], lookups=crackling)
    assert result.walkthrough_steps is None and len(calls) == 3, (result.walkthrough_steps, len(calls))
    # A clarifying question satisfies it.
    result, calls = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        _resp(_tool_use("ask_clarifying_question", {}, "q1")),
    ], lookups=crackling)
    assert len(calls) == 2, len(calls)
    print("PASS: after a bucket lookup the turn can't end on prose alone -- one forced re-ask; a question counts.")


async def test_alternative_pick_refused_separate_request_joins() -> None:
    crackling = [_bucket("crackling", {"CPU overload": {"open_setting": "buffer size"},
                                       "sample rate mismatch": {"open_setting": "sample rate"}})]
    result, _ = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        _resp(_tool_use("open_setting", {"name": "buffer size"}, "w1"), _tool_use("open_setting", {"name": "sample rate"}, "w2")),
        _resp(_text("done")),
    ], lookups=crackling)
    outs = [c["output"] for c in result.trace if c["tool"] == "open_setting"]
    assert outs[0]["attached"] and outs[1] == {"attached": False, "reason": pipeline._FALLBACK_REFUSAL}, outs
    two = [_bucket("crackling", {"CPU overload": {"open_setting": "buffer size"}}),
           _bucket("freezing", {"track freeze (technique)": {"open_setting": "freeze track"}})]
    result, _ = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1"), _tool_use("lookup_concept", {"problem": "freeze"}, "l2")),
        _resp(_tool_use("open_setting", {"name": "buffer size"}, "w1"), _tool_use("open_setting", {"name": "freeze track"}, "w2")),
        _resp(_text("done")),
    ], lookups=two)
    assert result.walkthrough_steps == _route("buffer size") + _route("freeze track"), result.walkthrough_steps
    print("PASS: a second candidate from the same bucket is refused; a separate request joins the card.")


async def test_resume_restores_lookup_queued_action() -> None:
    steps = _route("buffer size")
    look = {**_single("buffer size", {"open_setting": "buffer size"}),
            "on_card": [{"tool": "open_setting", "input": {"name": "buffer size"},
                         "output": {"attached": True, "destination": "buffer size", "steps": steps}}]}
    parked = [
        {"role": "user", "content": "raise my buffer and tell me about 1176s"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "l1", "name": "lookup_concept", "input": {"problem": "buffer size"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "l1", "content": json.dumps(look)}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "web_research", "input": {"query": "1176"}}]},
    ]
    with patch.object(pipeline.research, "web_research", new=AsyncMock(return_value={"findings": "x", "sources": []})):
        result, _ = await _run([_resp(_text("done"))], messages=parked, resume="allow_research")
    assert result.walkthrough_steps == steps and result.card_from_lookup is True, (result.walkthrough_steps, result.card_from_lookup)
    print("PASS: a resumed turn keeps the action its lookup queued, and still doesn't auto-run it.")


async def test_endorsing_a_lookup_queued_action() -> None:
    # The model calls the action the lookup already queued: agreement, not a
    # second request -- its version (with the track) replaces the queued step,
    # nothing is refused, and the card is model-called so it may auto-run.
    lookups = [_single("channel eq", {"open_plugin": {"plugin": "Channel EQ"}})]
    result, _ = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "channel eq"}, "l1")),
        _resp(_open("Channel EQ", "a1", track="Audio 1")),
        _resp(_text("done")),
    ], lookups=lookups, messages=[{"role": "user", "content": "open channel eq"}])
    assert result.walkthrough_steps == [{"ax_open_plugin": "Channel EQ", "track": "Audio 1"}], result.walkthrough_steps
    opens = [c["output"] for c in result.trace if c["tool"] == "open_plugin"]
    assert opens[0].get("attached"), opens[0]                    # the lookup's own queue
    assert opens[-1].get("already_queued"), opens[-1]            # the model endorsing it
    # The step is model-called now (card_from_lookup clears), but the turn still
    # consulted the KB, so the card waits either way (see respond()).
    assert result.card_from_lookup is False and result.auto_run is False, (result.card_from_lookup, result.auto_run)
    print("PASS: a direct call for an already-queued action replaces it and isn't refused.")


async def test_a_lookup_turn_never_auto_runs() -> None:
    # 2026-09-22: on a turn whose own answer was "I don't have a verified fix",
    # the model queued a global settings change off a moderate match and it was
    # cleared to auto-run. Anything the KB led to waits, however it got queued.
    lookups = [_bucket("track deselects", {"bypass control surfaces": {"open_setting": "bypass control surfaces"}})]
    result, _ = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "track deselects on plugin click"}, "l1")),
        _resp(_tool_use("open_setting", {"name": "bypass control surfaces"}, "w1")),
        _resp(_text("try disabling control surfaces")),
    ], lookups=lookups, messages=[{"role": "user", "content": "logic randomly deselects my track when i click a knob"}])
    assert result.walkthrough_steps and result.auto_run is False, (result.walkthrough_steps, result.auto_run)
    # The same action, asked for outright with no lookup, still auto-runs.
    result, _ = await _run([_resp(_tool_use("open_setting", {"name": "bypass control surfaces"}, "w1")), _resp(_text("ok"))],
                           messages=[{"role": "user", "content": "turn off control surfaces"}])
    assert result.auto_run is True, result.auto_run
    print("PASS: a card the KB led to waits for Run; the same action asked for directly auto-runs.")


async def test_wait_for_run_route_never_auto_runs() -> None:
    # A route whose target is whatever the user has selected (`wait_for_run`)
    # waits even on a direct command, so "select the region first" comes first.
    result, _ = await _run([_resp(_tool_use("open_setting", {"name": "follow tempo", "value": "Off"}, "w1")),
                            _resp(_text("ok"))],
                           messages=[{"role": "user", "content": "set smart tempo off on this region"}])
    assert result.walkthrough_steps and result.auto_run is False, (result.walkthrough_steps, result.auto_run)
    print("PASS: a wait_for_run route waits for Run on a direct command.")


def test_offered_text() -> None:
    # The reply the user answers offers its options, question or statement
    # (live 2026-09-28: "you can choose between X or Y." then "the last one").
    from app.pipeline import _offered_text
    q = [{"role": "user", "content": "what are my smart tempo options"},
         {"role": "assistant", "content": "You can choose between On + Align Bars or On + Align Bars and Beats."},
         {"role": "user", "content": "the last one"}]
    assert _offered_text(q).startswith("you can choose"), _offered_text(q)
    assert _offered_text(q[:1]) == ""   # a first turn has nothing offered
    print("PASS: the reply the user answers offers its options to the next turn; a first turn has none.")


async def test_weak_bucket_does_not_force_a_pick() -> None:
    weak = [{"match": "problem", "problem": "something else", "match_confidence": "weak — likely not relevant",
             "solutions": [{"name": "buffer size", "action": {"open_setting": "buffer size"}}]}]
    result, calls = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "phantom power"}, "l1")),
        _resp(_text("I don't have a verified answer for that.")),
    ], lookups=weak)
    assert result.walkthrough_steps is None and len(calls) == 2, (result.walkthrough_steps, len(calls))
    print("PASS: a weak match never forces a pick -- an honest 'no verified answer' stands.")


async def test_auto_attach_respects_commit_to_one() -> None:
    # Bucket pick first, then the model looks the sibling candidate up by name
    # (a strong single): its action must NOT slip onto the card.
    lookups = [
        _bucket("crackling", {"CPU overload": {"open_setting": "buffer size"},
                              "sample rate mismatch": {"open_setting": "sample rate"}}),
        _single("sample rate mismatch", {"open_setting": "sample rate"}),
    ]
    result, _ = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        _resp(_tool_use("open_setting", {"name": "buffer size"}, "w1")),
        _resp(_tool_use("lookup_concept", {"problem": "sample rate mismatch"}, "l2")),
        _resp(_text("done")),
    ], lookups=lookups)
    assert result.walkthrough_steps == _route("buffer size"), result.walkthrough_steps
    second = [c for c in result.trace if c["tool"] == "lookup_concept"][1]["output"]
    assert second["on_card"][0]["output"] == {"attached": False, "reason": pipeline._FALLBACK_REFUSAL}, second
    print("PASS: a lookup can't auto-queue a second candidate for a problem already on the card.")


def test_card_descriptions() -> None:
    trace = [
        {"tool": "open_plugin", "input": {"plugin": "Channel EQ", "track": "Audio 1"}, "output": {"attached": True}},
        {"tool": "set_param", "input": {"plugin": "Channel EQ", "param": "Low Cut Frequency", "value": "80"},
         "output": {"attached": True}},
        {"tool": "open_setting", "input": {"name": "buffer size"}, "output": {"attached": True}},
        {"tool": "open_plugin", "input": {"plugin": "Compressor"}, "output": {"attached": False, "reason": "x"}},
    ]
    lines = pipeline.card_descriptions(trace)
    assert lines[0] == "opens Channel EQ on Audio 1", lines
    assert lines[1] == "sets Channel EQ Low Cut Frequency to 80", lines
    assert lines[2].startswith("opens buffer size -- Audio settings pane"), lines
    assert len(lines) == 3, "a refused call is not on the card"
    print("PASS: the writer's card description covers every queued step and nothing else.")


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


async def test_lookup_cap() -> None:
    limit = pipeline.LOOKUP_ATTEMPT_LIMIT
    # Default (COUNT_ALL_LOOKUPS): every lookup counts, whatever it returned --
    # restored 2026-09-22 after distinct moderate wrong buckets ("productive"
    # under the other rule) let an off-KB turn run past the cap into research.
    distinct = [{"match": "problem", "problem": f"bucket {i}", "match_confidence": "strong"} for i in range(limit + 1)]
    with patch.object(pipeline, "MAX_ITERATIONS", limit + 3):
        result, _ = await _run(_lookups(limit + 1) + [_resp(_text("done"))], lookups=distinct)
    outs = _lookup_outputs(result)
    assert "error" in outs[-1] and all("error" not in o for o in outs[:limit]), outs

    # The unproductive-only variant (flag off): distinct real buckets don't count...
    with patch.object(pipeline, "COUNT_ALL_LOOKUPS", False), patch.object(pipeline, "MAX_ITERATIONS", limit + 3):
        result, _ = await _run(_lookups(limit + 1) + [_resp(_text("done"))], lookups=distinct)
        assert all("error" not in o for o in _lookup_outputs(result)), _lookup_outputs(result)
        # ...but no-match ones do.
        result, _ = await _run(_lookups(limit + 1) + [_resp(_text("done"))], lookups=[{"match": "none"}])
        outs = _lookup_outputs(result)
        assert "error" in outs[-1] and all("error" not in o for o in outs[:-1]), outs
    print("PASS: the cap counts every lookup; COUNT_ALL_LOOKUPS=False counts only unproductive ones.")


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


def _setting(name: str, value: str | None = None, said: str = "", offered: str = "") -> dict:
    token, token2 = tools.TURN_USER_TEXT.set(said), tools.TURN_OFFERED_TEXT.set(offered)
    try:
        return tools.queue_open_setting({"name": name, **({"value": value} if value else {})})
    finally:
        tools.TURN_USER_TEXT.reset(token)
        tools.TURN_OFFERED_TEXT.reset(token2)


def test_dropdown_routes() -> None:
    # No value: the pane only -- the dropdown is never clicked open (found live
    # 2026-09-22: the open menu swallowed the next step's keystrokes).
    out = _setting("buffer size")
    assert out["steps"] == [{"menu_path": ["Logic Pro", "Settings", "Audio"]}], out
    assert "pane only" in out["note"], out
    # A value the user named: the route, then a choose step carrying the route for revert.
    out = _setting("buffer size", "256 samples", said="set my buffer to 256")
    assert out["chooses"] == "256" and out["steps"][-1]["choose"] == "256", out
    assert out["steps"][-1]["reopen"] == out["steps"][:-1] == [
        {"menu_path": ["Logic Pro", "Settings", "Audio"]}, {"click_value_of": "I/O Buffer Size"}], out
    # A number the user didn't name is refused; a direction is the model's to pick.
    out = _setting("buffer size", "1024", said="my audio keeps crackling")
    assert out["attached"] is False and "larger or smaller" in out["reason"], out
    assert _setting("buffer size", "larger", said="my audio keeps crackling")["chooses"] == "larger"
    assert _setting("buffer size", "Smaller")["chooses"] == "smaller"
    out = _setting("buffer size", "2048", said="set buffer to 2048")
    assert out["attached"] is False and "isn't one of" in out["reason"], out
    # Sample rate: named in any common spelling, never chosen for the user, no direction.
    assert _setting("sample rate", "48000", said="change the sample rate to 48k")["chooses"] == "48 kHz"
    assert _setting("sample rate", "44.1", said="set it to 44100")["chooses"] == "44.1 kHz"
    assert _setting("sample rate", "48 kHz", said="my file sounds slowed down")["attached"] is False
    assert _setting("sample rate", "larger", said="set sample rate higher")["attached"] is False
    # Options the model may pick itself.
    # Threads: the KB says max rather than Automatic on Apple Silicon, and the
    # route can't pick a number yet -- Automatic only when the user names it.
    assert _setting("processing threads", "automatic", said="system overload on my M3")["attached"] is False
    assert _setting("processing threads", "automatic", said="set processing threads to automatic")["chooses"] == "Automatic"
    assert _setting("flex time", "mono", said="the vocal timing is off")["chooses"] == "Monophonic"
    assert _setting("flex time", "Speed (FX)", said="fix the timing")["attached"] is False
    assert _setting("flex time", "Speed (FX)", said="set flex to speed")["chooses"] == "Speed (FX)"
    # Region Smart Tempo: pane only keeps the dropped row as the Region click's
    # `expect` (the client skips the toggling click when the row already shows).
    assert _setting("follow tempo")["steps"] == [
        {"shortcut": "I"}, {"click_text": "Region", "expect": ["Smart Tempo"]}]
    assert _setting("follow tempo", "off", said="stop the region stretching")["chooses"] == "Off"
    assert _setting("follow tempo", "On", said="fix the stretching")["attached"] is False
    out = _setting("follow tempo", "On + Align Bars", said="set smart tempo to on + align bars")
    assert out["steps"][-1]["shows"] == {"On + Align Bars": "Bars", "On + Align Bars and Beats": "Beats"}, out
    assert "shows" not in _setting("buffer size", "256", said="set my buffer to 256")["steps"][-1]
    assert out["chooses"] == "On + Align Bars" and out["steps"][-1]["reopen"] == [
        {"shortcut": "I"}, {"click_text": "Region"}, {"click_value_of": "Smart Tempo"}], out
    # What the control displays names the option ("beats" -> On + Align Bars and Beats).
    assert _setting("follow tempo", "beats", said="set smart tempo to beats")["chooses"] == "On + Align Bars and Beats"
    assert _setting("follow tempo", "On + Align Bars", said="set it to bars")["chooses"] == "On + Align Bars"
    # An option the previous reply offered is the user's pick once they answer --
    # they don't have to type it out. Without the offer it's still refused.
    offer = "which one do you want: on, on + align bars, or on + align bars and beats?"
    assert _setting("follow tempo", "On + Align Bars and Beats", said="the last one", offered=offer)["chooses"] == \
        "On + Align Bars and Beats"
    assert _setting("buffer size", "1024", said="yes", offered="want me to set it to 1024?")["chooses"] == "1024"
    assert _setting("follow tempo", "On + Align Bars and Beats", said="the last one")["attached"] is False
    assert _setting("sample rate", "48 kHz", said="still slow", offered="")["attached"] is False
    # A pane-only route whose kept last step is a menu gets no expect.
    assert "expect" not in _setting("buffer size")["steps"][-1]
    # A plain route takes no value.
    out = _setting("mixer", "on")
    assert out["attached"] is False and "no value" in out["reason"], out
    assert "note" not in _setting("mixer") and _setting("mixer")["steps"] == [{"shortcut": "X"}]
    print("PASS: dropdown routes pick a checked value or stop at the pane; plain routes unchanged.")


async def test_dropdown_value_replaces_lookup_queued_pane() -> None:
    # A lookup queues buffer size (pane only). The model tries an exact number
    # the user never said -- refused, and the queued step stays put -- then
    # sends a direction, which replaces the queued step instead of joining it.
    lookups = [_single("CPU overload", {"open_setting": "buffer size"})]
    result, _ = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        _resp(_tool_use("open_setting", {"name": "buffer size", "value": "1024"}, "w1")),
        _resp(_tool_use("open_setting", {"name": "buffer size", "value": "larger"}, "w2")),
        _resp(_text("done")),
    ], lookups=lookups, messages=[{"role": "user", "content": "my playback keeps crackling"}])
    look = next(c["output"] for c in result.trace if c["tool"] == "lookup_concept")
    assert pipeline._PANE_ONLY_NOTE in look["on_card_note"], look
    sets = [c for c in result.trace if c["tool"] == "open_setting" and not c.get("auto_from")]
    assert sets[0]["output"]["attached"] is False and sets[1]["output"].get("already_queued"), sets
    assert result.walkthrough_steps == _setting("buffer size", "larger")["steps"], result.walkthrough_steps
    print("PASS: a refused dropdown value leaves the lookup's pane step queued; a valid one replaces it.")


async def test_dropdown_value_is_still_an_alternative() -> None:
    # A value doesn't make a second candidate from the same bucket a new request.
    crackling = [_bucket("crackling", {"CPU overload": {"open_setting": "buffer size"},
                                       "sample rate mismatch": {"open_setting": "sample rate"}})]
    result, _ = await _run([
        _resp(_tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        _resp(_tool_use("open_setting", {"name": "sample rate"}, "w1"),
              _tool_use("open_setting", {"name": "buffer size", "value": "larger"}, "w2")),
        _resp(_text("done")),
    ], lookups=crackling, messages=[{"role": "user", "content": "crackling"}])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_setting"]
    assert outs[0]["attached"] and outs[1] == {"attached": False, "reason": pipeline._FALLBACK_REFUSAL}, outs
    print("PASS: commit-to-one still refuses a same-bucket alternative that carries a value.")


async def test_one_visit_per_setting() -> None:
    # Two values for the same dropdown in one reply: the later call replaces
    # the earlier one (found in the 2026-09-22 battery: both ran; then with
    # "first stands", the card said Rhythmic while the reply said Monophonic).
    result, _ = await _run([
        _resp(_tool_use("open_setting", {"name": "flex time", "value": "Rhythmic"}, "w1"),
              _tool_use("open_setting", {"name": "buffer size"}, "w2"),
              _tool_use("open_setting", {"name": "flex time", "value": "Monophonic"}, "w3")),
        _resp(_text("done")),
    ], messages=[{"role": "user", "content": "flex my vocal"}])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_setting"]
    assert outs[0].get("replaced") and outs[0]["attached"] is False, outs[0]
    assert outs[2]["chooses"] == "Monophonic" and outs[2].get("replaces_earlier"), outs[2]
    want = _setting("flex time", "Monophonic")["steps"] + _route("buffer size")
    assert result.walkthrough_steps == want, result.walkthrough_steps
    assert [ln.split(" -- ")[0] for ln in pipeline.card_descriptions(result.trace)] == ["opens buffer size", "opens flex time"]
    print("PASS: a second call for a setting already on the card replaces it in place; one value per setting.")


async def test_resume_keeps_replacements() -> None:
    # A turn parked for research after endorsing a lookup's pane-only route
    # with a value and re-visiting a setting: the resumed card is the one the
    # live turn had built, not the lookup's original steps.
    pane = _route("buffer size")
    look = {**_single("CPU overload", {"open_setting": "buffer size"}),
            "on_card": [{"tool": "open_setting", "input": {"name": "buffer size"},
                         "output": {"attached": True, "destination": "buffer size", "steps": pane}}]}
    larger = {**_setting("buffer size", "larger"), "already_queued": True}
    rhythmic = _setting("flex time", "Rhythmic", said="rhythmic")
    mono = {**_setting("flex time", "Monophonic"), "replaces_earlier": True}

    def call(id_, name, inp):
        return {"role": "assistant", "content": [{"type": "tool_use", "id": id_, "name": name, "input": inp}]}

    def res(id_, out):
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id_, "content": json.dumps(out)}]}
    parked = [
        {"role": "user", "content": "crackles, flex my vocal, and tell me about 1176s"},
        call("l1", "lookup_concept", {"problem": "crackling"}), res("l1", look),
        call("w1", "open_setting", {"name": "buffer size", "value": "larger"}), res("w1", larger),
        call("w2", "open_setting", {"name": "flex time", "value": "Rhythmic"}), res("w2", rhythmic),
        call("w3", "open_setting", {"name": "flex time", "value": "Monophonic"}), res("w3", mono),
        call("r1", "web_research", {"query": "1176"}),
    ]
    with patch.object(pipeline.research, "web_research", new=AsyncMock(return_value={"findings": "x", "sources": []})):
        result, _ = await _run([_resp(_text("done"))], messages=parked, resume="allow_research")
    assert result.walkthrough_steps == larger["steps"] + mono["steps"], result.walkthrough_steps
    print("PASS: a resumed turn rebuilds endorsed and replaced steps the way the live turn had them.")


def test_card_descriptions_dropdowns() -> None:
    trace = [
        {"tool": "open_setting", "input": {"name": "buffer size", "value": "larger"},
         "output": {"attached": True, "chooses": "larger"}},
        {"tool": "open_setting", "input": {"name": "sample rate", "value": "48k"},
         "output": {"attached": True, "chooses": "48 kHz"}},
        {"tool": "open_setting", "input": {"name": "processing threads"}, "output": {"attached": True}},
    ]
    lines = pipeline.card_descriptions(trace)
    assert lines[0].endswith("then moves it one step larger than whatever it's set to now"), lines
    assert lines[1].endswith("then sets it to 48 kHz"), lines
    assert lines[2].endswith("(opens the pane only -- no value is chosen; the user picks there)"), lines
    print("PASS: the writer is told whether a dropdown route picks a value or stops at the pane.")


async def main() -> None:
    test_executor_validation()
    test_dropdown_routes()
    test_offered_text()
    test_card_descriptions_dropdowns()
    await test_dropdown_value_replaces_lookup_queued_pane()
    await test_dropdown_value_is_still_an_alternative()
    await test_one_visit_per_setting()
    await test_resume_keeps_replacements()
    test_is_question()
    test_card_descriptions()
    await test_actions_combine_same_response()
    await test_actions_combine_across_iterations()
    await test_early_exit_direct_action()
    await test_early_exit_not_on_refusal()
    await test_early_exit_not_after_lookup()
    await test_action_cap()
    await test_duplicate_action_refused()
    await test_route_and_action_share_the_card()
    await test_unknown_route_refused()
    await test_auto_attach_strong_single()
    await test_moderate_single_is_pick_or_ask()
    await test_route_toggle_gate()
    await test_pick_or_ask()
    await test_alternative_pick_refused_separate_request_joins()
    await test_resume_restores_lookup_queued_action()
    await test_endorsing_a_lookup_queued_action()
    await test_a_lookup_turn_never_auto_runs()
    await test_wait_for_run_route_never_auto_runs()
    await test_weak_bucket_does_not_force_a_pick()
    await test_auto_attach_respects_commit_to_one()
    await test_auto_run()
    await test_first_call_is_any()
    await test_lookup_cap()
    await test_resume_restores_actions()


if __name__ == "__main__":
    asyncio.run(main())
