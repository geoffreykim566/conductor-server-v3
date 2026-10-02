"""Resuming a turn parked at a research prompt rebuilds the card it had queued."""
from __future__ import annotations

import json

from conftest import queue_setting, resp, route, single, text_block


def _call(id_: str, name: str, inp: dict) -> dict:
    return {"role": "assistant", "content": [{"type": "tool_use", "id": id_, "name": name, "input": inp}]}


def _result(id_: str, out: dict) -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id_, "content": json.dumps(out)}]}


async def test_resume_restores_actions(run) -> None:
    """A resumed turn keeps the action queued before it parked."""
    step = {"ax_open_plugin": "Compressor"}
    parked = [
        {"role": "user", "content": "put a compressor on this and tell me about 1176s"},
        _call("a1", "open_plugin", {"plugin": "Compressor"}),
        _result("a1", {"attached": True, "action": "open_plugin", "steps": [step]}),
        _call("r1", "web_research", {"query": "1176"}),
    ]
    result, _ = await run([resp(text_block("done"))], messages=parked, resume="allow_research")
    assert result.walkthrough_steps == [step], result.walkthrough_steps


async def test_resume_restores_lookup_queued_action(run) -> None:
    """A resumed turn keeps the action its lookup queued, and still doesn't auto-run it."""
    steps = route("buffer size")
    look = {**single("buffer size", {"open_setting": "buffer size"}),
            "on_card": [{"tool": "open_setting", "input": {"name": "buffer size"},
                         "output": {"attached": True, "destination": "buffer size", "steps": steps}}]}
    parked = [
        {"role": "user", "content": "raise my buffer and tell me about 1176s"},
        _call("l1", "lookup_concept", {"problem": "buffer size"}),
        _result("l1", look),
        _call("r1", "web_research", {"query": "1176"}),
    ]
    result, _ = await run([resp(text_block("done"))], messages=parked, resume="allow_research")
    assert result.walkthrough_steps == steps and result.card_from_lookup is True, (result.walkthrough_steps, result.card_from_lookup)


async def test_resume_keeps_replacements(run) -> None:
    """A resumed turn rebuilds endorsed and replaced steps the way the live turn
    had them, not the lookup's original pane-only steps."""
    pane = route("buffer size")
    look = {**single("CPU overload", {"open_setting": "buffer size"}),
            "on_card": [{"tool": "open_setting", "input": {"name": "buffer size"},
                         "output": {"attached": True, "destination": "buffer size", "steps": pane}}]}
    larger = {**queue_setting("buffer size", "larger"), "already_queued": True}
    rhythmic = queue_setting("flex time", "Rhythmic", said="rhythmic")
    mono = {**queue_setting("flex time", "Monophonic"), "replaces_earlier": True}
    parked = [
        {"role": "user", "content": "crackles, flex my vocal, and tell me about 1176s"},
        _call("l1", "lookup_concept", {"problem": "crackling"}), _result("l1", look),
        _call("w1", "open_setting", {"name": "buffer size", "value": "larger"}), _result("w1", larger),
        _call("w2", "open_setting", {"name": "flex time", "value": "Rhythmic"}), _result("w2", rhythmic),
        _call("w3", "open_setting", {"name": "flex time", "value": "Monophonic"}), _result("w3", mono),
        _call("r1", "web_research", {"query": "1176"}),
    ]
    result, _ = await run([resp(text_block("done"))], messages=parked, resume="allow_research")
    assert result.walkthrough_steps == larger["steps"] + mono["steps"], result.walkthrough_steps
