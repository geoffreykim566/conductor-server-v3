"""Deterministic tests for the research-confirm two-phase turn (2026-09-13).

Mocks the decider (pipeline._client.messages.create) and the research
executor (pipeline.research.web_research) so the gate/resume logic is tested
independently of whether the live model would call web_research at all.

  1. research_confirm=True + decider calls web_research -> turn parks:
     nothing executed, Result.pending_research_query set, messages end in
     the pending tool_use.
  2. resume="deny_research" -> executor never runs, the decider gets
     _RESEARCH_DECLINED as the tool_result, the turn finishes; trace and an
     earlier walkthrough attach are restored from the parked history.
  3. resume="allow_research" -> executor runs with the parked query.
  4. research_confirm omitted (installed v0.3.1 clients) -> research runs
     inline, no parking.
  5. api.py: resume with no pending web_research in history -> None from
     _pending_research_query (the endpoint 422s on that).

No DB needed: every model response is scripted, so cite_kb never
actually executes. Run inside the app container:
    docker compose exec app python -m app.test_research_confirm
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import pipeline
from app.api import _pending_research_query


def _usage() -> SimpleNamespace:
    return SimpleNamespace(input_tokens=10, output_tokens=10,
                           cache_creation_input_tokens=0, cache_read_input_tokens=0)


def _tool_use(name: str, input_: dict, id_: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


async def _run(messages: list[dict], responses: list, **kwargs):
    """Drive respond() through scripted decider responses; return
    (result, research_mock)."""
    call_count = 0

    async def fake_create(**_):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    research_mock = AsyncMock(return_value={"findings": "Found it.", "sources": [{"url": "https://x", "title": "x"}]})
    with patch.object(pipeline._client.messages, "create", new=AsyncMock(side_effect=fake_create)), \
         patch.object(pipeline.research, "web_research", new=research_mock):
        result = await pipeline.respond(messages, **kwargs)
    return result, research_mock


_USER = [{"role": "user", "content": "what compressor settings did serban ghenea use on that record"}]
_RESEARCH_CALL = SimpleNamespace(
    content=[_text("Let me look that up."), _tool_use("web_research", {"query": "serban ghenea compressor settings"}, "call_r")],
    usage=_usage(),
)
_FINAL = SimpleNamespace(content=[_text("Here is what I found.")], usage=_usage())


async def test_parks_on_research_when_confirm_requested() -> None:
    result, research = await _run(_USER, [_RESEARCH_CALL, _FINAL], research_confirm=True)
    assert result.pending_research_query == "serban ghenea compressor settings", result.pending_research_query
    assert result.text == "", "no answer should be written while parked"
    assert research.await_count == 0, "research must not run before the user approves"
    last = result.messages[-1]
    assert last["role"] == "assistant" and any(
        b.get("type") == "tool_use" and b.get("name") == "web_research" for b in last["content"]
    ), f"parked transcript must end in the pending tool_use: {last}"
    # Nothing has been answered, so no tool_result for it yet.
    assert not any(m["role"] == "user" and isinstance(m["content"], list) for m in result.messages[1:])
    print("PASS: research_confirm parks the turn at the web_research call, nothing executed.")
    return result.messages


def _parked_history_with_prior_attach() -> list[dict]:
    """A parked transcript where the decider attached a walkthrough (via a
    successful open_setting) BEFORE deciding to research something else."""
    return _USER + [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_l", "name": "cite_kb", "input": {"entries": ["sample rate mismatch"]}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_l", "content": json.dumps({"match": "problem"})},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Sample rate is under Project Settings."},
            {"type": "tool_use", "id": "call_w", "name": "open_setting", "input": {"name": "sample rate"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_w",
             "content": json.dumps({"attached": True, "destination": "sample rate", "steps": [["menu", "File"]]})},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_r", "name": "web_research", "input": {"query": "serban ghenea compressor settings"}},
        ]},
    ]


async def test_resume_deny_skips_research_and_restores_state() -> None:
    parked = _parked_history_with_prior_attach()
    result, research = await _run(parked, [_FINAL], resume="deny_research", research_confirm=True)
    assert research.await_count == 0, "deny must never run the research executor"
    tools_called = [c["tool"] for c in result.trace]
    assert tools_called == ["cite_kb", "open_setting", "web_research"], tools_called
    assert result.trace[-1]["output"] == pipeline._RESEARCH_DECLINED, result.trace[-1]["output"]
    assert result.walkthrough_steps == [["menu", "File"]], "attach made before the prompt must survive the resume"
    # The model sees the decline as a normal tool_result, so the transcript is API-valid.
    tr = result.messages[len(parked)]
    assert tr["role"] == "user" and tr["content"][0]["tool_use_id"] == "call_r", tr
    assert json.loads(tr["content"][0]["content"]) == pipeline._RESEARCH_DECLINED
    assert result.pending_research_query is None and result.text, "deny must finish the turn with an answer"
    print("PASS: deny_research injects the decline, restores trace + attach, finishes the turn.")


async def test_resume_allow_runs_research_with_parked_query() -> None:
    parked = _parked_history_with_prior_attach()
    result, research = await _run(parked, [_FINAL], resume="allow_research", research_confirm=True)
    assert research.await_count == 1, research.await_count
    assert research.await_args.args == ("serban ghenea compressor settings",), research.await_args
    assert result.trace[-1]["tool"] == "web_research" and result.trace[-1]["output"].get("findings")
    assert result.pending_research_query is None and result.text
    # Tier grades off the earlier attach ("strong"), not the research aside --
    # _confidence_tier's documented mixed-KB-and-research rule.
    assert result.confidence_tier == "strong", result.confidence_tier
    print("PASS: allow_research runs the executor with the parked query and finishes the turn.")


async def test_no_confirm_runs_research_inline() -> None:
    result, research = await _run(_USER, [_RESEARCH_CALL, _FINAL])
    assert research.await_count == 1, "an old client (no research_confirm) must get the pre-existing behaviour"
    assert result.pending_research_query is None and result.text
    print("PASS: without research_confirm, research runs inline as before.")


def test_pending_query_detection() -> None:
    assert _pending_research_query(None) is None
    assert _pending_research_query(_USER) is None
    assert _pending_research_query(_parked_history_with_prior_attach()) == "serban ghenea compressor settings"
    finished = _parked_history_with_prior_attach() + [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_r", "content": "{}"}]},
    ]
    assert _pending_research_query(finished) is None, "a transcript with the result already in it is not pending"
    print("PASS: _pending_research_query only accepts a transcript ending in an unanswered web_research call.")


async def main() -> None:
    await test_parks_on_research_when_confirm_requested()
    await test_resume_deny_skips_research_and_restores_state()
    await test_resume_allow_runs_research_with_parked_query()
    await test_no_confirm_runs_research_inline()
    test_pending_query_detection()


if __name__ == "__main__":
    asyncio.run(main())
