"""The research-confirm two-phase turn: parking at web_research, resuming on allow or deny,
and the api's check that a history really is parked (app.api.history.pending_research_query)."""
from __future__ import annotations

import json

import pytest

from app.api.history import pending_research_query
from app.pipeline import notes
from conftest import resp, text_block, tool_use

USER = [{"role": "user", "content": "what compressor settings did serban ghenea use on that record"}]
RESEARCH_CALL = resp(text_block("Let me look that up."),
                     tool_use("web_research", {"query": "serban ghenea compressor settings"}, "call_r"))
FINAL = resp(text_block("Here is what I found."))


def parked_history_with_prior_attach() -> list[dict]:
    """A parked transcript where the decider attached a walkthrough (via a
    successful open_setting) BEFORE deciding to research something else."""
    return USER + [
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


@pytest.fixture(autouse=True)
def _findings(research):
    research.return_value = {"findings": "Found it.", "sources": [{"url": "https://x", "title": "x"}]}


async def test_parks_on_research_when_confirm_requested(run, research) -> None:
    """research_confirm parks the turn at the web_research call, nothing executed."""
    result, _ = await run([RESEARCH_CALL, FINAL], messages=USER, research_confirm=True)
    assert result.pending_research_query == "serban ghenea compressor settings", result.pending_research_query
    assert result.text == "", "no answer should be written while parked"
    assert research.await_count == 0, "research must not run before the user approves"
    last = result.messages[-1]
    assert last["role"] == "assistant" and any(
        b.get("type") == "tool_use" and b.get("name") == "web_research" for b in last["content"]
    ), f"parked transcript must end in the pending tool_use: {last}"
    # Nothing has been answered, so no tool_result for it yet.
    assert not any(m["role"] == "user" and isinstance(m["content"], list) for m in result.messages[1:])


async def test_resume_deny_skips_research_and_restores_state(run, research) -> None:
    """deny_research injects the decline, restores trace + attach, finishes the turn."""
    parked = parked_history_with_prior_attach()
    result, _ = await run([FINAL], messages=parked, resume="deny_research", research_confirm=True)
    assert research.await_count == 0, "deny must never run the research executor"
    tools_called = [c["tool"] for c in result.trace]
    assert tools_called == ["cite_kb", "open_setting", "web_research"], tools_called
    assert result.trace[-1]["output"] == notes.RESEARCH_DECLINED, result.trace[-1]["output"]
    assert result.walkthrough_steps == [["menu", "File"]], "attach made before the prompt must survive the resume"
    # The model sees the decline as a normal tool_result, so the transcript is API-valid.
    tr = result.messages[len(parked)]
    assert tr["role"] == "user" and tr["content"][0]["tool_use_id"] == "call_r", tr
    assert json.loads(tr["content"][0]["content"]) == notes.RESEARCH_DECLINED
    assert result.pending_research_query is None and result.text, "deny must finish the turn with an answer"


async def test_resume_allow_runs_research_with_parked_query(run, research) -> None:
    """allow_research runs the executor with the parked query and finishes the turn."""
    parked = parked_history_with_prior_attach()
    result, _ = await run([FINAL], messages=parked, resume="allow_research", research_confirm=True)
    assert research.await_count == 1, research.await_count
    assert research.await_args.args == ("serban ghenea compressor settings",), research.await_args
    assert result.trace[-1]["tool"] == "web_research" and result.trace[-1]["output"].get("findings")
    assert result.pending_research_query is None and result.text
    # Tier grades off the earlier attach (a card is "strong"), not the research aside.
    assert result.confidence_tier == "strong", result.confidence_tier


async def test_no_confirm_runs_research_inline(run, research) -> None:
    """Without research_confirm (older clients), research runs inline as before."""
    result, _ = await run([RESEARCH_CALL, FINAL], messages=USER)
    assert research.await_count == 1, "an old client (no research_confirm) must get the pre-existing behaviour"
    assert result.pending_research_query is None and result.text


def test_pending_query_detection() -> None:
    """pending_research_query only accepts a transcript ending in an unanswered web_research call."""
    assert pending_research_query(None) is None
    assert pending_research_query(USER) is None
    assert pending_research_query(parked_history_with_prior_attach()) == "serban ghenea compressor settings"
    finished = parked_history_with_prior_attach() + [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_r", "content": "{}"}]},
    ]
    assert pending_research_query(finished) is None, "a transcript with the result already in it is not pending"
