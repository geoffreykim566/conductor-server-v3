"""Shared fakes: scripted model responses, KB/route helpers, and the `run` fixture
that drives one pipeline turn with the model client, lookup and research mocked.

Test modules import the builders with `from conftest import ...` (tests/ is on
sys.path under pytest's default import mode; keep this the only conftest.py)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app import pipeline, tools
from app.pipeline import executors, model_io

# --- model response builders -------------------------------------------------


def usage() -> SimpleNamespace:
    return SimpleNamespace(input_tokens=10, output_tokens=10,
                           cache_creation_input_tokens=0, cache_read_input_tokens=0)


def tool_use(name: str, input_: dict, id_: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def resp(*blocks) -> SimpleNamespace:
    """One messages.create response holding `blocks`."""
    return SimpleNamespace(content=list(blocks), usage=usage())


def open_plugin(plugin: str, id_: str, **extra) -> SimpleNamespace:
    return tool_use("open_plugin", {"plugin": plugin, **extra}, id_)


def set_param(plugin: str, param: str, value: str, id_: str) -> SimpleNamespace:
    return tool_use("set_param", {"plugin": plugin, "param": param, "value": value}, id_)


# --- lookup_concept outputs --------------------------------------------------


def single(solution: str, action: dict, conf: str = "strong") -> dict:
    """A single-solution lookup match whose solution maps to `action`."""
    return {"match": "single", "problem": solution, "match_confidence": conf,
            "solutions": [{"name": solution, "action": action}]}


def bucket(problem: str, cands: dict[str, dict]) -> dict:
    """A strong problem-bucket match: candidate name -> its action."""
    return {"match": "problem", "problem": problem, "match_confidence": "strong",
            "solutions": [{"name": n, "action": a} for n, a in cands.items()]}


def fake_lookup(outputs: list[dict]):
    """lookup_concept executor stub returning `outputs` in order (last one repeats)."""
    n = 0

    async def fake(inp: dict, fixture) -> dict:
        nonlocal n
        out = outputs[min(n, len(outputs) - 1)]
        n += 1
        return out
    return fake


# --- routes (real seed/routes.json, no DB) -----------------------------------


def route(name: str) -> list:
    """The pane-only steps open_setting queues for an approved route."""
    return tools.queue_open_setting({"name": name})["steps"]


def queue_setting(name: str, value: str | None = None, said: str = "", offered: str = "") -> dict:
    """open_setting's executor as if the user's message were `said` and the
    previous reply `offered` (the turn-text context vars the pipeline sets)."""
    token, token2 = tools.TURN_USER_TEXT.set(said), tools.TURN_OFFERED_TEXT.set(offered)
    try:
        return tools.queue_open_setting({"name": name, **({"value": value} if value else {})})
    finally:
        tools.TURN_USER_TEXT.reset(token)
        tools.TURN_OFFERED_TEXT.reset(token2)


# --- fixtures ----------------------------------------------------------------


class FakeModel:
    """Stands in for messages.create. Decider calls (they pass tools=) take the
    next scripted response, the last one repeating; the writer call (no tools=)
    gets a plain "writer text" reply and doesn't consume the script."""

    def __init__(self) -> None:
        self.responses: list = []
        self.decider_calls: list[dict] = []

    def script(self, responses: list) -> list[dict]:
        """Start a new turn's script; returns the list decider kwargs land in."""
        self.responses, self.decider_calls = list(responses), []
        return self.decider_calls

    async def create(self, **kw):
        if "tools" not in kw:
            return resp(text_block("writer text"))
        self.decider_calls.append(kw)
        return self.responses[min(len(self.decider_calls) - 1, len(self.responses) - 1)]


@pytest.fixture
def model():
    """Patches the Anthropic client for the test; script it via model.script()."""
    fake = FakeModel()
    with patch.object(model_io._client.messages, "create", new=AsyncMock(side_effect=fake.create)):
        yield fake


@pytest.fixture
def research():
    """web_research mocked for the test; set .return_value to change findings."""
    mock = AsyncMock(return_value={"findings": "x", "sources": []})
    with patch.object(executors.research, "web_research", new=mock):
        yield mock


@pytest.fixture
def run(model, research):
    """async run(responses, lookups=None, messages=None, **respond_kwargs) ->
    (Result, decider_calls). Drives pipeline.respond() through scripted decider
    responses with lookup_concept stubbed to return `lookups` in order."""
    async def _run(responses: list, lookups: list[dict] | None = None,
                   messages: list[dict] | None = None, **kwargs):
        calls = model.script(responses)
        stubs = {"lookup_concept": fake_lookup(lookups or [{"match": "none"}])}
        with patch.dict(executors.EXECUTORS, stubs):
            result = await pipeline.respond(messages or [{"role": "user", "content": "test"}], **kwargs)
        return result, calls
    return _run
