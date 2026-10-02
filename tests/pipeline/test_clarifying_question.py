"""ask_clarifying_question ends the turn, and its `question` is the reply's text when the decider wrote none."""
from __future__ import annotations

import importlib
from unittest.mock import patch

from conftest import resp, text_block, tool_use

# app.pipeline re-exports the respond() function under the module's name.
respond_mod = importlib.import_module("app.pipeline.respond")

QUESTION = "Which track is the vocal on: 1) Audio 1, 2) Audio 2?"


async def _finalize_text(run, responses: list) -> tuple[str, list]:
    """Run a turn and return the decider text handed to the writer, plus the decider calls."""
    real = respond_mod.finalize_answer
    with patch.object(respond_mod, "finalize_answer", side_effect=real) as spy:
        _, calls = await run(responses)
    return spy.call_args.args[2], calls


async def test_question_used_when_no_text(run) -> None:
    """A bare ask_clarifying_question call ends the turn and its question goes to the writer."""
    text, calls = await _finalize_text(run, [resp(tool_use("ask_clarifying_question", {"question": QUESTION}, "q1")),
                                             resp(text_block("more"))])
    assert len(calls) == 1, len(calls)
    assert text == QUESTION, text


async def test_written_text_wins_over_question(run) -> None:
    """When the decider also wrote the question as text, that text stands and isn't doubled."""
    text, _ = await _finalize_text(run, [resp(text_block("Which track?"),
                                              tool_use("ask_clarifying_question", {"question": QUESTION}, "q1"))])
    assert text == "Which track?", text
