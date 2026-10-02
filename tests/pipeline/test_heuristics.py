"""Text heuristics on the user's message and the reply it answers (app.pipeline.heuristics)."""
from __future__ import annotations

import pytest

from app.pipeline.heuristics import is_question, is_undo_request, last_user_text, offered_text


def _said(t: str) -> str:
    return last_user_text([{"role": "user", "content": t}])


@pytest.mark.parametrize("t", [
    "how do i open channel eq", "what does ratio do", "why is my mix harsh, add something",
    "channel eq?", "show me how to open channel eq", "can i add a compressor here",
    "can you explain what a compressor does", "hey, how do i mute a track", "wheres the buffer size",
    "do i need a limiter", "is my vocal too loud",
])
def test_is_question(t: str) -> None:
    assert is_question(_said(t)), t


@pytest.mark.parametrize("t", [
    "open channel eq", "put valhalla on the vocal", "my vocal sounds muddy, fix it",
    "can you add a compressor", "could you set the low cut to 80?", "please add an eq",
    "yes please, do it for me", "do it", "set the threshold to -18",
])
def test_instruction_is_not_question(t: str) -> None:
    assert not is_question(_said(t)), t


@pytest.mark.parametrize("t", ["undo that", "revert", "Revert it.", "can you undo that", "put it back",
                               "change that back please"])
def test_is_undo_request(t: str) -> None:
    assert is_undo_request(_said(t)), t


@pytest.mark.parametrize("t", ["how do i undo a cut in logic", "how do i revert an audio region back to how it was",
                               "what does revert do", "undo that?", "set the buffer to 256"])
def test_not_undo_request(t: str) -> None:
    assert not is_undo_request(_said(t)), t


def test_offered_text() -> None:
    """The reply the user answers offers its options, question or statement; a first turn has none."""
    q = [{"role": "user", "content": "what are my smart tempo options"},
         {"role": "assistant", "content": "You can choose between On + Align Bars or On + Align Bars and Beats."},
         {"role": "user", "content": "the last one"}]
    assert offered_text(q).startswith("you can choose"), offered_text(q)
    assert offered_text(q[:1]) == ""
