"""The card description lines the writer is given (pipeline.card_descriptions)."""
from __future__ import annotations

from app import pipeline


def test_card_descriptions() -> None:
    """The writer's card description covers every queued step and nothing else."""
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


def test_card_descriptions_dropdowns() -> None:
    """The writer is told whether a dropdown route picks a value or stops at the pane."""
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
