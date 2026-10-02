"""Argument validation in the open_plugin / set_param executors."""
from __future__ import annotations

import pytest

from app import tools


def test_open_plugin_validation() -> None:
    assert tools.queue_open_plugin({"plugin": "  "})["attached"] is False
    assert tools.queue_open_plugin({"plugin": "Channel EQ", "new_instance": True})["steps"] == [
        {"ax_open_plugin": "Channel EQ", "new": True}]
    assert tools.queue_open_plugin({"plugin": "Channel EQ", "new_instance": False})["steps"] == [
        {"ax_open_plugin": "Channel EQ"}]


@pytest.mark.parametrize("raw, want", [
    ("80", "80"), ("80hz", "80"), ("-18 dB", "-18"), ("2.5", "2.5"), ("On", "on"), ("disabled", "off"),
    (-18, "-18"), (4.0, "4"), (True, "on"),
])
def test_set_param_parses_value(raw, want) -> None:
    got = tools.queue_set_param({"plugin": "P", "param": "X", "value": raw})
    assert got["attached"] and got["steps"][0]["ax_set_param"]["value"] == want, (raw, got)


@pytest.mark.parametrize("bad", ["4:1", "loud", "", None])
def test_set_param_refuses_non_number(bad) -> None:
    got = tools.queue_set_param({"plugin": "P", "param": "X", "value": bad})
    assert got["attached"] is False and "plain number" in got["reason"], (bad, got)
