"""The tool_choice forced on a turn's first decider call."""
from __future__ import annotations

from unittest.mock import patch

from app.pipeline import settings
from conftest import open_plugin, resp, text_block, tool_use


async def test_first_call_is_any(run) -> None:
    """First call is tool_choice any; later calls are unforced."""
    _, calls = await run([resp(open_plugin("Compressor", "a1")), resp(text_block("done"))])
    assert calls[0]["tool_choice"] == {"type": "any"}, calls[0].get("tool_choice")
    assert "tool_choice" not in calls[1], calls[1].get("tool_choice")


async def test_force_first_lookup_flag(run) -> None:
    """FORCE_FIRST_LOOKUP restores the forced lookup_concept first call."""
    with patch.object(settings, "FORCE_FIRST_LOOKUP", True):
        _, calls = await run([resp(tool_use("lookup_concept", {"problem": "x"}, "l1")), resp(text_block("done"))])
    assert calls[0]["tool_choice"] == {"type": "tool", "name": "lookup_concept"}, calls[0].get("tool_choice")
