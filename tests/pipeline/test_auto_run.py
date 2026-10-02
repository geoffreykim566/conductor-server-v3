"""When a turn's card may run without the user pressing Run (Result.auto_run)."""
from __future__ import annotations

import pytest

from conftest import bucket, open_plugin, resp, text_block, tool_use


async def test_a_lookup_turn_never_auto_runs(run) -> None:
    """A card the KB led to waits for Run; the same action asked for directly auto-runs."""
    lookups = [bucket("track deselects", {"bypass control surfaces": {"open_setting": "bypass control surfaces"}})]
    result, _ = await run([
        resp(tool_use("lookup_concept", {"problem": "track deselects on plugin click"}, "l1")),
        resp(tool_use("open_setting", {"name": "bypass control surfaces"}, "w1")),
        resp(text_block("try disabling control surfaces")),
    ], lookups=lookups, messages=[{"role": "user", "content": "logic randomly deselects my track when i click a knob"}])
    assert result.walkthrough_steps and result.auto_run is False, (result.walkthrough_steps, result.auto_run)
    result, _ = await run([resp(tool_use("open_setting", {"name": "bypass control surfaces"}, "w1")), resp(text_block("ok"))],
                          messages=[{"role": "user", "content": "turn off control surfaces"}])
    assert result.auto_run is True, result.auto_run


async def test_wait_for_run_route_never_auto_runs(run) -> None:
    """A wait_for_run route (targets the user's selection) waits for Run even on a direct command."""
    result, _ = await run([resp(tool_use("open_setting", {"name": "follow tempo", "value": "Off"}, "w1")),
                           resp(text_block("ok"))],
                          messages=[{"role": "user", "content": "set smart tempo off on this region"}])
    assert result.walkthrough_steps and result.auto_run is False, (result.walkthrough_steps, result.auto_run)


@pytest.mark.parametrize("message, want", [
    ("put a compressor on this", True),
    ("my vocal sounds muddy, fix it", True),
    ("can you add a compressor to this track", True),
    ("how do i add a compressor", False),
    ("explain what a compressor does and put one on this", False),
    ("channel eq?", False),
    ("put a compressor on every track", False),
])
async def test_auto_run_follows_the_message(run, message, want) -> None:
    """auto_run follows the message: instruction vs question vs bulk."""
    result, _ = await run([resp(open_plugin("Compressor", "a1")), resp(text_block("done"))],
                          messages=[{"role": "user", "content": message}])
    assert result.auto_run is want, (message, result.auto_run)


async def test_auto_run_needs_a_card(run) -> None:
    result, _ = await run([resp(text_block("hi there"))], messages=[{"role": "user", "content": "put a compressor on this"}])
    assert result.walkthrough_steps is None and result.auto_run is False
