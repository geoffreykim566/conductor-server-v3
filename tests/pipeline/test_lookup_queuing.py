"""How a lookup_concept result queues its solution's action onto the card."""
from __future__ import annotations

from app.pipeline import notes
from conftest import open_plugin, queue_setting, resp, route, single, text_block, tool_use


async def test_auto_attach_strong_single(run) -> None:
    """A strong single lookup queues its solution's action; a lookup-queued card never auto-runs."""
    lookups = [single("buffer size", {"open_setting": "buffer size"})]
    result, calls = await run([
        resp(tool_use("lookup_concept", {"problem": "buffer size"}, "l1")),
        resp(text_block("It's the I/O buffer size.")),
    ], lookups=lookups, messages=[{"role": "user", "content": "raise my buffer size"}])
    assert result.walkthrough_steps == route("buffer size"), result.walkthrough_steps
    assert result.card_from_lookup is True and result.auto_run is False, (result.card_from_lookup, result.auto_run)
    look = next(c["output"] for c in result.trace if c["tool"] == "lookup_concept")
    assert look["on_card_note"] == notes.ON_CARD_NOTE + notes.PANE_ONLY_NOTE, look  # a dropdown route
    assert len(calls) == 2, "no extra decider call on an auto-attached turn"


async def test_route_toggle_gate(run) -> None:
    """The toggle gate on a route refuses an auto-attach that would switch it away."""
    lookups = [single("input monitoring toggle", {"open_setting": "input monitoring toggle"})]
    result, _ = await run([resp(tool_use("lookup_concept", {"problem": "monitor button"}, "l1")), resp(text_block("ok"))],
                          lookups=lookups, ax_fixture={"input_monitoring_button_visible": True})
    look = next(c["output"] for c in result.trace if c["tool"] == "lookup_concept")
    assert result.walkthrough_steps is None and look["on_card_note"].startswith("This solution's action was NOT queued"), look


async def test_endorsing_a_lookup_queued_action(run) -> None:
    """A direct call for an already-queued action replaces it and isn't refused:
    agreement, not a second request, so its version (with the track) wins."""
    lookups = [single("channel eq", {"open_plugin": {"plugin": "Channel EQ"}})]
    result, _ = await run([
        resp(tool_use("lookup_concept", {"problem": "channel eq"}, "l1")),
        resp(open_plugin("Channel EQ", "a1", track="Audio 1")),
        resp(text_block("done")),
    ], lookups=lookups, messages=[{"role": "user", "content": "open channel eq"}])
    assert result.walkthrough_steps == [{"ax_open_plugin": "Channel EQ", "track": "Audio 1"}], result.walkthrough_steps
    opens = [c["output"] for c in result.trace if c["tool"] == "open_plugin"]
    assert opens[0].get("attached"), opens[0]                    # the lookup's own queue
    assert opens[-1].get("already_queued"), opens[-1]            # the model endorsing it
    # The step is model-called now (card_from_lookup clears), but the turn still
    # consulted the KB, so the card waits either way (see respond()).
    assert result.card_from_lookup is False and result.auto_run is False, (result.card_from_lookup, result.auto_run)


async def test_dropdown_value_replaces_lookup_queued_pane(run) -> None:
    """A refused dropdown value leaves the lookup's pane step queued; a valid one
    (a direction) replaces it instead of joining it."""
    lookups = [single("CPU overload", {"open_setting": "buffer size"})]
    result, _ = await run([
        resp(tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        resp(tool_use("open_setting", {"name": "buffer size", "value": "1024"}, "w1")),
        resp(tool_use("open_setting", {"name": "buffer size", "value": "larger"}, "w2")),
        resp(text_block("done")),
    ], lookups=lookups, messages=[{"role": "user", "content": "my playback keeps crackling"}])
    look = next(c["output"] for c in result.trace if c["tool"] == "lookup_concept")
    assert notes.PANE_ONLY_NOTE in look["on_card_note"], look
    sets = [c for c in result.trace if c["tool"] == "open_setting" and not c.get("auto_from")]
    assert sets[0]["output"]["attached"] is False and sets[1]["output"].get("already_queued"), sets
    assert result.walkthrough_steps == queue_setting("buffer size", "larger")["steps"], result.walkthrough_steps
