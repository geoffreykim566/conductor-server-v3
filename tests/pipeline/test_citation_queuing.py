"""How a cite_kb result's recommended solutions queue their actions onto the card."""
from __future__ import annotations

from app import kb
from app.pipeline import notes
from conftest import cite_kb, open_plugin, queue_setting, resp, route, text_block, tool_use


async def test_cited_solution_queues_its_action(run) -> None:
    """A cited solution queues its action with no extra decider call; a citation-queued card never auto-runs."""
    result, calls = await run([
        resp(cite_kb(["buffer size"], "l1")),
        resp(text_block("It's the I/O buffer size.")),
    ], messages=[{"role": "user", "content": "raise my buffer size"}])
    assert result.walkthrough_steps == route("buffer size"), result.walkthrough_steps
    assert result.card_from_lookup is True and result.auto_run is False, (result.card_from_lookup, result.auto_run)
    cite = next(c["output"] for c in result.trace if c["tool"] == "cite_kb")
    assert cite["on_card_note"] == notes.ON_CARD_NOTE + notes.PANE_ONLY_NOTE, cite  # a dropdown route
    assert len(calls) == 2, "no extra decider call on an auto-attached turn"


async def test_cited_bucket_alone_queues_nothing(run) -> None:
    """A bucket with several causes, cited with no solution picked, recommends and queues nothing."""
    assert kb.cite(["crackling during playback"])["recommended"] == []
    result, _ = await run([resp(cite_kb(["crackling during playback"], "l1")),
                           resp(tool_use("ask_clarifying_question", {"question": "When does it crackle?"}, "q1"))])
    assert result.walkthrough_steps is None, result.walkthrough_steps


async def test_single_cause_bucket_counts_as_its_pick(run) -> None:
    """A bucket whose only cause shares its name queues that cause's action."""
    name = "MIDI keyboard not triggering notes in Logic"
    result, _ = await run([resp(cite_kb([name], "l1")), resp(text_block("ok"))])
    assert result.walkthrough_steps == route("bypass control surfaces"), result.walkthrough_steps


async def test_recommended_actions_queue_in_citation_order(run) -> None:
    """Several cited solutions from different buckets all queue, in citation order."""
    result, _ = await run([resp(cite_kb(["channel eq", "buffer size"], "l1")), resp(text_block("ok"))])
    assert result.walkthrough_steps == [{"ax_open_plugin": "Channel EQ"}] + route("buffer size"), result.walkthrough_steps


async def test_route_toggle_gate(run) -> None:
    """The toggle gate on a route refuses an auto-attach that would switch it away."""
    result, _ = await run([resp(cite_kb(["input monitoring toggle"], "l1")), resp(text_block("ok"))],
                          ax_fixture={"input_monitoring_button_visible": True})
    cite = next(c["output"] for c in result.trace if c["tool"] == "cite_kb")
    assert result.walkthrough_steps is None and cite["on_card_note"].startswith("This solution's action was NOT queued"), cite


async def test_endorsing_a_citation_queued_action(run) -> None:
    """A direct call for an already-queued action replaces it and isn't refused:
    agreement, not a second request, so its version (with the track) wins."""
    result, _ = await run([
        resp(cite_kb(["channel eq"], "l1")),
        resp(open_plugin("Channel EQ", "a1", track="Audio 1")),
        resp(text_block("done")),
    ], messages=[{"role": "user", "content": "open channel eq"}])
    assert result.walkthrough_steps == [{"ax_open_plugin": "Channel EQ", "track": "Audio 1"}], result.walkthrough_steps
    opens = [c["output"] for c in result.trace if c["tool"] == "open_plugin"]
    assert opens[0].get("attached"), opens[0]                    # the citation's own queue
    assert opens[-1].get("already_queued"), opens[-1]            # the model endorsing it
    # The step is model-called now (card_from_lookup clears), but the turn still
    # cited the KB, so the card waits either way (see respond()).
    assert result.card_from_lookup is False and result.auto_run is False, (result.card_from_lookup, result.auto_run)


async def test_dropdown_value_replaces_citation_queued_pane(run) -> None:
    """A refused dropdown value leaves the citation's pane step queued; a valid one
    (a direction) replaces it instead of joining it."""
    result, _ = await run([
        resp(cite_kb(["CPU overload"], "l1")),
        resp(tool_use("open_setting", {"name": "buffer size", "value": "1024"}, "w1")),
        resp(tool_use("open_setting", {"name": "buffer size", "value": "larger"}, "w2")),
        resp(text_block("done")),
    ], messages=[{"role": "user", "content": "my playback keeps crackling"}])
    cite = next(c["output"] for c in result.trace if c["tool"] == "cite_kb")
    assert notes.PANE_ONLY_NOTE in cite["on_card_note"], cite
    sets = [c for c in result.trace if c["tool"] == "open_setting" and not c.get("auto_from")]
    assert sets[0]["output"]["attached"] is False and sets[1]["output"].get("already_queued"), sets
    assert result.walkthrough_steps == queue_setting("buffer size", "larger")["steps"], result.walkthrough_steps
