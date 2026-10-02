"""How model-called action and route tools build up the turn's one card."""
from __future__ import annotations

from app import pipeline
from app.pipeline import notes, settings
from conftest import open_plugin, queue_setting, resp, route, set_param, text_block, tool_use


async def test_actions_combine_same_response(run) -> None:
    """Two actions in one response -> one card, call order, value '80hz' parsed to '80'."""
    result, _ = await run([
        resp(open_plugin("Channel EQ", "a1"), set_param("Channel EQ", "Low Cut Frequency", "80hz", "a2")),
        resp(text_block("done")),
    ])
    assert result.walkthrough_steps == [
        {"ax_open_plugin": "Channel EQ"},
        {"ax_set_param": {"plugin": "Channel EQ", "param": "Low Cut Frequency", "value": "80"}},
    ], result.walkthrough_steps


async def test_actions_combine_across_iterations(run) -> None:
    """Actions in separate iterations concatenate; track carried on the wire step."""
    result, _ = await run([
        resp(open_plugin("Compressor", "a1", track="Vocal")),
        resp(set_param("Compressor", "Threshold", "-18 dB", "a2")),
        resp(text_block("done")),
    ])
    assert result.walkthrough_steps == [
        {"ax_open_plugin": "Compressor", "track": "Vocal"},
        {"ax_set_param": {"plugin": "Compressor", "param": "Threshold", "value": "-18"}},
    ], result.walkthrough_steps


async def test_action_past_cap_refused(run) -> None:
    """The action after MAX_ATTACHES_PER_TURN is refused at the cap."""
    calls = [open_plugin(f"Plugin {n}", f"a{n}") for n in range(1, settings.MAX_ATTACHES_PER_TURN + 2)]
    result, _ = await run([resp(*calls), resp(text_block("done"))])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_plugin"]
    assert sum(o["attached"] for o in outs) == settings.MAX_ATTACHES_PER_TURN, outs
    assert outs[-1] == {"attached": False, "reason": notes.ATTACH_CAP_REFUSAL}, outs[-1]
    assert len(result.walkthrough_steps) == settings.MAX_ATTACHES_PER_TURN


async def test_duplicate_action_refused(run) -> None:
    """An exact duplicate action is refused; it runs once."""
    result, _ = await run([resp(open_plugin("Compressor", "a1"), open_plugin("Compressor", "a2")),
                           resp(text_block("done"))])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_plugin"]
    assert outs[0]["attached"] and outs[1] == {"attached": False, "reason": notes.DUPLICATE_ATTACH_REFUSAL}, outs
    assert result.walkthrough_steps == [{"ax_open_plugin": "Compressor"}]


async def test_route_and_action_share_the_card(run) -> None:
    """A route and a plugin action share one card, in call order."""
    result, _ = await run([
        resp(tool_use("open_setting", {"name": "buffer size"}, "w1"), open_plugin("Compressor", "a1")),
        resp(text_block("done")),
    ])
    assert result.walkthrough_steps == route("buffer size") + [{"ax_open_plugin": "Compressor"}], result.walkthrough_steps
    assert result.card_from_lookup is False


async def test_unknown_route_refused(run) -> None:
    """open_setting refuses a name that isn't an approved route."""
    result, _ = await run([resp(tool_use("open_setting", {"name": "frame rate"}, "w1")), resp(text_block("done"))])
    out = next(c["output"] for c in result.trace if c["tool"] == "open_setting")
    assert out["attached"] is False and "approved route" in out["reason"], out
    assert result.walkthrough_steps is None


async def test_one_visit_per_setting(run) -> None:
    """A second call for a setting already on the card replaces it in place; one
    value per setting (both used to run, then the card and reply disagreed)."""
    result, _ = await run([
        resp(tool_use("open_setting", {"name": "flex time", "value": "Rhythmic"}, "w1"),
             tool_use("open_setting", {"name": "buffer size"}, "w2"),
             tool_use("open_setting", {"name": "flex time", "value": "Monophonic"}, "w3")),
        resp(text_block("done")),
    ], messages=[{"role": "user", "content": "flex my vocal"}])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_setting"]
    assert outs[0].get("replaced") and outs[0]["attached"] is False, outs[0]
    assert outs[2]["chooses"] == "Monophonic" and outs[2].get("replaces_earlier"), outs[2]
    want = queue_setting("flex time", "Monophonic")["steps"] + route("buffer size")
    assert result.walkthrough_steps == want, result.walkthrough_steps
    assert [ln.split(" -- ")[0] for ln in pipeline.card_descriptions(result.trace)] == ["opens buffer size", "opens flex time"]
