"""Commit-to-one: only one solution per problem bucket goes on the card."""
from __future__ import annotations

from app.pipeline import notes
from conftest import cite_kb, resp, route, text_block, tool_use

CRACKLING = "crackling during playback"   # CPU overload -> buffer size, sample rate mismatch -> sample rate


async def test_alternative_pick_refused(run) -> None:
    """A second candidate from the same cited bucket is refused."""
    result, _ = await run([
        resp(cite_kb([CRACKLING], "l1")),
        resp(tool_use("open_setting", {"name": "buffer size"}, "w1"), tool_use("open_setting", {"name": "sample rate"}, "w2")),
        resp(text_block("done")),
    ])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_setting"]
    assert outs[0]["attached"] and outs[1] == {"attached": False, "reason": notes.FALLBACK_REFUSAL}, outs


async def test_separate_request_joins(run) -> None:
    """A candidate from a different cited bucket is a separate request and joins the card."""
    result, _ = await run([
        resp(cite_kb([CRACKLING], "l1"), cite_kb(["vocals are out of tune"], "l2")),
        resp(tool_use("open_setting", {"name": "buffer size"}, "w1"), tool_use("open_setting", {"name": "flex pitch"}, "w2")),
        resp(text_block("done")),
    ])
    assert result.walkthrough_steps == route("buffer size") + route("flex pitch"), result.walkthrough_steps


async def test_dropdown_value_is_still_an_alternative(run) -> None:
    """Commit-to-one still refuses a same-bucket alternative that carries a value."""
    result, _ = await run([
        resp(cite_kb([CRACKLING], "l1")),
        resp(tool_use("open_setting", {"name": "sample rate"}, "w1"),
             tool_use("open_setting", {"name": "buffer size", "value": "larger"}, "w2")),
        resp(text_block("done")),
    ], messages=[{"role": "user", "content": "crackling"}])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_setting"]
    assert outs[0]["attached"] and outs[1] == {"attached": False, "reason": notes.FALLBACK_REFUSAL}, outs


async def test_one_solutions_actions_are_not_alternatives(run) -> None:
    """Two actions of one solution ("do both": buffer + low latency mode) both join
    the card; only another solution's action is an alternative."""
    result, _ = await run([
        resp(cite_kb(["high latency while monitoring"], "l1")),
        resp(tool_use("open_setting", {"name": "buffer size", "value": "smaller"}, "w1"),
             tool_use("open_setting", {"name": "low latency monitoring mode"}, "w2")),
        resp(text_block("ok")),
    ], messages=[{"role": "user", "content": "i hear myself delayed when i sing"}])
    refused = [c for c in result.trace if c["tool"] == "open_setting" and not c["output"].get("attached")]
    assert not refused, refused
    assert len(result.walkthrough_steps) > len(route("buffer size")), result.walkthrough_steps


async def test_commit_to_one_inside_one_citation(run) -> None:
    """Citing two solutions from one bucket queues the first; the second is refused as its alternative."""
    result, _ = await run([resp(cite_kb([CRACKLING, "CPU overload", "sample rate mismatch"], "l1")),
                           resp(text_block("ok"))])
    assert result.walkthrough_steps == route("buffer size"), result.walkthrough_steps
    cite = next(c["output"] for c in result.trace if c["tool"] == "cite_kb")
    assert [o["output"] for o in cite["on_card"]][1] == {"attached": False, "reason": notes.FALLBACK_REFUSAL}, cite


async def test_auto_attach_respects_commit_to_one(run) -> None:
    """A citation can't auto-queue a second candidate for a problem already on the
    card (the model cites the sibling candidate by name after picking)."""
    result, _ = await run([
        resp(cite_kb([CRACKLING], "l1")),
        resp(tool_use("open_setting", {"name": "buffer size"}, "w1")),
        resp(cite_kb(["sample rate mismatch"], "l2")),
        resp(text_block("done")),
    ])
    assert result.walkthrough_steps == route("buffer size"), result.walkthrough_steps
    second = [c for c in result.trace if c["tool"] == "cite_kb"][1]["output"]
    assert second["on_card"][0]["output"] == {"attached": False, "reason": notes.FALLBACK_REFUSAL}, second
