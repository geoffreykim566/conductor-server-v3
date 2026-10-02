"""Commit-to-one: only one candidate per problem bucket goes on the card."""
from __future__ import annotations

from app.pipeline import notes
from conftest import bucket, resp, route, single, text_block, tool_use

CRACKLING = [bucket("crackling", {"CPU overload": {"open_setting": "buffer size"},
                                  "sample rate mismatch": {"open_setting": "sample rate"}})]


async def test_alternative_pick_refused_separate_request_joins(run) -> None:
    """A second candidate from the same bucket is refused; a separate request joins the card."""
    result, _ = await run([
        resp(tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        resp(tool_use("open_setting", {"name": "buffer size"}, "w1"), tool_use("open_setting", {"name": "sample rate"}, "w2")),
        resp(text_block("done")),
    ], lookups=CRACKLING)
    outs = [c["output"] for c in result.trace if c["tool"] == "open_setting"]
    assert outs[0]["attached"] and outs[1] == {"attached": False, "reason": notes.FALLBACK_REFUSAL}, outs
    two = [bucket("crackling", {"CPU overload": {"open_setting": "buffer size"}}),
           bucket("freezing", {"track freeze (technique)": {"open_setting": "freeze track"}})]
    result, _ = await run([
        resp(tool_use("lookup_concept", {"problem": "crackling"}, "l1"), tool_use("lookup_concept", {"problem": "freeze"}, "l2")),
        resp(tool_use("open_setting", {"name": "buffer size"}, "w1"), tool_use("open_setting", {"name": "freeze track"}, "w2")),
        resp(text_block("done")),
    ], lookups=two)
    assert result.walkthrough_steps == route("buffer size") + route("freeze track"), result.walkthrough_steps


async def test_dropdown_value_is_still_an_alternative(run) -> None:
    """Commit-to-one still refuses a same-bucket alternative that carries a value."""
    result, _ = await run([
        resp(tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        resp(tool_use("open_setting", {"name": "sample rate"}, "w1"),
             tool_use("open_setting", {"name": "buffer size", "value": "larger"}, "w2")),
        resp(text_block("done")),
    ], lookups=CRACKLING, messages=[{"role": "user", "content": "crackling"}])
    outs = [c["output"] for c in result.trace if c["tool"] == "open_setting"]
    assert outs[0]["attached"] and outs[1] == {"attached": False, "reason": notes.FALLBACK_REFUSAL}, outs


async def test_auto_attach_respects_commit_to_one(run) -> None:
    """A lookup can't auto-queue a second candidate for a problem already on the
    card (the model looks the sibling candidate up by name after picking)."""
    lookups = CRACKLING + [single("sample rate mismatch", {"open_setting": "sample rate"})]
    result, _ = await run([
        resp(tool_use("lookup_concept", {"problem": "crackling"}, "l1")),
        resp(tool_use("open_setting", {"name": "buffer size"}, "w1")),
        resp(tool_use("lookup_concept", {"problem": "sample rate mismatch"}, "l2")),
        resp(text_block("done")),
    ], lookups=lookups)
    assert result.walkthrough_steps == route("buffer size"), result.walkthrough_steps
    second = [c for c in result.trace if c["tool"] == "lookup_concept"][1]["output"]
    assert second["on_card"][0]["output"] == {"attached": False, "reason": notes.FALLBACK_REFUSAL}, second
