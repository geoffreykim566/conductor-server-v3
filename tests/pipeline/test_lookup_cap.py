"""The per-turn lookup_concept cap (settings.LOOKUP_ATTEMPT_LIMIT / COUNT_ALL_LOOKUPS)."""
from __future__ import annotations

from unittest.mock import patch

from app.pipeline import settings
from conftest import resp, text_block, tool_use


def _lookups(n: int) -> list:
    return [resp(tool_use("lookup_concept", {"problem": f"q{i}"}, f"l{i}")) for i in range(n)]


def _lookup_outputs(result) -> list[dict]:
    return [c["output"] for c in result.trace if c["tool"] == "lookup_concept"]


async def test_cap_counts_every_lookup(run) -> None:
    """Default COUNT_ALL_LOOKUPS: every lookup counts, whatever it returned, so
    distinct moderate wrong buckets can't run an off-KB turn past the cap."""
    limit = settings.LOOKUP_ATTEMPT_LIMIT
    distinct = [{"match": "problem", "problem": f"bucket {i}", "match_confidence": "strong"} for i in range(limit + 1)]
    with patch.object(settings, "MAX_ITERATIONS", limit + 3):
        result, _ = await run(_lookups(limit + 1) + [resp(text_block("done"))], lookups=distinct)
    outs = _lookup_outputs(result)
    assert "error" in outs[-1] and all("error" not in o for o in outs[:limit]), outs


async def test_unproductive_only_variant(run) -> None:
    """COUNT_ALL_LOOKUPS=False: distinct real buckets don't count, no-match ones do."""
    limit = settings.LOOKUP_ATTEMPT_LIMIT
    distinct = [{"match": "problem", "problem": f"bucket {i}", "match_confidence": "strong"} for i in range(limit + 1)]
    with patch.object(settings, "COUNT_ALL_LOOKUPS", False), patch.object(settings, "MAX_ITERATIONS", limit + 3):
        result, _ = await run(_lookups(limit + 1) + [resp(text_block("done"))], lookups=distinct)
        assert all("error" not in o for o in _lookup_outputs(result)), _lookup_outputs(result)
        result, _ = await run(_lookups(limit + 1) + [resp(text_block("done"))], lookups=[{"match": "none"}])
        outs = _lookup_outputs(result)
        assert "error" in outs[-1] and all("error" not in o for o in outs[:-1]), outs
