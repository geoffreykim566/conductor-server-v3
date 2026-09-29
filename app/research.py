"""Web research tool: search -> findings, ported from v1's routing/research.py.

web_research() is the executor for tools.WEB_RESEARCH_SCHEMA, dispatched by
pipeline.py's _EXECUTORS exactly like lookup_concept/get_walkthrough -- an
app-code executor that does real work (here, its own nested Sonnet +
web_search call) and returns a JSON tool_result. Unlike v1, there's no
separate router stage deciding *whether* to call this: the decider tool-loop
makes that call itself, guided only by tools.WEB_RESEARCH_SCHEMA's
description (v3-log.md 2026-09-04 scoping note: prompt-only trigger,
matching how get_walkthrough already works, not a deterministic gate --
firing a real web-search call on every weak/no-hit lookup_concept result
would be the costlier, more eager choice, in a project that hasn't yet
measured per-turn cost since the writer-split).

The coarse confidence badge IS wired end-to-end: a successful call earns
pipeline.py's _confidence_tier "research" label (trusted like "strong"), and
`sources` flows through Result.sources into api.py's "sources" response field
and client-v3's badge/source-chip UI. Stale note this docstring used to carry
here ("NOT wired... mid-flight elsewhere") was left over from before that
landed -- corrected 2026-09-04. What's still NOT wired: the fine-grained
per-call `source_tier` this module computes (confirmed-research /
community-research / genre-inference, see _CONFIDENCE_TO_TIER) never reaches
_confidence_tier -- every successful call gets the same "research" badge
regardless of which of those three it actually was. Also latent as of the
same date: _parse_confidence's regex requires a line starting with literal
"CONFIDENCE:" and misses the model's own bold-markdown output
("**CONFIDENCE:**" was observed live), silently defaulting to
commonly_believed -- harmless while the fine-grained tier isn't surfaced
anywhere, will matter once it is.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time as _time

from anthropic import AsyncAnthropic

from app.config import CENTRAL_ANTHROPIC_KEY, RESEARCH_CALL_TIMEOUT_S, RESEARCH_MODEL

log = logging.getLogger(__name__)

_client = AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY)

# Extraction-only prompt, same split v1 used: this call's job is to find and
# report facts, not to frame them for the end user -- the decider (with its
# own SYSTEM_PROMPT) still owns turning findings into a final answer, same as
# it already does with a lookup_concept/get_walkthrough result.
_RESEARCH_SYSTEM = (
    "You are a music production researcher. Search the web for accurate information.\n"
    "Report what you find: specific techniques, gear, signal chains, processing details.\n"
    "Name exactly what sources describe -- include third-party plugins and gear as mentioned.\n\n"
    "After your main answer, output exactly two lines:\n"
    "CONFIDENCE: confirmed | commonly_believed | genre_inferred\n"
    "CHARACTERISTICS: [2-5 word sonic/technical description, for future KB matching]\n\n"
    "Confidence guide:\n"
    "  confirmed -- direct evidence (interviews, official sources, documented gear lists)\n"
    "  commonly_believed -- multiple reliable sources agree, not directly verified\n"
    "  genre_inferred -- only general genre/technique knowledge found, nothing specific"
)

_CONFIDENCE_TO_TIER: dict[str, str] = {
    "confirmed": "confirmed-research",
    "commonly_believed": "community-research",
    "genre_inferred": "genre-inference",
}

# Bounds on what this tool's result carries forward. Unlike ax_state/screenshots
# (system-prompt-only, never touch `msgs` at all -- see pipeline.py's respond()),
# a tool_result has to stay in conversation history to pair with its tool_use
# (Anthropic API requirement -- there's no screenshot-style total exclusion
# available here, omitting it would 400 every later request in the same
# conversation, not just quietly grow it). So this has to actually be capped,
# not excluded. Found live 2026-09-04: an uncapped 22-source, long-findings
# result (query: rage-rap production techniques) pushed a later turn's replayed
# history past api.py's 20,000-char per-block cap, 422ing every subsequent
# request in that conversation. Applied once here rather than per-caller so the
# decider/writer and the persisted history version are always the same content.
_MAX_FINDINGS_CHARS = 6_000
_MAX_SOURCES = 10


def _parse_confidence(text: str) -> str:
    m = re.search(
        r"^CONFIDENCE:\s*(confirmed|commonly_believed|genre_inferred)",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    return m.group(1).lower() if m else "commonly_believed"


def _strip_meta(text: str) -> str:
    text = re.sub(r"^CONFIDENCE:.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^CHARACTERISTICS:.*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_sources(all_content: list, confidence: str) -> list[dict]:
    """Structured citations from web_search_tool_result blocks (preferred)."""
    sources: list[dict] = []
    seen: set[str] = set()
    for block in all_content:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        for item in getattr(block, "content", []) or []:
            url = getattr(item, "url", None)
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({
                "title": getattr(item, "title", None) or url,
                "url": url,
                "type": confidence,
            })
    return sources


def _extract_sources_from_text(text: str, confidence: str) -> list[dict]:
    """Fallback: bare URLs from response text when structured blocks are absent."""
    sources: list[dict] = []
    seen: set[str] = set()
    for url in re.findall(r"https?://[^\s\)\]\>\"']+", text):
        url = url.rstrip(".,;:")
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": url, "url": url, "type": confidence})
    return sources


def _usage_dict(total_in: int, total_out: int) -> dict:
    # Same shape as pipeline.py's _usage_dict, minus cache fields -- this call
    # never sends a cache_control breakpoint (its system prompt is tiny and
    # not reused loop-to-loop the way the decider's is).
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


async def web_research(query: str) -> dict:
    """Executor for tools.WEB_RESEARCH_SCHEMA. One Sonnet + web_search_20260209
    call, up to 2 pause_turn continuations (same budget as v1's research.py).

    Returns a dict safe to json.dumps into a tool_result. The "_usage" key is
    popped back out by pipeline.py's respond() before that happens (see the
    tool-dispatch loop) so it's counted toward the turn's real token spend
    without ever reaching the model as visible tool-result content.
    """
    tools_param = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
    api_messages: list[dict] = [{"role": "user", "content": query}]
    all_content: list = []
    response_text = ""
    total_in = total_out = 0
    max_continuations = 2
    continuations = 0
    # Whole-call latency (all continuations included), not just one iteration --
    # added 2026-09-04 after RESEARCH_CALL_TIMEOUT_S got bumped 90s->150s on
    # observed timeouts with no actual success-path number to size the new
    # value against. Logged unconditionally (success and failure) so real
    # per-call latency is visible without re-instrumenting later.
    call_start = _time.monotonic()

    while True:
        t0 = _time.monotonic()
        try:
            response = await asyncio.wait_for(
                _client.messages.create(
                    model=RESEARCH_MODEL,
                    max_tokens=2048,
                    system=_RESEARCH_SYSTEM,
                    messages=api_messages,
                    tools=tools_param,
                ),
                timeout=RESEARCH_CALL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.warning(
                "[web_research_timing] TIMEOUT after %.1fs total (%d continuation(s), query=%r)",
                _time.monotonic() - call_start, continuations, query,
            )
            return {
                "error": "web research timed out",
                "_usage": _usage_dict(total_in, total_out),
            }
        except Exception:
            log.exception(
                "[web_research_timing] FAILED after %.1fs total (query=%r)",
                _time.monotonic() - call_start, query,
            )
            return {
                "error": "web research failed",
                "_usage": _usage_dict(total_in, total_out),
            }

        all_content.extend(response.content)
        total_in += response.usage.input_tokens
        total_out += response.usage.output_tokens
        for block in response.content:
            if getattr(block, "type", None) == "text":
                response_text += block.text

        if response.stop_reason == "pause_turn" and continuations < max_continuations:
            log.warning(
                "[web_research_timing] continuation %d after %.1fs this leg (query=%r)",
                continuations + 1, _time.monotonic() - t0, query,
            )
            api_messages.append({"role": "assistant", "content": response.content})
            continuations += 1
            continue
        break

    confidence = _parse_confidence(response_text)
    findings = _strip_meta(response_text)
    source_tier = _CONFIDENCE_TO_TIER.get(confidence, "community-research")

    sources = _extract_sources(all_content, confidence)
    if not sources and findings:
        sources = _extract_sources_from_text(findings, confidence)

    if len(findings) > _MAX_FINDINGS_CHARS:
        findings = findings[:_MAX_FINDINGS_CHARS] + "\n\n... (truncated)"
    sources = sources[:_MAX_SOURCES]

    log.warning(
        "[web_research_timing] SUCCESS after %.1fs total (%d continuation(s), %d sources, query=%r)",
        _time.monotonic() - call_start, continuations, len(sources), query,
    )

    result: dict = {
        "findings": findings,
        "source_tier": source_tier,
        "sources": sources,
        "_usage": _usage_dict(total_in, total_out),
    }
    if not findings:
        result["note"] = "no findings returned -- tell the user you couldn't find reliable information"
    return result
