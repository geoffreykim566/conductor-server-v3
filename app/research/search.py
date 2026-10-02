"""web_research: one nested Sonnet + web_search call that returns findings and sources.
Executor for tools.WEB_RESEARCH_SCHEMA. See README.md for what is and isn't wired."""
from __future__ import annotations

import asyncio
import logging
import time as _time

from anthropic import AsyncAnthropic

from app.core.config import CENTRAL_ANTHROPIC_KEY, RESEARCH_CALL_TIMEOUT_S, RESEARCH_MODEL
from app.research.parse import (
    CONFIDENCE_TO_TIER,
    extract_sources,
    extract_sources_from_text,
    parse_confidence,
    strip_meta,
)

log = logging.getLogger(__name__)

_client = AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY)

# Extraction only: the decider, not this call, turns findings into the answer.
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


# The tool_result stays in round-tripped history, so it must be capped (an uncapped
# result 422'd every later request in that conversation).
_MAX_FINDINGS_CHARS = 6_000
_MAX_SOURCES = 10


def _usage_dict(total_in: int, total_out: int) -> dict:
    # pipeline's usage shape; this call never sets a cache breakpoint.
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


async def web_research(query: str) -> dict:
    """Up to 2 pause_turn continuations. The returned "_usage" key is popped by
    pipeline/dispatch.py so the spend is counted but never shown to the model."""
    tools_param = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
    api_messages: list[dict] = [{"role": "user", "content": query}]
    all_content: list = []
    response_text = ""
    total_in = total_out = 0
    max_continuations = 2
    continuations = 0
    # Whole-call latency, logged on success and failure (sizes RESEARCH_CALL_TIMEOUT_S).
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

    confidence = parse_confidence(response_text)
    findings = strip_meta(response_text)
    source_tier = CONFIDENCE_TO_TIER.get(confidence, "community-research")

    sources = extract_sources(all_content, confidence)
    if not sources and findings:
        sources = extract_sources_from_text(findings, confidence)

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
