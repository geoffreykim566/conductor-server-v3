"""The turn's confidence tier (strong / moderate / research / generic), computed in
code from the trace, and the sources a research call cited. The tier no longer
reaches the client (badge disabled); the battery and logs still use it.
Rules and history: README.md (Confidence tier)."""


def _moderate_grounded_lookup_exists(trace: list[dict]) -> bool:
    """Some lookup hit at least moderate. A weak hit means "not actually relevant",
    so it counts as no real attempt. startswith: only "strong" comes back bare."""
    return any(
        c["tool"] == "lookup_concept"
        and str(c["output"].get("match_confidence", "")).startswith(("strong", "moderate"))
        for c in trace
    )


def _strong_grounded_lookup_exists(trace: list[dict]) -> bool:
    """Some lookup hit strong AND surfaced a solution with an action. A strong
    diagnosis whose solutions carry no fix is no grounding for a fix."""
    return any(
        c["tool"] == "lookup_concept"
        and c["output"].get("match_confidence") == "strong"
        and any(s.get("action") for s in c["output"].get("solutions") or [])
        for c in trace
    )


def _successful_research_exists(trace: list[dict]) -> bool:
    """Some web_research call returned findings (a timed-out or empty one doesn't count)."""
    return any(
        c["tool"] == "web_research" and not c["output"].get("error") and c["output"].get("findings")
        for c in trace
    )


def confidence_tier(
    trace: list[dict],
    walkthrough_steps: list | None,
    is_clarifying_question: bool = False,
) -> str:
    """A clarifying question claims nothing -> generic. A card -> strong. Successful
    research -> research. No tool calls -> generic. Then strong / moderate / generic
    from the lookups."""
    if is_clarifying_question:
        return "generic"
    if walkthrough_steps is not None:
        return "strong"
    if _successful_research_exists(trace):
        return "research"
    if not trace:
        return "generic"
    if _strong_grounded_lookup_exists(trace):
        return "strong"
    return "moderate" if _moderate_grounded_lookup_exists(trace) else "generic"


def collect_sources(trace: list[dict]) -> list[dict]:
    """Sources cited by this turn's non-error web_research calls, in call order,
    de-duplicated by URL."""
    sources: list[dict] = []
    seen: set[str] = set()
    for c in trace:
        if c["tool"] != "web_research" or c["output"].get("error"):
            continue
        for src in c["output"].get("sources") or []:
            url = src.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append(src)
    return sources
