"""Pulls the confidence line, findings text and cited sources out of a research response."""
import re

CONFIDENCE_TO_TIER: dict[str, str] = {
    "confirmed": "confirmed-research",
    "commonly_believed": "community-research",
    "genre_inferred": "genre-inference",
}


def parse_confidence(text: str) -> str:
    m = re.search(
        r"^CONFIDENCE:\s*(confirmed|commonly_believed|genre_inferred)",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    return m.group(1).lower() if m else "commonly_believed"


def strip_meta(text: str) -> str:
    text = re.sub(r"^CONFIDENCE:.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^CHARACTERISTICS:.*$", "", text, flags=re.MULTILINE)
    return text.strip()


def extract_sources(all_content: list, confidence: str) -> list[dict]:
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


def extract_sources_from_text(text: str, confidence: str) -> list[dict]:
    """Fallback: bare URLs from response text when structured blocks are absent."""
    sources: list[dict] = []
    seen: set[str] = set()
    for url in re.findall(r"https?://[^\s\)\]\>\"']+", text):
        url = url.rstrip(".,;:")
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": url, "url": url, "type": confidence})
    return sources
