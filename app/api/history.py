"""Shaping the round-tripped history: trimming it, and spotting a parked research turn."""
from app.core.config import MAX_HISTORY_MESSAGES


def pending_research_query(history: list[dict] | None) -> str | None:
    """The web_research query a parked transcript is waiting on, or None if
    `history` isn't shaped like one (must end in an assistant message whose
    tool_use blocks include web_research, with no results after it)."""
    if not history:
        return None
    last = history[-1]
    if last.get("role") != "assistant" or not isinstance(last.get("content"), list):
        return None
    for b in last["content"]:
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "web_research":
            q = (b.get("input") or {}).get("query")
            return q if isinstance(q, str) else ""
    return None


def trim_history(messages: list[dict]) -> list[dict]:
    """Keep the newest MAX_HISTORY_MESSAGES, snapped forward to the next real
    turn boundary (a plain-string user message) so a trim never starts on a
    dangling tool_result whose tool_use got cut — that shape is rejected by
    the Anthropic API outright. If no such boundary exists in the window
    (a pathologically long single tool-loop), don't truncate at all rather
    than risk sending a broken message list.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    window = messages[-MAX_HISTORY_MESSAGES:]
    for i, msg in enumerate(window):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return window[i:]
    return messages
