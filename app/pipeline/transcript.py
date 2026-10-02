"""Reading the message list: where this turn starts, and a parked turn's pending tool calls."""
from types import SimpleNamespace


def last_user_text_index(msgs: list[dict]) -> int | None:
    """Index of the newest plain-string user message -- the turn boundary
    (the same boundary api/history.py's trim snaps to)."""
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return i
    return None


def pending_tool_uses(msgs: list[dict]) -> list:
    """tool_use blocks of a parked transcript's final assistant message, as
    objects with the .name/.input/.id the dispatch loop reads off SDK blocks.
    Empty unless the transcript really ends that way (no results after it)."""
    if not msgs or msgs[-1].get("role") != "assistant" or not isinstance(msgs[-1].get("content"), list):
        return []
    return [
        SimpleNamespace(type="tool_use", name=b.get("name"), input=b.get("input") or {}, id=b.get("id"))
        for b in msgs[-1]["content"]
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
