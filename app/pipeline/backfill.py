"""Re-showing an earlier card when the user asks to see a destination again and the
model answered from memory without calling a tool. Gate and history: README.md (Backfill)."""
import json

from app.pipeline.heuristics import is_reask


def backfill_walkthrough(messages: list[dict], text: str) -> dict | None:
    """Only on explicit re-ask language in the user's own message (heuristics
    _REASK_SIGNALS). Finds earlier successful route attaches by result shape
    (attached + destination + steps) and re-attaches the one whose name appears
    in this reply; zero or several matches are left alone, not guessed. Coverage
    of hostile phrasings lives in tests/pipeline/test_backfill.py."""
    if not is_reask(messages):
        return None

    attached_by_destination: dict[str, list] = {}
    for msg in messages:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            try:
                result = json.loads(block["content"])
            except (TypeError, ValueError):
                continue
            # on_card: an action a lookup queued itself, recorded only there.
            for r in [result] + [o.get("output") or {} for o in result.get("on_card") or []]:
                if r.get("attached") and r.get("destination") and r.get("steps"):
                    attached_by_destination[r["destination"]] = r["steps"]

    matches = [dest for dest in attached_by_destination if dest.lower() in text.lower()]
    if len(matches) != 1:
        return None
    return {"destination": matches[0], "steps": attached_by_destination[matches[0]]}
