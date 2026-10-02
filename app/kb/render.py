"""The KB rendered as text for the decider's cached system prompt (KB_TEXT), and one
solution's summary (also what a citation returns per candidate)."""
import json

from app.kb.data import BUCKETS_OF, PROBLEM_LIST, SOLUTION_LIST, SOLUTIONS
from app.kb.paths import where_line

_FIELDS = (
    ("symptom", "Symptom"), ("cause", "Cause"), ("fix", "Fix"),
    ("zone", "Zone"), ("remove_when", "Remove when"), ("remove_move", "Remove move"),
    ("add_when", "Add when"), ("add_move", "Add move"), ("do_not", "Do not"),
    ("opens", "Opens"),
)


def _action_line(action) -> str | None:
    if not action:
        return None
    calls = action if isinstance(action, list) else [action]
    parts = []
    for call in calls:
        for tool, arg in call.items():
            parts.append(f'{tool} "{arg}"' if isinstance(arg, str) else f"{tool} {json.dumps(arg)}")
    return "Action: " + "; ".join(parts)


def summary(name: str) -> str:
    """One solution's content as labeled lines, ending in its Where line."""
    content = SOLUTIONS[name].get("content") or {}
    lines = [name] + [f"{label}: {content[key]}" for key, label in _FIELDS if content.get(key)]
    where = where_line(SOLUTIONS[name])
    if where:
        lines.append(where)
    return "\n".join(lines)


def _render() -> str:
    out = ["## Problems (symptom buckets)", ""]
    for p in PROBLEM_LIST:
        out.append(f"### {p['name']}")
        if p.get("aliases"):
            out.append("Also described as: " + "; ".join(p["aliases"]))
        if p.get("note"):
            out.append(f"Note: {p['note']}")
        out.append("Candidate causes, most common first:")
        for link in sorted(p["solutions"], key=lambda l: -(l.get("seed_weight") or 0)):
            line = f"- {link['ref']} (prior {link.get('seed_weight')})"
            if link.get("distinguisher"):
                line += f": {link['distinguisher']}"
            out.append(line)
        out.append("")
    out += ["## Solutions", ""]
    for s in SOLUTION_LIST:
        out.append(f"### {s['name']}")
        buckets = [b for b, _ in BUCKETS_OF.get(s["name"], [])]
        if buckets:
            out.append("Candidate for: " + "; ".join(buckets))
        if s.get("aliases"):
            out.append("Also described as: " + "; ".join(s["aliases"]))
        action = _action_line(s.get("action"))
        if action:
            out.append(action)
        out += summary(s["name"]).split("\n")[1:]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


KB_TEXT = _render()
