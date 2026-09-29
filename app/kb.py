"""The knowledge base, in the prompt (2026-09-28, v043).

The whole KB (34 problem buckets, 104 solutions, ~16-18k tokens) is rendered
once, at import, into the decider's cached system prompt. The model reads all
of it and cites the entries it relied on with cite_kb -- nothing is retrieved.
Replaces lookup_concept's embedding search (Voyage + pgvector): each lookup
cost a full extra decider call (~2.5 s), off-KB questions burned up to 4 of
them, and correct and wrong matches overlapped in distance, so no confidence
cutoff could separate them (reports/Conductor retrieval architecture options.md).

seed/problems.json stays the source file for now; the model only ever sees the
rendered text. cite_kb's result keeps lookup_concept's shape (match / problem /
solutions with their actions) so commit-to-one, the auto-queued action, backfill
and the grader work on it unchanged.
"""
import json
from pathlib import Path

_SEED = json.loads((Path(__file__).parent.parent / "seed" / "problems.json").read_text())

SOLUTIONS: dict[str, dict] = {s["name"]: s for s in _SEED["solutions"]}
PROBLEMS: dict[str, dict] = {p["name"]: p for p in _SEED["problems"]}
# Every citable name. 15 buckets share their name with a solution (mostly
# one-cause buckets, e.g. "plosives on vocals"): citing one resolves as the
# bucket, and counts as a pick only when that solution is its only cause.
ENTRY_NAMES: list[str] = sorted(set(PROBLEMS) | set(SOLUTIONS))

# solution name -> [(problem name, seed_weight)], strongest bucket first
_BUCKETS_OF: dict[str, list[tuple[str, int]]] = {}
for _p in _SEED["problems"]:
    for _link in _p["solutions"]:
        _BUCKETS_OF.setdefault(_link["ref"], []).append((_p["name"], _link.get("seed_weight") or 0))
for _v in _BUCKETS_OF.values():
    _v.sort(key=lambda b: -b[1])

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
    """One solution's content as labeled lines -- what lookup_concept returned
    as a candidate's summary, and what the prompt shows under its heading."""
    content = SOLUTIONS[name].get("content") or {}
    lines = [name] + [f"{label}: {content[key]}" for key, label in _FIELDS if content.get(key)]
    return "\n".join(lines)


def _render() -> str:
    out = ["## Problems (symptom buckets)", ""]
    for p in _SEED["problems"]:
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
    for s in _SEED["solutions"]:
        out.append(f"### {s['name']}")
        buckets = [b for b, _ in _BUCKETS_OF.get(s["name"], [])]
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


def _candidate(name: str, link: dict | None = None, bucket: str | None = None) -> dict:
    return {
        "name": name,
        "bucket": bucket,
        "seed_weight": (link or {}).get("seed_weight"),
        "distinguisher": (link or {}).get("distinguisher"),
        "summary": summary(name),
        "action": SOLUTIONS[name].get("action"),
    }


def cite(entries: list[str]) -> dict:
    """What cite_kb returns: the first cited entry resolved the way
    lookup_concept resolved its top hit -- a problem, or a solution inside a
    bucket, comes back as that bucket with every candidate; a standalone
    solution comes back alone. `recommended` lists the cited solutions (in
    citation order) whose action the pipeline may queue if the model didn't
    call it itself; each candidate's `bucket` is what commit-to-one checks."""
    names = list(dict.fromkeys(e for e in entries if e in PROBLEMS or e in SOLUTIONS))
    unknown = [e for e in entries if e not in PROBLEMS and e not in SOLUTIONS]
    if not names:
        out = {"match": "none", "cited": []}
        if unknown:
            out["error"] = f"not KB entries: {unknown}"
        return out

    first = names[0]
    problem = None
    if first in PROBLEMS:
        problem = first
    elif _BUCKETS_OF.get(first):
        # A cited bucket wins over the solution's strongest bucket, so "song
        # sounds slowed" + "sample rate mismatch" resolves to that bucket.
        cited_buckets = [b for b, _ in _BUCKETS_OF[first] if b in names]
        problem = cited_buckets[0] if cited_buckets else _BUCKETS_OF[first][0][0]

    if problem is not None:
        p = PROBLEMS[problem]
        solutions = [_candidate(l["ref"], l, problem)
                     for l in sorted(p["solutions"], key=lambda l: -(l.get("seed_weight") or 0))]
        out = {"match": "problem", "problem": problem, "match_confidence": "strong",
               "solutions": solutions, "note": p.get("note")}
    else:
        solutions = [_candidate(first)]
        out = {"match": "single", "problem": first, "match_confidence": "strong",
               "solutions": solutions, "note": None}
    listed = {s["name"] for s in solutions}
    out["solutions"] += [_candidate(n) for n in names if n in SOLUTIONS and n not in listed]
    out["cited"] = names
    out["recommended"] = [n for n in names if n in SOLUTIONS and SOLUTIONS[n].get("action")
                          and len((PROBLEMS.get(n) or {}).get("solutions") or [None]) == 1]
    if unknown:
        out["error"] = f"not KB entries: {unknown}"
    return out
