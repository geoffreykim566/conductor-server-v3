"""cite_kb's executor: the cited entries resolved into lookup_concept's old result
shape, so commit-to-one, auto-queued actions, backfill and the grader work on it."""
from app.kb.data import BUCKETS_OF, PROBLEMS, SOLUTIONS
from app.kb.render import summary


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
    """The first cited entry decides the match: a problem, or a solution inside a
    bucket, comes back as that bucket with every candidate; a standalone solution
    comes back alone. `recommended` lists the cited solutions (citation order)
    whose action the pipeline may queue if the model didn't call it; each
    candidate's `bucket` is what commit-to-one checks."""
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
    elif BUCKETS_OF.get(first):
        # A cited bucket wins over the solution's strongest one.
        cited_buckets = [b for b, _ in BUCKETS_OF[first] if b in names]
        problem = cited_buckets[0] if cited_buckets else BUCKETS_OF[first][0][0]

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
    # A bucket's own name counts as a pick only when that solution is its only cause.
    out["recommended"] = [n for n in names if n in SOLUTIONS and SOLUTIONS[n].get("action")
                          and len((PROBLEMS.get(n) or {}).get("solutions") or [None]) == 1]
    if unknown:
        out["error"] = f"not KB entries: {unknown}"
    return out
