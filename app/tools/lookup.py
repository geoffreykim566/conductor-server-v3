"""lookup_concept: vector search over problems and solutions together. A solution hit
resolves to its problem bucket. Ranking rules and their history: README.md."""
import asyncio

from app.core import db
from app.kb.embed import embed

# Advisory bands shown to the model, never a hard filter (see README).
_STRONG_MATCH = 0.40
_MODERATE_MATCH = 0.60


def confidence_label(distance: float) -> str:
    if distance <= _STRONG_MATCH:
        return "strong"
    if distance <= _MODERATE_MATCH:
        return "moderate — treat with more skepticism, consider asking or trying a different phrasing"
    return "weak — likely not actually relevant; don't treat this as a real match"


def _confidence_short(distance: float) -> str:
    """Bare strong/moderate/weak for query_log (confidence_label is model-facing prose)."""
    if distance <= _STRONG_MATCH:
        return "strong"
    if distance <= _MODERATE_MATCH:
        return "moderate"
    return "weak"


async def _embed_with_retry(texts: list[str], input_type: str) -> list[list[float]]:
    """Backoff on Voyage errors so one transient 429 can't kill a battery run."""
    for attempt in range(6):
        try:
            return await embed(texts, input_type=input_type)
        except Exception as exc:
            if attempt == 5:
                raise
            wait = 15 * (attempt + 1)
            print(f"  [retry] embed() failed ({exc}); retrying in {wait}s")
            await asyncio.sleep(wait)


def _content_summary(name: str, content: dict) -> str:
    lines = [name]
    for key, label in (
        ("symptom", "Symptom"), ("cause", "Cause"), ("fix", "Fix"),
        ("zone", "Zone"), ("remove_when", "Remove when"), ("remove_move", "Remove move"),
        ("add_when", "Add when"), ("add_move", "Add move"), ("do_not", "Do not"),
        ("opens", "Opens"),
    ):
        if content.get(key):
            lines.append(f"{label}: {content[key]}")
    return "\n".join(lines)


async def lookup_concept(problem: str, _vec: list[float] | None = None) -> dict:
    # _vec: a caller that already embedded the query skips a second Voyage call.
    vec = _vec if _vec is not None else (await _embed_with_retry([problem], input_type="query"))[0]
    rows = await db.pool().fetch(
        """
        select * from (
            select 'problem'::text as kind, id, name, null::jsonb as content,
                   null::jsonb as action, note,
                   (embedding <=> $1) as distance
            from problems
            union all
            select 'solution'::text as kind, id, name, content,
                   action, null::text as note,
                   (embedding <=> $1) as distance
            from solutions
        ) combined
        order by distance
        limit 5
        """,
        vec,
    )
    top_results = [
        {"kind": r["kind"], "name": r["name"], "distance": float(r["distance"])}
        for r in rows
    ]
    if not rows:
        await db.insert_query_log(query=problem, confidence="none", top_results=[])
        return {"match": "none"}

    top = rows[0]
    confidence = confidence_label(top["distance"])
    await db.insert_query_log(
        query=problem, confidence=_confidence_short(top["distance"]), top_results=top_results
    )

    problem_id = problem_name = problem_note = None
    if top["kind"] == "problem":
        problem_id, problem_name, problem_note = top["id"], top["name"], top["note"]
    else:
        # A solution in several buckets resolves to the one it's weighted highest in.
        link = await db.pool().fetchrow(
            """
            select p.id, p.name, p.note
            from problem_solutions ps
            join problems p on p.id = ps.problem_id
            where ps.solution_id = $1
            order by ps.seed_weight desc nulls last
            limit 1
            """,
            top["id"],
        )
        if link:
            problem_id, problem_name, problem_note = link["id"], link["name"], link["note"]

    if problem_id is not None:
        links = await db.pool().fetch(
            """
            select s.name, s.content, s.action, ps.seed_weight, ps.distinguisher
            from problem_solutions ps
            join solutions s on s.id = ps.solution_id
            where ps.problem_id = $1
            order by ps.seed_weight desc nulls last
            """,
            problem_id,
        )
        solutions = [
            {
                "name": r["name"],
                "seed_weight": r["seed_weight"],
                "distinguisher": r["distinguisher"],
                "summary": _content_summary(r["name"], r["content"] or {}),
                "action": r["action"],
            }
            for r in links
        ]
        return {
            "match": "problem",
            "problem": problem_name,
            "match_confidence": confidence,
            "solutions": solutions,
            "note": problem_note or "seed_weight is a population prior — override it on direct evidence.",
        }

    return {
        "match": "single",
        "problem": top["name"],
        "match_confidence": confidence,
        "solutions": [{
            "name": top["name"],
            "seed_weight": None,
            "distinguisher": None,
            "summary": _content_summary(top["name"], top["content"] or {}),
            "action": top["action"],
        }],
        "note": None,
    }
