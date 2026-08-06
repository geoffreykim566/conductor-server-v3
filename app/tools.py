"""v3 tools: lookup_concept, get_walkthrough, read_ax_state.

lookup_concept queries problems and solutions TOGETHER, in one ranked pass, and
lets whichever is actually closest by distance win — same fix as the v2 pilot's
combined-retrieval bug (a type checked first unconditionally starves a better
match; found live 2026-07-30). If the top hit is a solution that belongs to a
problem bucket, it always resolves as that bucket, never in isolation —
otherwise raw embedding-distance noise can let a bucket member win the top-1
slot and hide its siblings entirely (found live 2026-07-31,
ax_override_critical_test resolving three different ways across three runs).
get_walkthrough reads a solution's own path if it has one, otherwise follows
its extends_to link to fetch the real path from the solution it points at
(never a second embedding call — extends_to is a fixed reference, resolved by
one join, not a re-search).
"""
import asyncio
import json

from app import db
from app.embed import embed
from app.walkthrough import path_to_walkthrough_steps

# Distance bands are advisory only — informing the model's own judgment, not a
# hard filter. The old hard `RELEVANCE_FLOOR` SQL cutoff was a leftover from v1,
# inconsistent with this project's own design direction (leave judgment calls to
# the model, not a hand-tuned number) and directly implicated in a real bug: it
# admitted a different wrong match on almost every retried phrasing for a topic
# with no real KB coverage, which kept the model retrying instead of ever
# cleanly recognizing "nothing relevant here." Found live 2026-07-30.
_STRONG_MATCH = 0.40
_MODERATE_MATCH = 0.60


def _confidence_label(distance: float) -> str:
    if distance <= _STRONG_MATCH:
        return "strong"
    if distance <= _MODERATE_MATCH:
        return "moderate — treat with more skepticism, consider asking or trying a different phrasing"
    return "weak — likely not actually relevant; don't treat this as a real match"


async def _embed_with_retry(texts: list[str], input_type: str) -> list[list[float]]:
    """Voyage rate-limits on bursty test runs — retry/backoff rather than letting
    one transient 429 kill an entire scenario battery run. embed.py now also
    proactively spaces calls to avoid triggering 429s in the first place; this is
    the safety net for whatever gets through anyway. Widened 2026-08-06 after two
    full-battery runs both exhausted the old 4-attempt/60s-total budget and died
    mid-run -- 6 attempts, longer steps, ~4.5min total budget before giving up."""
    for attempt in range(6):
        try:
            return await embed(texts, input_type=input_type)
        except Exception as exc:
            if attempt == 5:
                raise
            wait = 15 * (attempt + 1)
            print(f"  [retry] embed() failed ({exc}); retrying in {wait}s")
            await asyncio.sleep(wait)

LOOKUP_CONCEPT_SCHEMA = {
    "name": "lookup_concept",
    "description": (
        "Call this before diagnosing any Logic Pro problem or naming any menu path, "
        "shortcut, or settings location — never state one from memory. Results include "
        "a match_confidence ('strong'/'moderate'/'weak') — this is advisory, not a "
        "filter; judge for yourself whether a 'moderate' or 'weak' result is actually "
        "relevant rather than treating it as grounded. If you already know the exact "
        "name of a specific setting or destination (from a solution's own fix text, "
        "from a bucket you were just given, or from your own knowledge), query with "
        "that name directly rather than paraphrasing the user's original wording."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "problem": {
                "type": "string",
                "description": (
                    "Your own short (2-8 word) query — either a normalized description of the "
                    "user's symptom (not their literal wording; recognize the underlying issue "
                    "and phrase it the way this KB names things), or the exact name of a "
                    "specific setting/destination you already know you want."
                ),
            }
        },
        "required": ["problem"],
        "additionalProperties": False,
    },
}

GET_WALKTHROUGH_SCHEMA = {
    "name": "get_walkthrough",
    "description": (
        "Attach an executable, step-by-step walkthrough for a solution returned by "
        "lookup_concept. Call this once you're confident this destination matches what "
        "you're recommending — most navigation is low-stakes (opening a panel to look at "
        "or change), so don't withhold this out of over-caution. For anything more "
        "consequential (global settings, converting/resampling audio, changes harder to "
        "undo), still call it, just say plainly what to expect in your response. No call "
        "means no walkthrough is shown; do not describe steps in prose as a substitute "
        "for calling this."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "solution": {
                "type": "string",
                "description": "The exact solution name from a lookup_concept result.",
            }
        },
        "required": ["solution"],
        "additionalProperties": False,
    },
    # Cache breakpoint for the whole tools array -- both schemas are static
    # across every call in a run, but get resent unchanged on every loop
    # iteration and every turn otherwise (measured 2026-08-05: 66 calls,
    # 207,869 uncached input tokens across a 24-scenario battery).
    "cache_control": {"type": "ephemeral"},
}

TOOLS = [LOOKUP_CONCEPT_SCHEMA, GET_WALKTHROUGH_SCHEMA]


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


async def lookup_concept(problem: str) -> dict:
    [vec] = await _embed_with_retry([problem], input_type="query")
    rows = await db.pool().fetch(
        """
        select * from (
            select 'problem'::text as kind, id, name, null::jsonb as content,
                   null::jsonb as path, null::uuid as extends_to, note,
                   (embedding <=> $1) as distance
            from problems
            union all
            select 'solution'::text as kind, id, name, content,
                   path, extends_to, null::text as note,
                   (embedding <=> $1) as distance
            from solutions
        ) combined
        order by distance
        limit 5
        """,
        vec,
    )
    if not rows:
        return {"match": "none"}

    top = rows[0]
    confidence = _confidence_label(top["distance"])

    problem_id = problem_name = problem_note = None
    if top["kind"] == "problem":
        problem_id, problem_name, problem_note = top["id"], top["name"], top["note"]
    else:
        # A solution can belong to more than one bucket (e.g. `sample rate
        # mismatch` is a candidate for both `song sounds slowed` and
        # `crackling during playback`) -- when the top hit is the solution
        # itself rather than either parent, pick deterministically by which
        # bucket it's most strongly weighted in, not by unordered row order.
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
            select s.name, s.content, s.path, s.extends_to, ps.seed_weight, ps.distinguisher
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
                "has_path": bool(r["path"] or r["extends_to"]),
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
            "has_path": bool(top["path"] or top["extends_to"]),
        }],
        "note": None,
    }


def _is_truthy(value) -> bool:
    """Real (non-fixture) AX data may not always arrive as a JSON boolean --
    accept common truthy shapes rather than silently bypassing the toggle
    gate on them. Found via adversarial review, 2026-08-05: the original
    strict `is True` check would pass right through a "true"/1/"yes" value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, int):
        return value == 1
    return False


async def get_walkthrough(solution: str, ax_fixture: dict | None = None) -> dict:
    row = await db.pool().fetchrow(
        "select path, extends_to, toggle_ax_key from solutions where name = $1", solution
    )
    if not row:
        return {"attached": False, "reason": "solution not found"}

    # Refuses on the assumption the target is always "make it visible/on"
    # (true of every toggle-type destination in the KB today). A force=true
    # escape hatch for the opposite intent (turning something OFF) was tried
    # and reverted 2026-08-05: in 3 of 4 reruns the model set force=true on
    # the exact "I can't find it, where is it" query this guard exists to
    # protect -- the same soft-instruction-doesn't-reliably-constrain-
    # behavior failure this project keeps finding elsewhere. Left as a known,
    # undocumented-fix limitation rather than a fix that made things worse.
    if row["toggle_ax_key"] and ax_fixture and _is_truthy(ax_fixture.get(row["toggle_ax_key"])):
        return {
            "attached": False,
            "reason": (
                f"already in the target state ({row['toggle_ax_key']} is already true this "
                "turn) -- attaching would toggle it away, not reveal it; tell the user it's "
                "already there instead of walking through how to enable it"
            ),
        }

    path = row["path"]
    resolved_name = solution
    if not path and row["extends_to"]:
        target = await db.pool().fetchrow(
            "select name, path from solutions where id = $1", row["extends_to"]
        )
        if target:
            path = target["path"]
            resolved_name = target["name"]

    if not path:
        return {"attached": False, "reason": "no executable path for this solution"}
    steps = path_to_walkthrough_steps(path)
    if not steps:
        return {"attached": False, "reason": "path present but produced no executable steps"}
    return {"attached": True, "destination": resolved_name, "steps": steps}
