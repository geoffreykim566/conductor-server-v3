"""v3 tools: lookup_concept, get_walkthrough, read_ax_state.

lookup_concept queries problems and solutions TOGETHER, in one ranked pass, and
lets whichever is actually closest by distance win — same fix as the v2 pilot's
combined-retrieval bug (a type checked first unconditionally starves a better
match; found live 2026-07-30). get_walkthrough reads a solution's path directly
off its own row — no fix_via/requires indirection needed, since a solution's
path lives on the solution itself now, not on a separately-linked entry.
"""
import asyncio
import json

from app import db
from app.config import RELEVANCE_FLOOR
from app.embed import embed
from app.walkthrough import path_to_walkthrough_steps


async def _embed_with_retry(texts: list[str], input_type: str) -> list[list[float]]:
    """Voyage rate-limits on bursty test runs — small retry/backoff rather than
    letting one transient 429 kill an entire scenario battery run."""
    for attempt in range(4):
        try:
            return await embed(texts, input_type=input_type)
        except Exception as exc:
            if attempt == 3:
                raise
            wait = 10 * (attempt + 1)
            print(f"  [retry] embed() failed ({exc}); retrying in {wait}s")
            await asyncio.sleep(wait)

LOOKUP_CONCEPT_SCHEMA = {
    "name": "lookup_concept",
    "description": (
        "Call this before diagnosing any Logic Pro problem or naming any menu path, "
        "shortcut, or settings location — never state one from memory. Call again "
        "with a different phrasing if the user says a suggested fix did not work and "
        "the current bucket of solutions is exhausted."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "problem": {
                "type": "string",
                "description": "Your own short (2-8 word) paraphrase of the user's symptom, problem, or destination.",
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
}

READ_AX_STATE_SCHEMA = {
    "name": "read_ax_state",
    "description": (
        "Reads live ground-truth state from the running Logic project (e.g. the "
        "project's actual sample rate). Prefer this over guessing whenever a diagnosis "
        "depends on a checkable setting. In this test environment this returns a "
        "scripted fixture value, not a live read."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "enum": ["project_sample_rate"],
                "description": "Which piece of state to read.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

TOOLS = [LOOKUP_CONCEPT_SCHEMA, GET_WALKTHROUGH_SCHEMA, READ_AX_STATE_SCHEMA]


def _content_summary(name: str, content: dict) -> str:
    lines = [name]
    for key, label in (
        ("symptom", "Symptom"), ("cause", "Cause"), ("fix", "Fix"),
        ("zone", "Zone"), ("remove_when", "Remove when"), ("remove_move", "Remove move"),
        ("add_when", "Add when"), ("add_move", "Add move"), ("do_not", "Do not"),
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
                   null::jsonb as path, note, (embedding <=> $1) as distance
            from problems
            union all
            select 'solution'::text as kind, id, name, content,
                   path, null::text as note, (embedding <=> $1) as distance
            from solutions
        ) combined
        where distance <= $2
        order by distance
        limit 5
        """,
        vec, RELEVANCE_FLOOR,
    )
    if not rows:
        return {"match": "none"}

    top = rows[0]
    if top["kind"] == "problem":
        links = await db.pool().fetch(
            """
            select s.name, s.content, s.path, ps.seed_weight, ps.distinguisher
            from problem_solutions ps
            join solutions s on s.id = ps.solution_id
            where ps.problem_id = $1
            order by ps.seed_weight desc nulls last
            """,
            top["id"],
        )
        solutions = [
            {
                "name": r["name"],
                "seed_weight": r["seed_weight"],
                "distinguisher": r["distinguisher"],
                "summary": _content_summary(r["name"], r["content"] or {}),
                "has_path": bool(r["path"]),
            }
            for r in links
        ]
        return {
            "match": "problem",
            "problem": top["name"],
            "solutions": solutions,
            "note": top["note"] or "seed_weight is a population prior — override it on direct evidence.",
        }

    return {
        "match": "single",
        "problem": top["name"],
        "solutions": [{
            "name": top["name"],
            "seed_weight": None,
            "distinguisher": None,
            "summary": _content_summary(top["name"], top["content"] or {}),
            "has_path": bool(top["path"]),
        }],
        "note": None,
    }


async def get_walkthrough(solution: str) -> dict:
    row = await db.pool().fetchrow("select path from solutions where name = $1", solution)
    if not row:
        return {"attached": False, "reason": "solution not found"}
    path = row["path"]
    if not path:
        return {"attached": False, "reason": "no executable path for this solution"}
    steps = path_to_walkthrough_steps(path)
    if not steps:
        return {"attached": False, "reason": "path present but produced no executable steps"}
    return {"attached": True, "destination": solution, "steps": steps}


async def read_ax_state(query: str, fixture: dict | None = None) -> dict:
    if fixture and query in fixture:
        return {"value": fixture[query], "source": "fixture"}
    return {"error": "no fixture value set for this query in this scenario", "source": "fixture"}
