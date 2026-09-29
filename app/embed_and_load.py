"""Load seed/problems.json into the problems/solutions/problem_solutions tables.

Run inside the app container:
    docker compose exec app python -m app.embed_and_load

Re-runnable: upserts by unique name, then deletes anything no longer in the seed file.
"""
import asyncio
import json
from pathlib import Path

from app import db
from app.embed import embed

SEED_FILE = Path(__file__).parent.parent / "seed" / "problems.json"
ROUTES_FILE = Path(__file__).parent.parent / "seed" / "routes.json"


def _solution_embed_text(sol: dict) -> str:
    """B2 (v3-log.md): settled after live A/B0 measurement, not just design
    argument. B0 (diagnosis solutions embedded not at all) was tried and
    measurably regressed: every mechanism-named query ("deep bass", "wide
    mix", "audio interface not recognized in Logic") lost its only anchor in
    the vector space and drifted onto a semantically-adjacent problem's
    vocabulary instead -- e.g. "song sounds thin or hollow" and "mix sounds
    muddy" share enough low-end vocabulary in their aliases that solutions
    named for the former resolved into the latter's bucket, sometimes not
    even present in the solutions returned. Rich prose embedding (the
    original dual-mode design) was ALSO checked directly against the data and
    found not to be earning its keep -- novel symptom phrasings matched via
    problem aliases either way, prose never caught anything aliases didn't
    already catch, while causing the real collision bug found live
    2026-07-30 ("wheres buffer size" matching an unrelated latency diagnosis
    whose cause text happened to mention "buffer").

    So: every solution gets its own embedding, but "destination" gets name +
    curated aliases (unique, tight, unlikely to appear elsewhere), and
    "diagnosis" gets name only -- no prose, no aliases -- restoring each
    diagnosis solution's own precise anchor without reintroducing prose-driven
    collision risk.
    """
    if sol.get("kind") == "destination":
        parts = [sol["name"]]
        parts.extend(sol.get("aliases") or [])
        return ". ".join(parts)
    return sol["name"]


def _problem_embed_text(prob: dict) -> str:
    parts = [prob["name"]]
    parts.extend(prob.get("aliases") or [])
    return ". ".join(parts)


async def main() -> None:
    seed = json.loads(SEED_FILE.read_text())
    solutions = seed["solutions"]
    problems = seed["problems"]

    print(f"Loaded {len(solutions)} solutions, {len(problems)} problems from {SEED_FILE}")

    # Every open_setting a solution maps to must name a real route -- a typo
    # here would only surface as a refused action at run time.
    routes = json.loads(ROUTES_FILE.read_text())
    for sol in solutions:
        calls = sol.get("action") or []
        for call in (calls if isinstance(calls, list) else [calls]):
            name = call.get("open_setting")
            if isinstance(name, dict):
                name = name.get("name")
            if name is not None and name not in routes:
                raise ValueError(f"{sol['name']!r} maps to open_setting {name!r}, which isn't in routes.json")

    sol_texts = [_solution_embed_text(s) for s in solutions]
    prob_texts = [_problem_embed_text(p) for p in problems]

    print(f"Embedding {len(sol_texts)} solution texts...")
    sol_embeddings = await embed(sol_texts, input_type="document")
    print(f"Embedding {len(prob_texts)} problem texts...")
    prob_embeddings = await embed(prob_texts, input_type="document")

    await db.connect()
    try:
        pool = db.pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                sol_ids: dict[str, str] = {}
                for sol, emb in zip(solutions, sol_embeddings):
                    row = await conn.fetchrow(
                        """
                        insert into solutions (name, embedding, content, path, tier, extends_to,
                                               toggle_ax_key, value_ax_key, action)
                        values ($1, $2, $3, null, $4, null, null, null, $5)
                        on conflict (name) do update set
                            embedding      = excluded.embedding,
                            content        = excluded.content,
                            path           = null,
                            tier           = excluded.tier,
                            extends_to     = null,
                            toggle_ax_key  = null,
                            value_ax_key   = null,
                            action         = excluded.action
                        returning id
                        """,
                        sol["name"], emb, sol["content"], sol.get("tier", "established"), sol.get("action"),
                    )
                    sol_ids[sol["name"]] = row["id"]

                prob_ids: dict[str, str] = {}
                for prob, emb in zip(problems, prob_embeddings):
                    row = await conn.fetchrow(
                        """
                        insert into problems (name, aliases, embedding, note)
                        values ($1, $2, $3, $4)
                        on conflict (name) do update set
                            aliases   = excluded.aliases,
                            embedding = excluded.embedding,
                            note      = excluded.note
                        returning id
                        """,
                        prob["name"], prob.get("aliases", []), emb, prob.get("note"),
                    )
                    prob_ids[prob["name"]] = row["id"]

                await conn.execute("delete from problem_solutions where problem_id = any($1::uuid[])",
                                   list(prob_ids.values()))
                link_count = 0
                for prob in problems:
                    pid = prob_ids[prob["name"]]
                    for sol in prob["solutions"]:
                        sid = sol_ids[sol["ref"]]
                        await conn.execute(
                            """
                            insert into problem_solutions (problem_id, solution_id, seed_weight, distinguisher)
                            values ($1, $2, $3, $4)
                            """,
                            pid, sid, sol.get("seed_weight"), sol.get("distinguisher"),
                        )
                        link_count += 1

                seed_sol_names = {s["name"] for s in solutions}
                seed_prob_names = {p["name"] for p in problems}
                existing_sols = await conn.fetch("select id, name from solutions")
                orphan_sols = [r["id"] for r in existing_sols if r["name"] not in seed_sol_names]
                if orphan_sols:
                    await conn.execute("delete from solutions where id = any($1::uuid[])", orphan_sols)
                existing_probs = await conn.fetch("select id, name from problems")
                orphan_probs = [r["id"] for r in existing_probs if r["name"] not in seed_prob_names]
                if orphan_probs:
                    await conn.execute("delete from problems where id = any($1::uuid[])", orphan_probs)

        print(f"Upserted {len(sol_ids)} solutions, {len(prob_ids)} problems, {link_count} problem_solutions links.")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
