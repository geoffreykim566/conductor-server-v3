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


def _solution_embed_text(sol: dict) -> str:
    """Two embedding modes, deliberately different — mixing them caused a real bug
    (found live 2026-07-30): a "destination" solution's tight name+aliases doesn't
    bleed into unrelated diagnoses, but a "diagnosis" solution's rich symptom/cause
    text does need to match varied symptom phrasing, and letting settings ride
    along in that rich text caused false collisions ("wheres buffer size" matching
    an unrelated latency diagnosis whose cause text happened to mention "buffer").

    "destination": name + curated aliases only — no diagnostic prose, so a plain
    "wheres X" query can't collide with an unrelated diagnosis that merely mentions
    X in passing.
    "diagnosis" (default): name + symptom/cause/zone — needs to match varied
    real-world phrasing of a symptom, not just a name.
    """
    parts = [sol["name"]]
    if sol.get("kind") == "destination":
        parts.extend(sol.get("aliases") or [])
        return ". ".join(parts)

    c = sol.get("content", {})
    for key in ("symptom", "cause", "zone"):
        if c.get(key):
            parts.append(c[key])
    for key in ("remove_when", "add_when"):
        if c.get(key):
            parts.append(c[key])
    return ". ".join(parts)


def _problem_embed_text(prob: dict) -> str:
    parts = [prob["name"]]
    parts.extend(prob.get("aliases") or [])
    return ". ".join(parts)


async def main() -> None:
    seed = json.loads(SEED_FILE.read_text())
    solutions = seed["solutions"]
    problems = seed["problems"]

    print(f"Loaded {len(solutions)} solutions, {len(problems)} problems from {SEED_FILE}")

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
                        insert into solutions (name, embedding, content, path, tier)
                        values ($1, $2, $3, $4, $5)
                        on conflict (name) do update set
                            embedding = excluded.embedding,
                            content   = excluded.content,
                            path      = excluded.path,
                            tier      = excluded.tier
                        returning id
                        """,
                        sol["name"], emb, sol["content"], sol.get("path"), sol.get("tier", "established"),
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
