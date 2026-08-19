"""Instrumented, model-free probe harness for comparing KB embedding designs
(A = current dual-mode embed, B0 = diagnosis solutions never embedded, B1 = B0
+ real aliases). Calls lookup_concept() directly -- no Sonnet, no pipeline --
so results reflect only the retrieval mechanism itself, cheaply and
deterministically re-runnable across variants.

For each probe, logs the raw top-8 (name, kind, distance) independent of what
lookup_concept collapses that into, the actual production lookup_concept()
response (match shape, confidence, which problem it resolved to), which
routing path fired (problem_top / solution_top_then_parent / single / none),
and the top-2 distance margin as a collision-proneness signal.

Run inside the app container, tag the run with a variant label so results
don't overwrite each other across code variants:
    docker compose exec app python -m app.probe_lookup A
    docker compose exec app python -m app.probe_lookup B0
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app import db
from app.tools import _confidence_label, _embed_with_retry, lookup_concept

PROBES_FILE = Path(__file__).parent.parent / "scenarios" / "probes.json"
RESULTS_DIR = Path(__file__).parent.parent / "scenarios" / "probe_results"
SEED_FILE = Path(__file__).parent.parent / "seed" / "problems.json"

# `kind` (destination/diagnosis) is a seed-file-only concept -- embed_and_load.py
# reads it to decide embedding text, but never persists it as a DB column, so
# it has to be recovered here from the seed file, not queried from `solutions`.
_KIND_BY_NAME = {
    s["name"]: s.get("kind", "diagnosis")
    for s in json.loads(SEED_FILE.read_text())["solutions"]
}


async def _raw_top8(vec: list[float]) -> list[dict]:
    rows = await db.pool().fetch(
        """
        select * from (
            select 'problem'::text as src, name,
                   (embedding <=> $1) as distance
            from problems
            union all
            select 'solution'::text as src, name,
                   (embedding <=> $1) as distance
            from solutions
        ) combined
        order by distance
        limit 8
        """,
        vec,
    )
    return [{**dict(r), "sol_kind": _KIND_BY_NAME.get(r["name"]) if r["src"] == "solution" else None}
            for r in rows]


def _routing_path(raw: list[dict], result: dict) -> str:
    if not raw:
        return "none"
    top = raw[0]
    if result.get("match") == "none":
        return "none"
    if top["src"] == "problem":
        return "problem_top"
    if result.get("match") == "single":
        return "single"
    return "solution_top_then_parent"


async def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
    probes = json.loads(PROBES_FILE.read_text())

    await db.connect()
    try:
        results = []
        for i, probe in enumerate(probes):
            query = probe["query"]
            [vec] = await _embed_with_retry([query], input_type="query")
            raw = await _raw_top8(vec)
            result = await lookup_concept(query, _vec=vec)

            margin = None
            if len(raw) >= 2:
                margin = round(raw[1]["distance"] - raw[0]["distance"], 4)

            entry = {
                "group": probe["group"],
                "query": query,
                "expect_problem": probe.get("expect_problem"),
                "resolved_problem": result.get("problem"),
                "match": result.get("match"),
                "match_confidence": result.get("match_confidence"),
                "routing_path": _routing_path(raw, result),
                "top2_margin": margin,
                "top8": [
                    {"name": r["name"], "src": r["src"], "sol_kind": r["sol_kind"],
                     "distance": round(r["distance"], 4)}
                    for r in raw
                ],
                "solutions_returned": [s["name"] for s in result.get("solutions", [])],
            }
            results.append(entry)
            print(f"[{i+1}/{len(probes)}] {probe['group']:28s} {query[:50]:50s} "
                  f"-> {entry['routing_path']:24s} {entry['resolved_problem']}")

        RESULTS_DIR.mkdir(exist_ok=True)
        out_file = RESULTS_DIR / f"{variant}.json"
        out_file.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {len(results)} probe results to {out_file}")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
