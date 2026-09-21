"""Latency/quality harness for pipeline variants (built 2026-09-19 for the
pre-lookup + lookup-cap A/B; pre-lookup itself was rejected and removed, see
tag archive/v034-prelookup-2026-09-21). Runs a fixed, diverse subset of scenarios/battery.json under
several pipeline variants, interleaved (every scenario runs under every
variant back to back so all variants see the same Voyage/API conditions), and
records for every turn: wall time, each model call's time + cache accounting,
embed time and 429 count, lookup count, attach result, confidence tier, the
graded verdicts, and the FULL answer text. Latency without answers proved
useless on 2026-09-18 (memory: latency A/B must record full answers).

Run inside the app container (bind-mounted worktree, PYTHONPATH=/srv):
    docker compose -p v034 exec -T app python -m app.run_latency_variants [--runs N] [--variants A,A3] [--out DIR] [--scenarios a,b]

Outputs <out>/latency_variants.log (transcripts + tables) and
<out>/latency_variants.jsonl (one record per turn).
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

from app import db, pipeline, tools
from app import run_graded_battery as battery

# name -> pipeline module attributes to set for that variant. Every attribute
# any variant touches is reset to its import-time value before each turn, so
# variants can't leak into each other.
VARIANTS: dict[str, dict] = {
    "A": {},                              # pipeline as is
    "A3": {"LOOKUP_ATTEMPT_LIMIT": 3},    # one fewer lookup before the cap
}

# Diverse subset of the battery: direct / indirect / ambiguous in-KB, long
# rambling distinguishers (raw-text embedding stress), collisions, KB gaps and
# off-KB troubleshooting (cap stress), destructive carve-out (no pre-lookup
# must fire), web research, free-text AX state, and multi-turn follow-ups
# (the pre-lookup embeds the follow-up's raw text without prior context --
# the riskiest case for C).
SCENARIOS = [
    "slowed_direct",
    "buffer_size_direct_risky",
    "positive_direct_llm_alias",
    "slowed_indirect",
    "muddy_ambiguous",
    "thin_hollow_context",
    "positive_distinguisher_underdog_smart_tempo",
    "system_overload_apple_silicon_context",
    "collision_pan_law",
    "adversarial_gap_phantom_power",
    "research_no_fire_generic_troubleshooting_miss",
    "hedge_indirect_beat_from_scratch",
    "destructive_probe",
    "research_fire_artist_technique",
    "ax_state_orientation_query_uses_context",
    "ax_state_irrelevant_context_no_distraction",
    "multiturn_fallback_crackling_new_evidence_turn2",
    "multiturn_fix_worked_no_reattach",
    "multiturn_topic_pivot",
    "multiturn_clarify_then_resolve_thin_hollow",
]

BETWEEN_TURNS_S = 2.0

# ---- instrumentation -------------------------------------------------------
_model_calls: list[dict] = []
_embeds: list[dict] = []

_orig_call_model = pipeline._call_model
_orig_embed = tools.embed


async def _timed_call_model(system, tools_param, msgs, on_chunk, tool_choice=None, model=pipeline.MODEL):
    t0 = time.monotonic()
    resp = await _orig_call_model(system, tools_param, msgs, on_chunk, tool_choice, model)
    u = getattr(resp, "usage", None)
    _model_calls.append({
        "model": model.split("-")[1] if "-" in model else model,
        "s": round(time.monotonic() - t0, 2),
        "in": getattr(u, "input_tokens", None),
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "out": getattr(u, "output_tokens", None),
        "forced": tool_choice is not None,
    })
    return resp


async def _timed_embed_ok(texts, input_type="document"):
    t0 = time.monotonic()
    try:
        out = await _orig_embed(texts, input_type=input_type)
    except httpx.HTTPStatusError as exc:
        _embeds.append({"s": round(time.monotonic() - t0, 2), "status": exc.response.status_code})
        raise
    _embeds.append({"s": round(time.monotonic() - t0, 2), "status": 200})
    return out


_turn_records: list[dict] = []
_orig_respond = battery.respond


async def _timed_respond(messages, **kwargs):
    _model_calls.clear()
    _embeds.clear()
    t0 = time.monotonic()
    result = await _orig_respond(messages, **kwargs)
    wall = time.monotonic() - t0
    # _embed_with_retry sleeps 15/30/45... between attempts; that sleep is
    # not inside embed() so it shows up as wall - model - embed.
    model_s = sum(c["s"] for c in _model_calls)
    embed_s = sum(e["s"] for e in _embeds)
    pre = next((c for c in result.trace if c["tool"] == "lookup_concept"), None)
    _turn_records.append({
        "text": messages[-1]["content"] if isinstance(messages[-1].get("content"), str) else "",
        "wall_s": round(wall, 2),
        "model_s": round(model_s, 2),
        "embed_s": round(embed_s, 2),
        "other_s": round(wall - model_s - embed_s, 2),
        "model_calls": list(_model_calls),
        "n_model_calls": len(_model_calls),
        "n_embeds": len(_embeds),
        "n_429": sum(1 for e in _embeds if e["status"] == 429),
        "n_lookups": sum(1 for c in result.trace if c["tool"] == "lookup_concept"),
        "tools": [c["tool"] for c in result.trace],
        "first_lookup_query": (pre or {}).get("input", {}).get("problem"),
        "first_lookup_conf": ((pre or {}).get("output") or {}).get("match_confidence"),
        "first_lookup_match": ((pre or {}).get("output") or {}).get("match"),
        "first_lookup_problem": ((pre or {}).get("output") or {}).get("problem"),
        "attached": result.walkthrough_steps is not None,
        "tier": result.confidence_tier,
        "response": result.text,
    })
    return result


# ---- run -------------------------------------------------------------------
async def run(variants: list[str], runs: int, out_dir: Path) -> None:
    await db.connect()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "latency_variants.log"
    jsonl_path = out_dir / "latency_variants.jsonl"
    all_scn = {s["name"]: s for s in json.loads(battery.SCENARIOS_FILE.read_text())}
    missing = [n for n in SCENARIOS if n not in all_scn]
    if missing:
        sys.exit(f"scenarios not in battery.json: {missing}")

    pipeline._call_model = _timed_call_model
    tools.embed = _timed_embed_ok
    battery.respond = _timed_respond

    defaults = {attr: getattr(pipeline, attr) for v in VARIANTS.values() for attr in v}
    records: list[dict] = []
    with log_path.open("w") as log, jsonl_path.open("w") as jl:
        def out(s: str = "") -> None:
            log.write(s + "\n"); log.flush()
            print(s, flush=True)

        out(f"latency variants run {time.strftime('%Y-%m-%d %H:%M:%S')} variants={variants} runs={runs} scenarios={len(SCENARIOS)}")
        for run_i in range(1, runs + 1):
            for name in SCENARIOS:
                for v in variants:
                    overrides = VARIANTS[v]
                    for attr, val in {**defaults, **overrides}.items():
                        setattr(pipeline, attr, val)
                    _turn_records.clear()
                    buf = io.StringIO()
                    t0 = time.monotonic()
                    try:
                        with contextlib.redirect_stdout(buf):
                            res = await battery.run_scenario(all_scn[name])
                        err = None
                    except Exception as exc:  # keep going; record the failure
                        res = {"name": name, "verdicts": [], "usage": []}
                        err = f"{type(exc).__name__}: {exc}"
                    total = time.monotonic() - t0
                    verdicts = res["verdicts"]
                    n_pass = sum(1 for x in verdicts if x["pass"])
                    out(f"\n{'#' * 100}\n# {name}  variant={v} {overrides or '(defaults)'}  run={run_i}  "
                        f"scenario_wall={total:.1f}s  graded={n_pass}/{len(verdicts)}"
                        + (f"  ERROR={err}" if err else ""))
                    for ti, tr in enumerate(_turn_records, 1):
                        calls = " ".join(
                            f"{c['model'][:1]}{'F' if c['forced'] else ''}{c['s']}s(cr{c['cache_read']}/cw{c['cache_write']}/in{c['in']}/out{c['out']})"
                            for c in tr["model_calls"])
                        out(f"# turn {ti}: wall={tr['wall_s']}s model={tr['model_s']}s embed={tr['embed_s']}s other={tr['other_s']}s "
                            f"calls={tr['n_model_calls']} lookups={tr['n_lookups']} 429s={tr['n_429']} "
                            f"attached={tr['attached']} tier={tr['tier']} first_lookup={tr['first_lookup_match']}/{tr['first_lookup_conf']}")
                        out(f"#   model calls: {calls}")
                        rec = {"run": run_i, "scenario": name, "variant": v, "overrides": overrides,
                               "turn": ti, **tr,
                               "verdicts": [x for x in verdicts if x.get("turn") in (ti, "final") and ti == len(_turn_records)]
                                           if ti == len(_turn_records) else [x for x in verdicts if x.get("turn") == ti],
                               "error": err}
                        records.append(rec)
                        jl.write(json.dumps(rec) + "\n"); jl.flush()
                    out(buf.getvalue().rstrip())
                    await asyncio.sleep(BETWEEN_TURNS_S)

        # ---- summary ----
        out(f"\n\n{'=' * 100}\nSUMMARY  (per turn means; embed = Voyage call time only, retry sleeps land in 'other')")
        out(f"{'variant':8s} {'turns':>5s} {'wall':>7s} {'model':>7s} {'embed':>7s} {'other':>7s} {'calls':>6s} {'lookups':>8s} {'429s':>5s} {'attached':>9s} {'graded':>9s}")
        for v in variants:
            rs = [r for r in records if r["variant"] == v]
            if not rs:
                continue
            verd = [x for r in rs for x in r["verdicts"]]
            out(f"{v:8s} {len(rs):5d} {statistics.mean(r['wall_s'] for r in rs):7.1f} "
                f"{statistics.mean(r['model_s'] for r in rs):7.1f} {statistics.mean(r['embed_s'] for r in rs):7.1f} "
                f"{statistics.mean(r['other_s'] for r in rs):7.1f} {statistics.mean(r['n_model_calls'] for r in rs):6.1f} "
                f"{statistics.mean(r['n_lookups'] for r in rs):8.1f} {sum(r['n_429'] for r in rs):5d} "
                f"{sum(1 for r in rs if r['attached']):9d} {sum(1 for x in verd if x['pass']):4d}/{len(verd):<4d}")

        out(f"\nFIRST-LOOKUP CONFIDENCE + ANSWER TIER per variant (raw-text vs model-phrased query quality):")
        for v in variants:
            rs = [r for r in records if r["variant"] == v and r["n_lookups"]]
            conf = {}
            tier = {}
            for r in rs:
                c = (r["first_lookup_conf"] or "none").split(" ")[0]
                conf[c] = conf.get(c, 0) + 1
                tier[r["tier"]] = tier.get(r["tier"], 0) + 1
            out(f"  {v:4s} first_lookup_conf={dict(sorted(conf.items()))}  tier={dict(sorted(tier.items()))}")

        out(f"\nPER SCENARIO (final turn): wall s / model calls / lookups / attached / graded")
        header = f"{'scenario':48s}" + "".join(f"{v:>22s}" for v in variants)
        out(header)
        for name in SCENARIOS:
            cells = []
            for v in variants:
                rs = [r for r in records if r["variant"] == v and r["scenario"] == name]
                if not rs:
                    cells.append(f"{'-':>22s}"); continue
                last = rs[-1]
                verd = [x for r in rs for x in r["verdicts"]]
                wall = sum(r["wall_s"] for r in rs) / max(1, len({r['run'] for r in rs}))
                g = f"{sum(1 for x in verd if x['pass'])}/{len(verd)}" if verd else "n/a"
                cells.append(f"{wall:6.1f}s {last['n_model_calls']:2d}c {last['n_lookups']:2d}l {'Y' if last['attached'] else '-'} {g:>6s}".rjust(22))
            out(f"{name:48s}" + "".join(cells))

        fails = [(r["scenario"], r["variant"], x) for r in records for x in r["verdicts"] if not x["pass"]]
        out(f"\nFAILED ASSERTIONS ({len(fails)}):")
        for name, v, x in fails:
            out(f"  {name} [{v}] {x['field']}: expected={x['expected']!r} actual={x['actual']!r}")
        errs = [(r["scenario"], r["variant"], r["error"]) for r in records if r.get("error")]
        if errs:
            out(f"\nERRORS ({len(errs)}):")
            for e in errs:
                out(f"  {e}")
        out(f"\nwritten: {log_path} and {jsonl_path}")


def main() -> None:
    args = sys.argv[1:]
    runs, variants, out_dir = 1, ["A"], Path("test_runs") / time.strftime("%Y-%m-%d")
    if "--runs" in args:
        runs = int(args[args.index("--runs") + 1])
    if "--variants" in args:
        variants = args[args.index("--variants") + 1].split(",")
    if "--out" in args:
        out_dir = Path(args[args.index("--out") + 1])
    if "--scenarios" in args:
        SCENARIOS[:] = args[args.index("--scenarios") + 1].split(",")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        sys.exit(f"unknown variants {unknown}; known: {list(VARIANTS)}")
    asyncio.run(run(variants, runs, out_dir))


if __name__ == "__main__":
    main()
