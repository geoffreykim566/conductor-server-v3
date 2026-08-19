"""Graded, persisted battery runner. Same execution as run_scenarios.py, but checks
each scenario's trace against a written-in-advance 'expect' block (authored in
scenarios/battery.json before running, not fitted to whatever happened) and prints
both the full transcript and the per-assertion verdict -- meant to be redirected to
a file so results are diffable across runs, not eyeballed off a live terminal alone.

Grading only checks what's structurally verifiable from the trace (which tool was
called with what, what came back) -- match type, which solution a walkthrough was
requested for, its resolved destination, and whether any tool was called at all.
It does NOT grade response prose quality, tone, or whether a clarifying question was
phrased well -- that still needs a human reading the persisted transcript. A scenario
with no 'expect' block, or fields left out of one, isn't graded on those axes.

A scenario's top-level 'expect' is graded on the FINAL turn only. Multi-turn scenarios
can additionally put an 'expect' block on any individual turn (graded on that turn's
own trace) -- for behavior a scenario's note describes about an intermediate turn
(e.g. "turn 1 should decline with no tool call") that would otherwise only be checked
by a human reading the transcript, not actually asserted (found live 2026-08-05, via
an adversarial review that caught several notes describing unasserted turn-1 behavior).

Run inside the app container:
    docker compose exec app python -m app.run_graded_battery [scenario_name ...]
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from app import db
from app.pipeline import respond

SCENARIOS_FILE = Path(__file__).parent.parent / "scenarios" / "battery.json"
BETWEEN_SCENARIOS_DELAY_S = 8


def _actual_outcome(trace: list[dict], response_text: str = "") -> dict:
    lookup_calls = [c for c in trace if c["tool"] == "lookup_concept"]
    walkthrough_calls = [c for c in trace if c["tool"] == "get_walkthrough"]

    # Graded on the FIRST lookup_concept call -- the initial classification is
    # what this field is meant to test. A later call (e.g. re-querying a known
    # destination by exact name once a diagnosis is already in hand, which the
    # prompt explicitly encourages) is a legitimate second step, not a retraction
    # of the first one, and shouldn't overwrite it here.
    match = lookup_calls[0]["output"].get("match") if lookup_calls else None
    # Which bucket, not just what kind of match -- "match" alone can't tell a
    # correct fresh lookup apart from one that drifted back to a stale prior-turn
    # topic (both come back match="problem"). Found via Fable review, 2026-08-06:
    # multiturn_topic_pivot exists specifically to test that turn 2 doesn't stay
    # anchored to turn 1's diagnosis, but only asserted match="problem", which is
    # true either way. Same first-call rationale as match above.
    resolved_problem = lookup_calls[0]["output"].get("problem") if lookup_calls else None

    # Every destination actually reached this turn, not just the last call --
    # a model can legitimately call get_walkthrough more than once (e.g. once by
    # diagnosis name, once by the resolved destination's own name) and land on
    # the same correct destination both times; grading only the last call
    # treated that redundancy as a wrong answer (found live 2026-08-05).
    attached_destinations = [
        c["output"]["destination"] for c in walkthrough_calls if c["output"].get("attached")
    ]
    # The solution actually requested, not its resolved destination -- distinct fields
    # because extends_to lets multiple solutions share one destination (e.g. "no sound
    # output" and "Core Audio goes silent mid-session" both resolve to "audio settings"),
    # so attached_destinations alone can't tell a correct pick from a wrong one that
    # happens to land on the same screen.
    attached_solutions = [
        c["input"]["solution"] for c in walkthrough_calls if c["output"].get("attached")
    ]

    return {
        "match": match,
        "resolved_problem": resolved_problem,
        "attached_destinations": attached_destinations,
        "attached_solutions": attached_solutions,
        "no_tool_calls": len(trace) == 0,
        "response_text": response_text,
    }


def _membership_pass(expected, attached: list[str]) -> bool:
    """expected forms: None -> nothing should have attached; a string -> that
    value must be among the ones attached; a list -> at least one of the
    acceptable outcomes happened, where a literal None inside the list means
    "attaching nothing is also acceptable" (used for scenarios where asking a
    clarifying question is a legitimate alternative to committing)."""
    if expected is None:
        return attached == []
    if isinstance(expected, list):
        return any((e is None and attached == []) or e in attached for e in expected)
    return expected in attached


def _grade(expect: dict, actual: dict) -> list[dict]:
    """response_contains/response_not_contains grade the final response TEXT, not the
    trace -- added 2026-08-18 (Fable review) after two real bugs shipped through
    scenarios with no expect block at all: a fabricated UI detail riding alongside a
    correctly-attached walkthrough, and a response that silently dropped half of a
    two-part question. Case-insensitive substring match, not semantic -- brittle to
    paraphrasing, but strictly better than the zero coverage these two had before."""
    verdicts = []
    for key, expected in expect.items():
        if key == "walkthrough_destination":
            got = actual["attached_destinations"]
            ok = _membership_pass(expected, got)
        elif key == "walkthrough_solution":
            got = actual["attached_solutions"]
            ok = _membership_pass(expected, got)
        elif key in ("response_contains", "response_not_contains", "response_contains_any"):
            text = (actual.get("response_text") or "").lower()
            hits = [term for term in expected if term.lower() in text]
            if key == "response_contains":
                missing = [term for term in expected if term not in hits]
                ok = not missing
                got = "all present" if ok else f"missing: {missing}"
            elif key == "response_contains_any":
                ok = bool(hits)
                got = f"found: {hits}" if hits else "none present"
            else:
                ok = not hits
                got = "none present" if ok else f"found: {hits}"
        else:
            got = actual.get(key)
            ok = got == expected
        verdicts.append({"field": key, "expected": expected, "actual": got, "pass": ok})
    return verdicts


async def run_scenario(scenario: dict) -> dict:
    print(f"\n{'=' * 100}\nSCENARIO: {scenario['name']}")
    if scenario.get("note"):
        print(f"  note: {scenario['note']}")

    messages: list[dict] = []
    trace: list[dict] = []
    response_text: str = ""
    scenario_usage: list[dict] = []
    verdicts: list[dict] = []
    for i, turn in enumerate(scenario["turns"], 1):
        messages.append({"role": "user", "content": turn["text"]})
        ax_fixture = turn.get("ax_fixture")

        result = await respond(messages, ax_fixture=ax_fixture)
        messages = result.messages
        trace = result.trace  # graded against the scenario's top-level 'expect' below
        response_text = result.text  # ditto, for response_contains/response_not_contains
        scenario_usage.extend(result.usage)

        print(f"\n  --- turn {i}: {turn['text']!r} ---")
        if ax_fixture:
            print(f"  [ax_fixture: {ax_fixture}]")
        for call in result.trace:
            print(f"    tool call: {call['tool']}({call['input']})")
            out = call["output"]
            if call["tool"] == "lookup_concept":
                print(f"      -> match={out.get('match')!r} problem={out.get('problem')!r} confidence={out.get('match_confidence')!r}")
                for sol in out.get("solutions", []):
                    print(f"         solution: {sol['name']!r} weight={sol.get('seed_weight')} "
                          f"has_path={sol.get('has_path')} distinguisher={sol.get('distinguisher')!r}")
            else:
                print(f"      -> {out}")
        print(f"  response: {result.text}")
        print(f"  walkthrough attached: {result.walkthrough_steps is not None}"
              + (f" -> {result.walkthrough_steps}" if result.walkthrough_steps else ""))

        if "expect" in turn:
            turn_verdicts = _grade(turn["expect"], _actual_outcome(result.trace, result.text))
            for v in turn_verdicts:
                v["turn"] = i
            verdicts.extend(turn_verdicts)
            print(f"  GRADED (turn {i}):")
            for v in turn_verdicts:
                status = "PASS" if v["pass"] else "FAIL"
                print(f"    [{status}] {v['field']}: expected={v['expected']!r} actual={v['actual']!r}")

    if "expect" in scenario:
        final_verdicts = _grade(scenario["expect"], _actual_outcome(trace, response_text))
        for v in final_verdicts:
            v["turn"] = "final"
        verdicts.extend(final_verdicts)
        print("\n  GRADED (final):")
        for v in final_verdicts:
            status = "PASS" if v["pass"] else "FAIL"
            print(f"    [{status}] {v['field']}: expected={v['expected']!r} actual={v['actual']!r}")

    return {"name": scenario["name"], "verdicts": verdicts, "usage": scenario_usage}


async def run_battery_once(scenarios: list[dict], run_label: str = "") -> dict:
    """Runs the given scenarios once, printing full transcripts/grading exactly as
    a single run always has, plus its own summary. Returns a per-scenario signature
    (each individual assertion's turn/field/pass, not just the aggregate pass/total
    count) so main() can compare across repeated runs -- pass-rate variance across
    runs of identical code has been observed and hand-compared all session
    (43/41/41/43/42 across five runs, logged 2026-08-05); this is what makes that
    comparison automatic instead of eyeballed. Storing the full signature rather
    than just (n_pass, n_total) matters for multi-assertion scenarios: two
    different assertions failing on two different runs can produce the identical
    count, which would hide that anything changed at all (found via Fable review,
    2026-08-06)."""
    label = f" ({run_label})" if run_label else ""
    results = []
    wall_start = time.monotonic()
    await db.connect()
    try:
        for i, scenario in enumerate(scenarios):
            results.append(await run_scenario(scenario))
            if i < len(scenarios) - 1:
                await asyncio.sleep(BETWEEN_SCENARIOS_DELAY_S)
    finally:
        await db.disconnect()
    wall_elapsed = time.monotonic() - wall_start

    print(f"\n{'=' * 100}\nSUMMARY{label}")
    total_assertions = 0
    total_pass = 0
    per_scenario: dict[str, tuple] = {}
    for r in results:
        if not r["verdicts"]:
            print(f"  {r['name']}: not graded (no 'expect' block)")
            continue
        n_pass = sum(1 for v in r["verdicts"] if v["pass"])
        n_total = len(r["verdicts"])
        total_assertions += n_total
        total_pass += n_pass
        # str(v["turn"]) -- "turn" is an int for per-turn asserts but the literal
        # string "final" for the scenario's own expect block, and Python can't
        # sort a mix of the two (found live, 2026-08-06, crashed the first real
        # multi-assertion scenario this ran against).
        per_scenario[r["name"]] = tuple(
            sorted((str(v["turn"]), v["field"], v["pass"]) for v in r["verdicts"])
        )
        flag = "" if n_pass == n_total else "  <-- has failing assertion(s)"
        print(f"  {r['name']}: {n_pass}/{n_total} assertions passed{flag}")
    print(f"\nTotal{label}: {total_pass}/{total_assertions} assertions passed across {len(results)} scenario(s).")

    all_usage = [u for r in results for u in r["usage"]]
    totals = {
        "input_tokens": sum(u["input_tokens"] for u in all_usage),
        "output_tokens": sum(u["output_tokens"] for u in all_usage),
        "cache_creation_input_tokens": sum(u["cache_creation_input_tokens"] for u in all_usage),
        "cache_read_input_tokens": sum(u["cache_read_input_tokens"] for u in all_usage),
    }
    print(f"\nUSAGE{label} (excludes the {BETWEEN_SCENARIOS_DELAY_S}s between-scenario throttle delay from the")
    print(f"wall-clock figure below -- {len(scenarios) - 1} delays, {(len(scenarios) - 1) * BETWEEN_SCENARIOS_DELAY_S}s total, subtracted):")
    print(f"  API calls: {len(all_usage)}")
    print(f"  input_tokens (non-cache): {totals['input_tokens']}")
    print(f"  cache_creation_input_tokens: {totals['cache_creation_input_tokens']}")
    print(f"  cache_read_input_tokens: {totals['cache_read_input_tokens']}")
    print(f"  output_tokens: {totals['output_tokens']}")
    throttle_s = (len(scenarios) - 1) * BETWEEN_SCENARIOS_DELAY_S if len(scenarios) > 1 else 0
    print(f"  wall time: {wall_elapsed:.1f}s total, {wall_elapsed - throttle_s:.1f}s excluding throttle delay")

    return {"total_pass": total_pass, "total_assertions": total_assertions, "per_scenario": per_scenario}


async def main() -> None:
    all_scenarios = json.loads(SCENARIOS_FILE.read_text())
    args = sys.argv[1:]
    runs = 1
    if "--runs" in args:
        idx = args.index("--runs")
        runs = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]
    requested = args
    scenarios = [s for s in all_scenarios if s["name"] in requested] if requested else all_scenarios
    if requested and len(scenarios) != len(requested):
        missing = set(requested) - {s["name"] for s in scenarios}
        print(f"WARNING: scenario(s) not found: {missing}", file=sys.stderr)

    run_summaries = []
    for run_i in range(1, runs + 1):
        label = f"run {run_i}/{runs}" if runs > 1 else ""
        run_summaries.append(await run_battery_once(scenarios, run_label=label))
        if run_i < runs:
            await asyncio.sleep(BETWEEN_SCENARIOS_DELAY_S)

    if runs > 1:
        print(f"\n{'=' * 100}\nMULTI-RUN SUMMARY ({runs} runs)")
        totals_per_run = [f"{s['total_pass']}/{s['total_assertions']}" for s in run_summaries]
        print(f"  Per-run totals: {', '.join(totals_per_run)}")
        pass_counts = [s["total_pass"] for s in run_summaries]
        print(
            f"  Total assertions passed -- min: {min(pass_counts)}, max: {max(pass_counts)}, "
            f"avg: {sum(pass_counts) / len(pass_counts):.1f} (out of "
            f"{run_summaries[0]['total_assertions']} each)"
        )
        # The actual flakiness signal is WHICH scenario's outcome changed, not just
        # the aggregate moving -- a fixed number of assertions passing across runs
        # could still be a different scenario failing each time.
        all_names = sorted({name for s in run_summaries for name in s["per_scenario"]})
        flaky = [
            (name, [s["per_scenario"].get(name) for s in run_summaries])
            for name in all_names
            if len({s["per_scenario"].get(name) for s in run_summaries if name in s["per_scenario"]}) > 1
        ]
        if flaky:
            print("  Flaky scenarios (which specific assertion failed changed across runs):")
            for name, signatures in flaky:
                print(f"    {name}:")
                for run_i, sig in enumerate(signatures, 1):
                    if sig is None:
                        print(f"      run {run_i}: not present in this run")
                        continue
                    n_pass = sum(1 for _, _, ok in sig if ok)
                    failing = [f"turn={turn} {field}" for turn, field, ok in sig if not ok]
                    detail = "all passed" if not failing else "failing: " + ", ".join(failing)
                    print(f"      run {run_i}: {n_pass}/{len(sig)} -- {detail}")
        else:
            print("  No scenario's pass/fail changed across runs.")


if __name__ == "__main__":
    asyncio.run(main())
