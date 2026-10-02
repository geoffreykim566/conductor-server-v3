"""Graded battery runner: runs evals/scenarios/battery.json through the pipeline and
prints each transcript with its verdicts. Redirect to a file so runs are diffable.
How to run (and shard): README.md.

    docker compose exec app python -m evals.battery.runner [--runs N] [scenario_name ...]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from app.core import db
from app.pipeline import respond
from evals.battery.grader import action_calls, actual_outcome, grade

SCENARIOS_FILE = Path(__file__).parents[1] / "scenarios" / "battery.json"
# Voyage limits are high now; BATTERY_DELAY_S overrides per run.
BETWEEN_SCENARIOS_DELAY_S = float(os.environ.get("BATTERY_DELAY_S", 2))


async def run_scenario(scenario: dict) -> dict:
    print(f"\n{'=' * 100}\nSCENARIO: {scenario['name']}")
    if scenario.get("note"):
        print(f"  note: {scenario['note']}")

    messages: list[dict] = []
    trace: list[dict] = []
    auto_run = None
    response_text: str = ""
    confidence_tier: str = ""
    sources: list = []
    scenario_usage: list[dict] = []
    verdicts: list[dict] = []
    for i, turn in enumerate(scenario["turns"], 1):
        messages.append({"role": "user", "content": turn["text"]})
        ax_fixture = turn.get("ax_fixture")
        # ax_fixture: curated key/values for the toggle gate. ax_state: a real client AX dump.
        ax_state = turn.get("ax_state")

        result = await respond(messages, ax_fixture=ax_fixture, ax_state=ax_state)
        messages = result.messages
        trace = result.trace  # graded against the scenario's top-level 'expect' below
        auto_run = getattr(result, "auto_run", None)
        response_text = result.text  # ditto, for response_contains/response_not_contains
        confidence_tier = result.confidence_tier
        sources = result.sources
        scenario_usage.extend(result.usage)

        print(f"\n  --- turn {i}: {turn['text']!r} ---")
        if ax_fixture:
            print(f"  [ax_fixture: {ax_fixture}]")
        if ax_state:
            print(f"  [ax_state: {ax_state[:200]}{'...' if len(ax_state) > 200 else ''}]")
        for call in result.trace:
            print(f"    tool call: {call['tool']}({call['input']})")
            out = call["output"]
            if call["tool"] == "lookup_concept":
                print(f"      -> match={out.get('match')!r} problem={out.get('problem')!r} confidence={out.get('match_confidence')!r}")
                for sol in out.get("solutions", []):
                    print(f"         solution: {sol['name']!r} weight={sol.get('seed_weight')} "
                          f"action={sol.get('action')} distinguisher={sol.get('distinguisher')!r}")
            else:
                print(f"      -> {out}")
        print(f"  response: {result.text}")
        print(f"  walkthrough attached: {result.walkthrough_steps is not None}"
              + (f" -> {result.walkthrough_steps}" if result.walkthrough_steps else ""))
        print(f"  confidence_tier: {result.confidence_tier!r}, sources: {len(result.sources)}, "
              f"auto_run: {getattr(result, 'auto_run', None)!r}")
        acts = action_calls(result.trace)
        if acts:
            print("  action calls: " + "; ".join(
                f"{c['tool']}({c['args']}){'' if c['ok'] else ' REFUSED'}" for c in acts))

        if "expect" in turn:
            turn_verdicts = grade(turn["expect"], actual_outcome(result.trace, result.text, result.confidence_tier, result.sources,
                                                            getattr(result, "auto_run", None)))
            for v in turn_verdicts:
                v["turn"] = i
            verdicts.extend(turn_verdicts)
            print(f"  GRADED (turn {i}):")
            for v in turn_verdicts:
                status = "PASS" if v["pass"] else "FAIL"
                print(f"    [{status}] {v['field']}: expected={v['expected']!r} actual={v['actual']!r}")

    if "expect" in scenario:
        final_verdicts = grade(scenario["expect"], actual_outcome(trace, response_text, confidence_tier, sources, auto_run))
        for v in final_verdicts:
            v["turn"] = "final"
        verdicts.extend(final_verdicts)
        print("\n  GRADED (final):")
        for v in final_verdicts:
            status = "PASS" if v["pass"] else "FAIL"
            print(f"    [{status}] {v['field']}: expected={v['expected']!r} actual={v['actual']!r}")

    return {"name": scenario["name"], "verdicts": verdicts, "usage": scenario_usage}


async def run_battery_once(scenarios: list[dict], run_label: str = "") -> dict:
    """Run the scenarios once, printing transcripts, verdicts and a summary. Returns a
    per-scenario signature of every assertion's (turn, field, pass), not just counts,
    so --runs can show WHICH assertion flipped between runs."""
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
        # str(): "turn" is an int or the string "final", which don't sort together.
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
