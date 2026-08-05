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

Run inside the app container:
    docker compose exec app python -m app.run_graded_battery [scenario_name ...]
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app import db
from app.pipeline import respond

SCENARIOS_FILE = Path(__file__).parent.parent / "scenarios" / "battery.json"
BETWEEN_SCENARIOS_DELAY_S = 8


def _actual_outcome(trace: list[dict]) -> dict:
    lookup_calls = [c for c in trace if c["tool"] == "lookup_concept"]
    walkthrough_calls = [c for c in trace if c["tool"] == "get_walkthrough"]

    # Graded on the FIRST lookup_concept call -- the initial classification is
    # what this field is meant to test. A later call (e.g. re-querying a known
    # destination by exact name once a diagnosis is already in hand, which the
    # prompt explicitly encourages) is a legitimate second step, not a retraction
    # of the first one, and shouldn't overwrite it here.
    match = lookup_calls[0]["output"].get("match") if lookup_calls else None

    # Every destination actually reached this turn, not just the last call --
    # a model can legitimately call get_walkthrough more than once (e.g. once by
    # diagnosis name, once by the resolved destination's own name) and land on
    # the same correct destination both times; grading only the last call
    # treated that redundancy as a wrong answer (found live 2026-08-05).
    attached_destinations = [
        c["output"]["destination"] for c in walkthrough_calls if c["output"].get("attached")
    ]

    return {
        "match": match,
        "attached_destinations": attached_destinations,
        "no_tool_calls": len(trace) == 0,
    }


def _walkthrough_destination_pass(expected, attached_destinations: list[str]) -> bool:
    """expected forms: None -> nothing should have attached; a string -> that
    destination must be among the ones attached; a list -> at least one of the
    acceptable outcomes happened, where a literal None inside the list means
    "attaching nothing is also acceptable" (used for scenarios where asking a
    clarifying question is a legitimate alternative to committing)."""
    if expected is None:
        return attached_destinations == []
    if isinstance(expected, list):
        return any(
            (e is None and attached_destinations == []) or e in attached_destinations
            for e in expected
        )
    return expected in attached_destinations


def _grade(expect: dict, actual: dict) -> list[dict]:
    verdicts = []
    for key, expected in expect.items():
        if key == "walkthrough_destination":
            got = actual["attached_destinations"]
            ok = _walkthrough_destination_pass(expected, got)
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
    for i, turn in enumerate(scenario["turns"], 1):
        messages.append({"role": "user", "content": turn["text"]})
        ax_fixture = turn.get("ax_fixture")

        result = await respond(messages, ax_fixture=ax_fixture)
        messages = result.messages
        trace = result.trace  # graded on the final turn's trace

        print(f"\n  --- turn {i}: {turn['text']!r} ---")
        if ax_fixture:
            print(f"  [ax_fixture: {ax_fixture}]")
        if result.thinking:
            print("  --- thinking ---")
            for block in result.thinking:
                print(f"    {block}")
            print("  --- end thinking ---")
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

    verdicts: list[dict] = []
    if "expect" in scenario:
        actual = _actual_outcome(trace)
        verdicts = _grade(scenario["expect"], actual)
        print("\n  GRADED:")
        for v in verdicts:
            status = "PASS" if v["pass"] else "FAIL"
            print(f"    [{status}] {v['field']}: expected={v['expected']!r} actual={v['actual']!r}")

    return {"name": scenario["name"], "verdicts": verdicts}


async def main() -> None:
    all_scenarios = json.loads(SCENARIOS_FILE.read_text())
    requested = sys.argv[1:]
    scenarios = [s for s in all_scenarios if s["name"] in requested] if requested else all_scenarios
    if requested and len(scenarios) != len(requested):
        missing = set(requested) - {s["name"] for s in scenarios}
        print(f"WARNING: scenario(s) not found: {missing}", file=sys.stderr)

    results = []
    await db.connect()
    try:
        for i, scenario in enumerate(scenarios):
            results.append(await run_scenario(scenario))
            if i < len(scenarios) - 1:
                await asyncio.sleep(BETWEEN_SCENARIOS_DELAY_S)
    finally:
        await db.disconnect()

    print(f"\n{'=' * 100}\nSUMMARY")
    total_assertions = 0
    total_pass = 0
    for r in results:
        if not r["verdicts"]:
            print(f"  {r['name']}: not graded (no 'expect' block)")
            continue
        n_pass = sum(1 for v in r["verdicts"] if v["pass"])
        n_total = len(r["verdicts"])
        total_assertions += n_total
        total_pass += n_pass
        flag = "" if n_pass == n_total else "  <-- has failing assertion(s)"
        print(f"  {r['name']}: {n_pass}/{n_total} assertions passed{flag}")
    print(f"\nTotal: {total_pass}/{total_assertions} assertions passed across {len(results)} scenario(s).")


if __name__ == "__main__":
    asyncio.run(main())
