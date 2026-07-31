"""Static single-turn battery runner. Client-less: calls pipeline.respond() directly.

Run inside the app container:
    docker compose exec app python -m app.run_scenarios [scenario_name ...]
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


async def run_scenario(scenario: dict) -> None:
    print(f"\n{'=' * 100}\nSCENARIO: {scenario['name']}")
    if scenario.get("note"):
        print(f"  note: {scenario['note']}")

    messages: list[dict] = []
    for i, turn in enumerate(scenario["turns"], 1):
        messages.append({"role": "user", "content": turn["text"]})
        ax_fixture = turn.get("ax_fixture")

        result = await respond(messages, ax_fixture=ax_fixture)
        messages = result.messages

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


async def main() -> None:
    all_scenarios = json.loads(SCENARIOS_FILE.read_text())
    requested = sys.argv[1:]
    scenarios = [s for s in all_scenarios if s["name"] in requested] if requested else all_scenarios
    if requested and len(scenarios) != len(requested):
        missing = set(requested) - {s["name"] for s in scenarios}
        print(f"WARNING: scenario(s) not found: {missing}", file=sys.stderr)

    await db.connect()
    try:
        for i, scenario in enumerate(scenarios):
            await run_scenario(scenario)
            if i < len(scenarios) - 1:
                await asyncio.sleep(BETWEEN_SCENARIOS_DELAY_S)
    finally:
        await db.disconnect()

    print(f"\n{'=' * 100}\nDone: {len(scenarios)} scenario(s) run.")


if __name__ == "__main__":
    asyncio.run(main())
