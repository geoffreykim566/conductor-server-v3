"""Interactive single-turn driver for reactive multi-turn conversations — persists
state to a JSON file between separate `docker compose exec` invocations, since each
call is a fresh process. Use for scenarios where a pre-scripted turn sequence would
break the moment the model's actual response diverges from the assumed script.

Usage: docker compose exec app python -m evals.drive_turn <scenario_name> <text> [ax_fixture_json]
"""
import asyncio
import json
import sys

from app.core import db
from app.pipeline import respond

STATE_FILE = "/tmp/v3_interactive_state.json"


def _default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    raise TypeError(f"not serializable: {type(o)}")


async def main() -> None:
    scenario = sys.argv[1]
    text = sys.argv[2]
    ax_fixture = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None

    await db.connect()
    try:
        try:
            all_state = json.load(open(STATE_FILE))
        except FileNotFoundError:
            all_state = {}
        messages = all_state.get(scenario, [])
        messages.append({"role": "user", "content": text})

        result = await respond(messages, ax_fixture=ax_fixture)
        normalized = json.loads(json.dumps(result.messages, default=_default))
        all_state[scenario] = normalized
        json.dump(all_state, open(STATE_FILE, "w"))

        print(f"\n--- user: {text!r} ---")
        if ax_fixture:
            print(f"[ax_fixture: {ax_fixture}]")
        for call in result.trace:
            print(f"  tool: {call['tool']}({call['input']}) -> {call['output']}")
        print(f"\nresponse: {result.text}")
        print(f"\nwalkthrough attached: {result.walkthrough_steps is not None}"
              + (f" -> {result.walkthrough_steps}" if result.walkthrough_steps else ""))
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
