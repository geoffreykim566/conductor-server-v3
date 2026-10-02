"""Approved navigation routes (seed/routes.json). The model only ever names a route;
the path that runs is always this file's. Route format and `choice` blocks: README.md."""
import json
from pathlib import Path

ROUTES: dict[str, dict] = json.loads(
    (Path(__file__).parents[2] / "seed" / "routes.json").read_text()
)

# Values an ordered dropdown route accepts besides its listed options.
RELATIVE_VALUES = ("larger", "smaller")


def route_line(name: str, route: dict) -> str:
    line = f"- {name}: {route['desc']}"
    choice = route.get("choice")
    if choice is None:
        return line
    options = choice.get("options") or []
    if not options:
        return line + " [opens the pane only; no value]"
    values = " | ".join(options) + (" | larger | smaller" if choice.get("ordered") else "")
    free = choice.get("model_may_pick") or []
    who = (f"you may choose {' or '.join(free)} yourself; any other value only if the user named it"
           if free else "an exact value only if the user named it")
    if choice.get("ordered"):
        who += "; larger/smaller move one step from its current value"
    return line + f" [value: {values} -- {who}]"
