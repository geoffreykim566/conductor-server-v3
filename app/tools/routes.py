"""Approved navigation routes (seed/routes.json, loaded by app.kb). The model only ever
names a route; the path that runs is always the file's. Route format: README.md."""
from app import kb

# Runnable routes only: reference entries (kb.REFERENCES) can never be queued.
ROUTES: dict[str, dict] = kb.ROUTES

# Values an ordered dropdown route accepts besides its listed options.
RELATIVE_VALUES = ("larger", "smaller")


def route_line(name: str, route: dict) -> str:
    line = f"- {name}: {route['desc']} -- {kb.route_path_text(name)}"
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
