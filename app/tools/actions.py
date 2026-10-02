"""Executors for the action tools. Pure: they validate and emit the client's wire
steps; nothing runs until the user presses Run. Wire step shapes: README.md."""
import re

from app.tools.choices import pane_only_ask, resolve_choice
from app.tools.route_steps import path_to_walkthrough_steps
from app.tools.routes import ROUTES
from app.tools.schemas import ACTION_TOOLS
from app.tools.turn_text import TURN_OFFERED_TEXT, TURN_USER_TEXT


def _is_truthy(value) -> bool:
    """Live AX values aren't always JSON booleans ("true", 1, "yes")."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, int):
        return value == 1
    return False


_MAX_NAME = 64
_ON = ("on", "true", "enable", "enabled", "yes")
_OFF = ("off", "false", "disable", "disabled", "no")
_NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:[a-z%]+)?\s*$", re.IGNORECASE)


def _clean_name(value, what: str) -> tuple[str | None, str | None]:
    name = str(value or "").strip()
    if not name:
        return None, f"{what} is empty"
    if len(name) > _MAX_NAME:
        return None, f"{what} is too long to be a {what} name"
    return name, None


def _parse_param_value(value) -> str | None:
    """'on'/'off' for a switch, else the plain number as a string ("80hz" ->
    "80", "-18 dB" -> "-18", "4:1" is not parsed). None if it's neither --
    the client's writers take float(value) or an on/off word, nothing else."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    text = str(value or "").strip().lower()
    if text in _ON:
        return "on"
    if text in _OFF:
        return "off"
    m = _NUMBER.match(text)
    return m.group(1) if m else None


def queue_open_plugin(inp: dict) -> dict:
    """Executor for open_plugin: validate, then emit the client's wire step.
    Pure -- the action runs client-side when the user presses Run."""
    plugin, err = _clean_name(inp.get("plugin"), "plugin")
    if err:
        return {"attached": False, "reason": err}
    step = {"ax_open_plugin": plugin}
    if inp.get("new_instance") is True:
        step["new"] = True
    if inp.get("track"):
        track, err = _clean_name(inp["track"], "track")
        if err:
            return {"attached": False, "reason": err}
        step["track"] = track
    return {"attached": True, "action": "open_plugin", "steps": [step]}


def queue_set_param(inp: dict) -> dict:
    """Executor for set_param: validate, then emit the client's wire step."""
    plugin, err = _clean_name(inp.get("plugin"), "plugin")
    if err:
        return {"attached": False, "reason": err}
    param, err = _clean_name(inp.get("param"), "param")
    if err:
        return {"attached": False, "reason": err}
    value = _parse_param_value(inp.get("value"))
    if value is None:
        return {
            "attached": False,
            "reason": (f"value {inp.get('value')!r} isn't a plain number or on/off -- pass the "
                       "number in the control's displayed unit (e.g. \"80\", \"-18\") or \"on\"/\"off\""),
        }
    spec = {"plugin": plugin, "param": param, "value": value}
    if inp.get("track"):
        track, err = _clean_name(inp["track"], "track")
        if err:
            return {"attached": False, "reason": err}
        spec["track"] = track
    return {"attached": True, "action": "set_param", "steps": [{"ax_set_param": spec}]}


def queue_open_setting(inp: dict, ax_fixture: dict | None = None) -> dict:
    """Executor for open_setting: an approved route's verified steps."""
    name = str(inp.get("name") or "").strip()
    route = ROUTES.get(name)
    if route is None:
        return {"attached": False, "reason": f"{name!r} isn't an approved route -- say you don't have a verified route for it"}
    # Toggle gate: running a toggle that's already in its target state would flip it away.
    key = route.get("toggle_ax_key")
    if key and ax_fixture and _is_truthy(ax_fixture.get(key)):
        return {
            "attached": False,
            "reason": (f"already in the target state ({key} is already true this turn) -- running this "
                       "would switch it away; tell the user it's already there instead"),
        }
    steps = path_to_walkthrough_steps(route["path"])
    if not steps:
        return {"attached": False, "reason": "route has no executable steps"}
    choice = route.get("choice")
    value = str(inp.get("value") or "").strip()
    chosen = None
    if value and choice is None:
        return {"attached": False, "reason": f"{name!r} has no value to choose -- call it without value"}
    if choice is not None:
        if not value:
            dropped, steps = steps[-1], steps[:-1]   # the pane only: never click the dropdown open
            # The dropped row becomes the last step's `expect` (README: Pane-only routes).
            if steps and "click_value_of" in dropped and ("click_text" in steps[-1] or "menu_path" in steps[-1]):
                steps[-1] = {**steps[-1], "expect": [dropped["click_value_of"]]}
        else:
            chosen, reason = resolve_choice(value, choice, TURN_USER_TEXT.get(), TURN_OFFERED_TEXT.get())
            if chosen is None:
                return {"attached": False, "reason": reason}
            # `reopen` lets the client's Revert get back to this dropdown.
            pick = {"choose": chosen, "reopen": list(steps)}
            if choice.get("shows"):   # the control's shorter display, for the read-back
                pick["shows"] = dict(choice["shows"])
            steps = steps + [pick]
    # A route fixed to one dropdown option ends on a choose step (README: `picks`).
    if route.get("picks") and choice is None:
        chosen = route["picks"]
        steps = steps + [{"choose": chosen, "reopen": list(steps)}]
    # attached + destination + steps is the shape pipeline/backfill.py looks for.
    out = {"attached": True, "action": "open_setting", "destination": name, "steps": steps}
    if chosen:
        out["chooses"] = chosen
    elif choice is not None:
        out["note"] = "opens the pane only -- no value is chosen; " + pane_only_ask(choice)
    return out


def action_calls(action) -> list[tuple[str, dict]]:
    """A solution's stored action ({"open_setting": "x"} / {"open_plugin": {...}}
    / a list of those) as (tool name, tool input) pairs -- the same shape a
    model's own call would have, so an auto-attached action and a direct call
    are indistinguishable downstream."""
    out = []
    for call in (action if isinstance(action, list) else [action] if action else []):
        for tool, arg in call.items():
            if tool == "open_setting":
                # a route name, or {"name", "value"} to choose a dropdown value,
                # checked like a model's call
                out.append((tool, dict(arg) if isinstance(arg, dict) else {"name": arg}))
            elif tool in ACTION_TOOLS and isinstance(arg, dict):
                out.append((tool, dict(arg)))
    return out
