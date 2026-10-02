"""How a route's path is written out for the model (Where lines, open_setting's route list)."""
from app.kb.data import ALL_ROUTES, REFERENCES


def _step_text(step: dict) -> str:
    value = step.get("value")
    if isinstance(value, list):  # anchors: whichever of these labels is showing
        return f"the control showing its current option (e.g. '{value[0]}')"
    if step.get("type") == "shortcut":
        return f"press {value}"
    if step.get("type") in ("click_text", "click_value_of"):
        return f"'{value}'"
    return str(value)


def route_path_text(name: str) -> str:
    """A route's path as the model may state it: its menus joined by " > ", then
    shortcuts / clicks, plus any display-only `also` alternative. A reference
    entry's text as written."""
    route = ALL_ROUTES[name]
    if route.get("reference"):
        return route["text"]
    parts: list[str] = []
    for step in route.get("path") or []:
        text = _step_text(step)
        if step.get("type") == "menu" and parts and not parts[-1].startswith(("press ", "'")):
            parts[-1] += " > " + text
        else:
            parts.append(text)
    out = ", then ".join(parts)
    if route.get("also"):
        out += f" (also {route['also']})"
    return out


def where_names(solution: dict) -> list[str]:
    """The routes / references a solution points at: its open_setting actions'
    routes, then its own `where` links."""
    names: list[str] = []
    action = solution.get("action") or []
    for call in (action if isinstance(action, list) else [action]):
        arg = call.get("open_setting")
        name = arg.get("name") if isinstance(arg, dict) else arg
        if name and name not in names:
            names.append(name)
    for name in solution.get("where") or []:
        if name not in names:
            names.append(name)
    return names


def where_line(solution: dict) -> str | None:
    names = where_names(solution)
    if not names:
        return None
    return "Where: " + "; ".join(
        f"{n}: {route_path_text(n)}" + (" [not runnable]" if n in REFERENCES else "") for n in names
    )
