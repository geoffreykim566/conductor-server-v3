"""Converts a solution's raw path (menu/shortcut/click steps) into client-executable
walkthrough steps. Ported as-is from the real server's router.py::_path_to_walkthrough_steps()
— pure function, no dependencies on anything v1-specific, logic unchanged."""
import re



class MissingArgs(ValueError):
    """A template path still had unfilled {placeholders}; .missing lists them."""

    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"missing args: {', '.join(missing)}")
        self.missing = missing


_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def _fill(obj, args: dict, missing: list[str]):
    """Substitute {name} placeholders in strings, recursively. A string that is
    exactly one placeholder takes the arg's own type (so numbers stay numbers)."""
    if isinstance(obj, str):
        whole = _PLACEHOLDER.fullmatch(obj)
        if whole:
            key = whole.group(1)
            if key in args and args[key] is not None:
                return args[key]
            missing.append(key)
            return obj
        def sub(m):
            key = m.group(1)
            if key in args and args[key] is not None:
                return str(args[key])
            missing.append(key)
            return m.group(0)
        return _PLACEHOLDER.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: _fill(v, args, missing) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, args, missing) for v in obj]
    return obj


def path_to_walkthrough_steps(path: list, args: dict | None = None) -> list | None:
    """Convert a raw path into walkthrough step dicts for the client, preserving order.

    Template paths carry {placeholders} (e.g. {plugin}); `args` fills them and a
    MissingArgs is raised if any remain -- the model then gets told what to supply.

    AX step types (2026-09-18, executed by client-v3's core.ax_executor, no OCR):
      ax_open_plugin  value = plugin name          -> {"ax_open_plugin": name[, "new": true]}
      ax_set_param    value = {plugin,param,value} -> {"ax_set_param": {...}}

    Consecutive "menu" steps (top menu bar items) collapse into one menu_path chain,
    since the client locates those together via menu-bar OCR. "click_text"/"click_value_of"
    steps are on-screen controls and stay one-per-step, since each needs its own locate
    call. "shortcut" steps are always their own step.
    """
    if not path:
        return None
    args = dict(args or {})
    missing: list[str] = []
    path = _fill(path, args, missing)
    if missing:
        raise MissingArgs(sorted(set(missing)))
    steps: list = []
    menu_buf: list = []

    def flush_menu() -> None:
        if menu_buf:
            steps.append({"menu_path": list(menu_buf)})
            menu_buf.clear()

    for step in path:
        if not isinstance(step, dict):
            continue
        stype = step.get("type", "")
        val = step.get("value", "")
        if not val:
            continue
        if stype == "menu":
            menu_buf.append(val)
        elif stype == "shortcut":
            flush_menu()
            steps.append({"shortcut": val})
        elif stype == "click_value_of":
            flush_menu()
            steps.append({"click_value_of": val})
        elif stype == "click_text":
            flush_menu()
            steps.append({"click_text": val})
        elif stype == "ax_open_plugin":
            flush_menu()
            step_out = {"ax_open_plugin": val}
            if str(args.get("new", "")).lower() in ("true", "1", "yes"):
                step_out["new"] = True
            if args.get("track"):
                step_out["track"] = str(args["track"])
            steps.append(step_out)
        elif stype == "ax_set_param":
            flush_menu()
            spec = dict(val)
            if args.get("track"):
                spec["track"] = str(args["track"])
            steps.append({"ax_set_param": spec})
    flush_menu()
    return steps or None
