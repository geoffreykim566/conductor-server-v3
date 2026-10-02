"""The reply's card: identity keys for actions, commit-to-one between candidates
from the same problem bucket, and span bookkeeping when a step is replaced.
Rules and why: README.md (One card per turn)."""
import json

from app import tools
from app.pipeline import settings


def attach_key(name: str, inp: dict) -> str:
    """Exact identity of a call: tool + every argument."""
    return json.dumps([name, inp], sort_keys=True, default=str)


def endorse_key(name: str, inp: dict) -> str:
    """Identity by target only. The model calling open_plugin{plugin, track} after a
    lookup queued open_plugin{plugin} is agreement, so its version replaces the step."""
    if name == "open_plugin":
        return f"open_plugin:{str(inp.get('plugin', '')).strip().lower()}"
    if name == "set_param":
        return f"set_param:{str(inp.get('plugin', '')).strip().lower()}:{str(inp.get('param', '')).strip().lower()}"
    if name == "open_setting":
        return f"open_setting:{str(inp.get('name', '')).strip().lower()}"
    return attach_key(name, inp)


def _cand_key(name: str, inp: dict) -> str:
    """Which candidate a call is: an open_setting call is its route, whatever value it picks."""
    if name == "open_setting":
        return attach_key(name, {"name": inp.get("name")})
    return attach_key(name, inp)


def bucket_candidates(trace: list[dict]) -> dict[str, str]:
    """Candidate key -> problem bucket, for every multi-candidate lookup this turn
    plus moderate single matches (settings.PICK_ON_MODERATE_SINGLE). Weak matches
    are skipped: forcing a pick on one pressures the model into queuing a half-match."""
    out: dict[str, str] = {}
    for c in trace:
        o = c["output"] if c["tool"] == "lookup_concept" else {}
        conf = str(o.get("match_confidence", ""))
        moderate_single = (
            settings.PICK_ON_MODERATE_SINGLE and o.get("match") == "single" and conf.startswith("moderate")
        )
        if o.get("match") != "problem" and not moderate_single:
            continue
        if conf.startswith("weak"):
            continue
        for sol in o.get("solutions") or []:
            for tool, inp in tools.action_calls(sol.get("action")):
                out.setdefault(_cand_key(tool, inp), o.get("problem"))
    return out


def is_alternative_pick(name: str, inp: dict, trace: list[dict]) -> bool:
    """True if another candidate from this action's bucket is already on the card:
    a fallback fix, not a second request."""
    cands = bucket_candidates(trace)
    key = _cand_key(name, inp)
    problem = cands.get(key)
    if problem is None:
        return False
    return any(
        c["tool"] in tools.ACTION_TOOLS and c["output"].get("attached")
        and cands.get(_cand_key(c["tool"], c["input"])) == problem
        and _cand_key(c["tool"], c["input"]) != key
        for c in trace
    )


def needs_pick(trace: list[dict], card: list) -> bool:
    """A bucket lookup returned actionable candidates and the turn is about to end
    with nothing on the card and no clarifying question."""
    if card or any(c["tool"] == "ask_clarifying_question" for c in trace):
        return False
    return bool(bucket_candidates(trace))


def replace_span(card_steps: list, start: int, end: int, new: list, *span_maps: dict) -> None:
    """Swap card_steps[start:end] for `new` and shift every recorded span after it."""
    card_steps[start:end] = new
    shift = len(new) - (end - start)
    if shift:
        for spans in span_maps:
            for k, (a, b) in list(spans.items()):
                if a >= end:
                    spans[k] = (a + shift, b + shift)


def mark_replaced(trace: list[dict], name: str) -> None:
    """Earlier open_setting calls for this route no longer describe the card; the
    writer (card_descriptions) and the grader stop counting them."""
    for c in trace:
        if c["tool"] == "open_setting" and c["input"].get("name") == name and c["output"].get("attached"):
            c["output"] = {**c["output"], "attached": False, "replaced": True}


def card_descriptions(trace: list[dict]) -> list[str]:
    """One plain line per queued step, built from the calls themselves: what the
    writer is told the card holds."""
    out = []
    for c in trace:
        if c["tool"] not in tools.ACTION_TOOLS or not c["output"].get("attached"):
            continue
        inp = c["input"]
        if c["tool"] == "open_plugin":
            where = f" on {inp['track']}" if inp.get("track") else " on the selected track"
            out.append(f"opens {inp.get('plugin')}{where}"
                       + (" (a second copy)" if inp.get("new_instance") else ""))
        elif c["tool"] == "set_param":
            where = f" on {inp['track']}" if inp.get("track") else ""
            out.append(f"sets {inp.get('plugin')} {inp.get('param')} to {inp.get('value')}{where}")
        elif c["tool"] == "open_setting":
            route = tools.ROUTES.get(inp.get("name"), {})
            line = f"opens {inp.get('name')} -- {route.get('desc', '')}".rstrip(" -")
            chosen = c["output"].get("chooses")
            if chosen in tools.RELATIVE_VALUES:
                line += f", then moves it one step {chosen} than whatever it's set to now"
            elif chosen:
                line += f", then sets it to {chosen}"
            elif route.get("choice") is not None:
                line += f" (opens the pane only -- no value is chosen; {tools.pane_only_ask(route['choice'])})"
            out.append(line)
    return out
