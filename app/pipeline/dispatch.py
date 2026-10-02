"""Running one tool call against the turn's card: duplicates, endorsements of
citation-queued steps, re-visited settings, the attach cap, commit-to-one, and the
auto-attach a citation triggers. Rules: README.md (One card per turn)."""
import json

from app import tools
from app.pipeline import notes, settings
from app.pipeline.cards import attach_key, endorse_key, is_alternative_pick, mark_replaced
from app.pipeline.executors import EXECUTORS
from app.pipeline.turn_state import TurnState


async def _execute(tu, state: TurnState, ax_fixture: dict | None, research_denied: bool) -> dict:
    name, inp = tu.name, tu.input
    if name in tools.ACTION_TOOLS and endorse_key(name, inp) in state.queued_by_lookup:
        # The model called what a citation already queued: agreement. Its version
        # replaces the step, which stops counting as lookup-queued (may auto-run).
        result = await EXECUTORS[name](inp, ax_fixture)
        if result.get("attached"):
            # Only on success: a refused version leaves the queued step in place.
            key = endorse_key(name, inp)
            start, end = state.queued_by_lookup.pop(key)
            new = result.get("steps") or []
            state.replace_steps(start, end, new)
            if name == "open_setting":
                state.route_spans[key] = (start, start + len(new))
            state.attach_keys.add(attach_key(name, inp))
            result = {**result, "already_queued": True}
        return result
    if name in tools.ACTION_TOOLS and attach_key(name, inp) in state.attach_keys:
        return {"attached": False, "reason": notes.DUPLICATE_ATTACH_REFUSAL}
    if name == "open_setting" and endorse_key(name, inp) in state.route_spans:
        # One visit per setting per card; the later call is the correction and wins.
        result = await EXECUTORS[name](inp, ax_fixture)
        if result.get("attached"):
            key = endorse_key(name, inp)
            start, end = state.route_spans[key]
            new = result.get("steps") or []
            state.replace_steps(start, end, new)
            state.route_spans[key] = (start, start + len(new))
            mark_replaced(state.trace, inp.get("name"))
            state.attach_keys.add(attach_key(name, inp))
            result = {**result, "replaces_earlier": True}
        return result
    if name in tools.ACTION_TOOLS and len(state.attach_keys) >= settings.MAX_ATTACHES_PER_TURN:
        return {"attached": False, "reason": notes.ATTACH_CAP_REFUSAL}
    if name in tools.ACTION_TOOLS and is_alternative_pick(name, inp, state.trace):
        return {"attached": False, "reason": notes.FALLBACK_REFUSAL}
    if name == "web_research" and research_denied:
        return dict(notes.RESEARCH_DECLINED)
    executor = EXECUTORS.get(name)
    return {"error": f"unknown tool {name!r}"} if executor is None else await executor(inp, ax_fixture)


async def _queue_from_citation(result: dict, state: TurnState, ax_fixture: dict | None) -> tuple[dict, list[dict]]:
    """Cited solutions' actions are queued, in citation order, if the model didn't
    call them. Commit-to-one holds inside one citation too: a second cited
    candidate from a bucket an earlier one filled is an alternative. Recorded in
    the citation's own result as on_card (the only record of it)."""
    on_card: list[dict] = []
    sols = {sol["name"]: sol for sol in result.get("solutions") or []}
    calls, used_buckets = [], set()
    for name in result["recommended"]:
        sol = sols.get(name) or {}
        alt = sol.get("bucket") is not None and sol["bucket"] in used_buckets
        if sol.get("bucket") is not None:
            used_buckets.add(sol["bucket"])
        calls += [(tool, inp, alt) for tool, inp in tools.action_calls(sol.get("action"))]
    for tool, inp, alt in calls:
        key = attach_key(tool, inp)
        if (key in state.attach_keys or len(state.attach_keys) >= settings.MAX_ATTACHES_PER_TURN
                or (tool == "open_setting" and endorse_key(tool, inp) in state.route_spans)):
            continue
        if alt or is_alternative_pick(tool, inp, state.trace):
            on_card.append({"tool": tool, "input": inp,
                            "output": {"attached": False, "reason": notes.FALLBACK_REFUSAL}})
            continue
        out = await EXECUTORS[tool](inp, ax_fixture)
        on_card.append({"tool": tool, "input": inp, "output": out})
        if out.get("attached"):
            state.queued_by_lookup[endorse_key(tool, inp)] = state.append_steps(tool, inp, out.get("steps"))
    if on_card:
        refused = [o["output"].get("reason") for o in on_card if not o["output"].get("attached")]
        note = notes.ON_CARD_NOTE
        if any(o["tool"] == "open_setting" and o["output"].get("note") for o in on_card):
            note += notes.PANE_ONLY_NOTE
        result = {**result, "on_card": on_card,
                  "on_card_note": note if not refused else
                  "This solution's action was NOT queued: " + "; ".join(r for r in refused if r)}
    return result, on_card


async def handle_tool_use(tu, state: TurnState, ax_fixture: dict | None,
                          research_denied: bool, usage: list[dict]) -> dict:
    """Run one tool call, record it on the state, and return its tool_result block."""
    result = await _execute(tu, state, ax_fixture, research_denied)
    # web_research reports its own API spend; count it, never show it to the model.
    call_usage = result.pop("_usage", None) if isinstance(result, dict) else None
    if call_usage:
        usage.append(call_usage)
    on_card: list[dict] = []
    if tu.name == "cite_kb" and result.get("recommended"):
        result, on_card = await _queue_from_citation(result, state, ax_fixture)
    state.trace.append({"tool": tu.name, "input": tu.input, "output": result})
    state.trace.extend({**o, "auto_from": result.get("problem")} for o in on_card)
    if (tu.name in tools.ACTION_TOOLS and result.get("attached")
            and not result.get("already_queued") and not result.get("replaces_earlier")):
        span = state.append_steps(tu.name, tu.input, result.get("steps"))
        if tu.name == "open_setting":
            state.route_spans[endorse_key(tu.name, tu.input)] = span
    return {"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(result)}
