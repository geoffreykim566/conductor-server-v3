"""Grading a scenario's trace against its written-in-advance `expect` block. Structural
checks only (which tool, what arguments, what came back) plus substring checks on the
reply text; prose quality still needs a human. Every expect key: README.md."""
import json
import re

# Tools whose calls queue an action the client runs. Local, not imported from
# app.tools, so the grader is independent of the build under test.
ACTION_TOOLS = {"open_plugin", "set_param"}

# Prose stating a queued action as already done. Nothing has run when the reply
# shows, so any of these on an action turn is false. Substring-brittle on purpose.
_COMPLETION_CLAIM = re.compile(
    r"\b(?:is|are|was|were|has been|have been|now)\s+(?:now\s+)?"
    r"(?:set|loaded|added|inserted|opened|muted|soloed|unmuted|unsoloed|renamed|removed|panned)\b"
    r"|\bI(?:'ve| have)?\s+(?:set|loaded|added|inserted|opened|muted|soloed|renamed|removed)\b",
    re.IGNORECASE,
)


def action_calls(trace: list[dict]) -> list[dict]:
    """Every action call in trace order, accepted or refused, as
    {"tool", "args", "ok"}. A refused call is kept (ok=False) so a scenario
    can require that nothing was refused and so the transcript shows it."""
    calls = []
    for c in trace:
        tool, inp = c["tool"], c.get("input") or {}
        if tool not in ACTION_TOOLS:
            continue
        calls.append({"tool": tool, "args": dict(inp), "ok": bool((c.get("output") or {}).get("attached"))})
    return calls


def _norm_arg(v) -> str:
    return str(v).lower().replace(" ", "")


def action_call_matches(exp: dict, got: dict) -> bool:
    """exp: {"tool", "args": {k: substring}, "ok": bool=True, "args_absent": [k]}.
    Args match case/space-insensitively by substring (the old walkthrough_args
    rule), so "valhalla" matches "ValhallaSupermassive". A falsy value counts
    as absent for args_absent (new_instance=false is the same as not passing it)."""
    if exp["tool"] != got["tool"] or exp.get("ok", True) != got["ok"]:
        return False
    for k, v in (exp.get("args") or {}).items():
        if k not in got["args"] or _norm_arg(v) not in _norm_arg(got["args"][k]):
            return False
    return not any(got["args"].get(k) for k in exp.get("args_absent", []))


def ordered_subsequence(expected: list[dict], got: list[dict]) -> bool:
    """Each expected call matches a distinct actual call, in order. Extra actual
    calls in between are allowed; use no_refused_actions to forbid refusals."""
    i = 0
    for g in got:
        if i < len(expected) and action_call_matches(expected[i], g):
            i += 1
    return i == len(expected)


def actual_outcome(trace: list[dict], response_text: str = "", confidence_tier: str = "", sources: list | None = None,
                   auto_run: bool | None = None) -> dict:
    # cite_kb keeps lookup_concept's result shape, so lookup fields grade citations the same way.
    lookup_calls = [c for c in trace if c["tool"] in ("lookup_concept", "cite_kb")]
    research_calls = [c for c in trace if c["tool"] == "web_research"]

    # match / resolved_problem grade the FIRST lookup: a later lookup by exact
    # name is a legitimate second step, not a retraction.
    match = lookup_calls[0]["output"].get("match") if lookup_calls else None
    resolved_problem = lookup_calls[0]["output"].get("problem") if lookup_calls else None

    # Every destination reached this turn (called directly, queued by a lookup, or backfilled).
    route_calls = [c for c in trace if c["tool"] == "open_setting"]
    attached_destinations = [
        c["output"]["destination"] for c in route_calls
        if c["output"].get("attached") and c["output"].get("destination")
    ]
    # Which KB solution each attached action came from, mapped back through the
    # lookups' candidate actions (several solutions can share one destination).
    by_action: dict[str, list[str]] = {}
    for c in lookup_calls:
        for sol in c["output"].get("solutions") or []:
            for call in (sol.get("action") if isinstance(sol.get("action"), list) else [sol.get("action")] if sol.get("action") else []):
                for tool, arg in call.items():
                    key = json.dumps([tool, {"name": arg} if tool == "open_setting" else arg], sort_keys=True)
                    by_action.setdefault(key, []).append(sol["name"])
    attached_solutions: list[str] = []
    for c in trace:
        if c["tool"] in ("open_setting", "open_plugin", "set_param") and c["output"].get("attached"):
            # a dropdown value doesn't change which solution a route belongs to
            inp = {"name": c["input"].get("name")} if c["tool"] == "open_setting" else c["input"]
            attached_solutions.extend(by_action.get(json.dumps([c["tool"], inp], sort_keys=True), []))
    setting_calls = [{"tool": "open_setting", "args": dict(c.get("input") or {}),
                      "ok": bool((c.get("output") or {}).get("attached"))}
                     for c in trace if c["tool"] == "open_setting"]

    return {
        "match": match,
        "resolved_problem": resolved_problem,
        "attached_destinations": attached_destinations,
        "attached_solutions": attached_solutions,
        "no_tool_calls": len(trace) == 0,
        "response_text": response_text,
        "web_research_called": len(research_calls) > 0,
        "web_research_queries": [c["input"].get("query") for c in research_calls],
        "confidence_tier": confidence_tier,
        "sources_present": bool(sources),
        "action_calls": action_calls(trace),
        "setting_calls": setting_calls,
        "first_tool": trace[0]["tool"] if trace else None,
        "lookup_count": len(lookup_calls),
        "auto_run": auto_run,  # None on a build that predates it
    }


def membership_pass(expected, attached: list[str]) -> bool:
    """expected forms: None -> nothing should have attached; a string -> that
    value must be among the ones attached; a list -> at least one of the
    acceptable outcomes happened, where a literal None inside the list means
    "attaching nothing is also acceptable" (used for scenarios where asking a
    clarifying question is a legitimate alternative to committing)."""
    if expected is None:
        return attached == []
    if isinstance(expected, list):
        return any((e is None and attached == []) or e in attached for e in expected)
    return expected in attached


def grade(expect: dict, actual: dict) -> list[dict]:
    """One verdict per expect key."""
    verdicts = []
    for key, expected in expect.items():
        if key == "walkthrough_destination":
            got = actual["attached_destinations"]
            ok = membership_pass(expected, got)
        elif key == "walkthrough_destination_absent":
            # This route must NOT be on the card -- e.g. live state rules the
            # candidate out (ax_override_critical_test).
            # A list: none of them may be on it.
            got = actual["attached_destinations"]
            ok = not set(expected if isinstance(expected, list) else [expected]) & set(got)
        elif key == "walkthrough_solution":
            got = actual["attached_solutions"]
            ok = membership_pass(expected, got)
        elif key == "action_calls":
            # null -> no action call at all (accepted or refused); a list ->
            # those calls, in order, as a subsequence of what was called.
            got = actual["action_calls"]
            ok = got == [] if expected is None else ordered_subsequence(expected, got)
        elif key == "setting_calls":
            # open_setting calls (name + dropdown value, 2026-09-22), graded
            # like action_calls but kept apart from them: routes predate the
            # action tools and existing scenarios grade them by destination.
            got = actual["setting_calls"]
            ok = ordered_subsequence(expected, got)
        elif key == "forbidden_setting_calls":
            # None of these may be among the open_setting calls (a refused call
            # only matches a spec that says ok: false).
            got = [g for g in actual["setting_calls"] if any(action_call_matches(e, g) for e in expected)]
            ok = got == []
        elif key == "no_action_calls":
            got = actual["action_calls"]
            ok = (got == []) == bool(expected)
        elif key == "no_refused_actions":
            got = [c for c in actual["action_calls"] if not c["ok"]]
            ok = (got == []) == bool(expected)
        elif key == "no_completion_claims":
            hits = [m.group(0) for m in _COMPLETION_CLAIM.finditer(actual.get("response_text") or "")]
            got = f"found: {hits}" if hits else "none present"
            ok = (not hits) == bool(expected)
        elif key == "first_tool":
            got = actual["first_tool"]
            ok = got in expected if isinstance(expected, list) else got == expected
        elif key in ("response_contains", "response_not_contains", "response_contains_any"):
            text = (actual.get("response_text") or "").lower()
            hits = [term for term in expected if term.lower() in text]
            if key == "response_contains":
                missing = [term for term in expected if term not in hits]
                ok = not missing
                got = "all present" if ok else f"missing: {missing}"
            elif key == "response_contains_any":
                ok = bool(hits)
                got = f"found: {hits}" if hits else "none present"
            else:
                ok = not hits
                got = "none present" if ok else f"found: {hits}"
        else:
            got = actual.get(key)
            ok = got == expected
        verdicts.append({"field": key, "expected": expected, "actual": got, "pass": ok})
    return verdicts
