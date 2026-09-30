"""Graded, persisted battery runner. Same execution as run_scenarios.py, but checks
each scenario's trace against a written-in-advance 'expect' block (authored in
scenarios/battery.json before running, not fitted to whatever happened) and prints
both the full transcript and the per-assertion verdict -- meant to be redirected to
a file so results are diffable across runs, not eyeballed off a live terminal alone.

Grading only checks what's structurally verifiable from the trace (which tool was
called with what, what came back) -- match type, which solution a walkthrough was
requested for, its resolved destination, and whether any tool was called at all.
It does NOT grade response prose quality, tone, or whether a clarifying question was
phrased well -- that still needs a human reading the persisted transcript. A scenario
with no 'expect' block, or fields left out of one, isn't graded on those axes.

A scenario's top-level 'expect' is graded on the FINAL turn only. Multi-turn scenarios
can additionally put an 'expect' block on any individual turn (graded on that turn's
own trace) -- for behavior a scenario's note describes about an intermediate turn
(e.g. "turn 1 should decline with no tool call") that would otherwise only be checked
by a human reading the transcript, not actually asserted (found live 2026-08-05, via
an adversarial review that caught several notes describing unasserted turn-1 behavior).

Run inside the app container:
    docker compose exec app python -m app.run_graded_battery [scenario_name ...]
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from app import db
from app.pipeline import respond

SCENARIOS_FILE = Path(__file__).parent.parent / "scenarios" / "battery.json"
# Was 8s, from when Voyage rate-limited every burst; the account's limits are
# high now (2026-09-22) and runs go 429-free, so the default is short and
# BATTERY_DELAY_S can override it per run.
BETWEEN_SCENARIOS_DELAY_S = float(os.environ.get("BATTERY_DELAY_S", 2))

# Tools that queue an action the client runs (typed action tools, v041). Local
# rather than imported from app.tools so the same grader runs against a build
# that predates them.
ACTION_TOOLS = {"open_plugin", "set_param", "set_track_toggle"}
# v040 reached the same two actions as template KB rows through get_walkthrough
# (archive/v040-2026-09-21). Mapped onto the typed-tool names so a v040
# baseline and a v041 run grade identically.
_LEGACY_ACTION_ROWS = {"open plugin": "open_plugin", "set plugin parameter": "set_param"}
_LEGACY_ARG_NAMES = {"new": "new_instance"}

# Prose that states an action as already done. Nothing a turn queues has run
# when the reply is shown -- it runs when the user presses Run -- so any of
# these on an action turn is false (09-18 args battery: "The threshold is set
# to -18 dB" for an attach that had been refused outright, graded PASS).
# Substring-style brittle like response_contains; a regression test, not a
# semantic check.
_COMPLETION_CLAIM = re.compile(
    r"\b(?:is|are|was|were|has been|have been|now)\s+(?:now\s+)?"
    r"(?:set|loaded|added|inserted|opened|muted|soloed|unmuted|unsoloed|renamed|removed|panned)\b"
    r"|\bI(?:'ve| have)?\s+(?:set|loaded|added|inserted|opened|muted|soloed|renamed|removed)\b",
    re.IGNORECASE,
)


def _action_calls(trace: list[dict]) -> list[dict]:
    """Every action call in trace order, accepted or refused, as
    {"tool", "args", "ok"}. A refused call is kept (ok=False) so a scenario
    can require that nothing was refused and so the transcript shows it."""
    calls = []
    for c in trace:
        tool, inp = c["tool"], c.get("input") or {}
        if tool in ACTION_TOOLS:
            args = dict(inp)
        elif tool == "get_walkthrough" and inp.get("solution") in _LEGACY_ACTION_ROWS:
            tool = _LEGACY_ACTION_ROWS[inp["solution"]]
            args = {_LEGACY_ARG_NAMES.get(k, k): v for k, v in (inp.get("args") or {}).items()}
        else:
            continue
        calls.append({"tool": tool, "args": args, "ok": bool((c.get("output") or {}).get("attached"))})
    return calls


def _norm_arg(v) -> str:
    return str(v).lower().replace(" ", "")


def _action_call_matches(exp: dict, got: dict) -> bool:
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


def _ordered_subsequence(expected: list[dict], got: list[dict]) -> bool:
    """Each expected call matches a distinct actual call, in order. Extra actual
    calls in between are allowed; use no_refused_actions to forbid refusals."""
    i = 0
    for g in got:
        if i < len(expected) and _action_call_matches(expected[i], g):
            i += 1
    return i == len(expected)


def _actual_outcome(trace: list[dict], response_text: str = "", confidence_tier: str = "", sources: list | None = None,
                    auto_run: bool | None = None) -> dict:
    # cite_kb (v043, KB in the prompt) carries lookup_concept's result shape,
    # so every lookup-graded field grades a citation the same way.
    lookup_calls = [c for c in trace if c["tool"] in ("lookup_concept", "cite_kb")]
    walkthrough_calls = [c for c in trace if c["tool"] == "get_walkthrough"]
    # web_research_called: fire/no-fire grading for the new tool (v3-log.md
    # 2026-09-04) -- the trigger is entirely model-decided (prompt.py + the
    # tool's own schema description), not a deterministic gate, so this is
    # the only way to check it structurally rather than eyeballing transcripts.
    research_calls = [c for c in trace if c["tool"] == "web_research"]

    # Graded on the FIRST lookup_concept call -- the initial classification is
    # what this field is meant to test. A later call (e.g. re-querying a known
    # destination by exact name once a diagnosis is already in hand, which the
    # prompt explicitly encourages) is a legitimate second step, not a retraction
    # of the first one, and shouldn't overwrite it here.
    match = lookup_calls[0]["output"].get("match") if lookup_calls else None
    # Which bucket, not just what kind of match -- "match" alone can't tell a
    # correct fresh lookup apart from one that drifted back to a stale prior-turn
    # topic (both come back match="problem"). Found via Fable review, 2026-08-06:
    # multiturn_topic_pivot exists specifically to test that turn 2 doesn't stay
    # anchored to turn 1's diagnosis, but only asserted match="problem", which is
    # true either way. Same first-call rationale as match above.
    resolved_problem = lookup_calls[0]["output"].get("problem") if lookup_calls else None

    # Every destination actually reached this turn, not just the last call --
    # a model can legitimately call get_walkthrough more than once (e.g. once by
    # diagnosis name, once by the resolved destination's own name) and land on
    # the same correct destination both times; grading only the last call
    # treated that redundancy as a wrong answer (found live 2026-08-05).
    # Since 2026-09-21 a route reaches the card through open_setting -- called
    # directly, queued by a lookup (auto_from), or backfilled. Legacy
    # get_walkthrough attaches (a v040/v041-pre-routes build) still count.
    route_calls = [c for c in trace if c["tool"] in ("open_setting", "get_walkthrough")]
    attached_destinations = [
        c["output"]["destination"] for c in route_calls
        if c["output"].get("attached") and c["output"].get("destination")
    ]
    # The solution actually requested, not its resolved destination -- distinct fields
    # because extends_to lets multiple solutions share one destination (e.g. "no sound
    # output" and "Core Audio goes silent mid-session" both resolve to "audio settings"),
    # so attached_destinations alone can't tell a correct pick from a wrong one that
    # happens to land on the same screen.
    # Which KB solution each attached action came from: the lookup results
    # list every candidate's action, so an attached call maps back to the
    # candidate(s) it belongs to.
    by_action: dict[str, list[str]] = {}
    for c in lookup_calls:
        for sol in c["output"].get("solutions") or []:
            for call in (sol.get("action") if isinstance(sol.get("action"), list) else [sol.get("action")] if sol.get("action") else []):
                for tool, arg in call.items():
                    key = json.dumps([tool, {"name": arg} if tool == "open_setting" else arg], sort_keys=True)
                    by_action.setdefault(key, []).append(sol["name"])
    attached_solutions = [
        c["input"]["solution"] for c in walkthrough_calls if c["output"].get("attached")
    ]
    for c in trace:
        if c["tool"] in ("open_setting", "open_plugin", "set_param") and c["output"].get("attached"):
            # a dropdown value doesn't change which solution a route belongs to
            inp = {"name": c["input"].get("name")} if c["tool"] == "open_setting" else c["input"]
            attached_solutions.extend(by_action.get(json.dumps([c["tool"], inp], sort_keys=True), []))
    action_calls = _action_calls(trace)
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
        "action_calls": action_calls,
        "setting_calls": setting_calls,
        "first_tool": trace[0]["tool"] if trace else None,
        "lookup_count": len(lookup_calls),
        # Whether the card may run without the user pressing Run (v041+;
        # None on a build that predates it, which fails any auto_run assert).
        "auto_run": auto_run,
    }


# A menu path as answers write it: "File > Project Settings > Audio". Items
# start with a capital or digit; later words may be lowercase connectors.
_PATH_ITEM = r"[A-Z0-9][\w&/.'’…-]*(?: (?:[A-Z0-9][\w&/.'’…-]*|and|of|to|in|for|as|or|with))*"
_MENU_PATH = re.compile(_PATH_ITEM + r"(?:\s*>\s*" + _PATH_ITEM + r")+")
_TRAILING = re.compile(r"(?: (?:and|of|to|in|for|as|or|with))+$")


def _norm_path(text: str) -> str:
    text = text.lower().replace("…", "").replace("...", "")
    return re.sub(r"\s*>\s*", " > ", re.sub(r"\s+", " ", text)).strip()


def _route_chains() -> list[str]:
    """Every verified path the model is shown (2026-09-29: routes.json is the
    only source): runnable routes' menu chains and display-only alternatives,
    and reference entries' text, each split into its " > " chains."""
    from app import kb
    chains = []
    for name, route in kb._ALL_ROUTES.items():
        text = kb.route_path_text(name)
        for piece in re.split(r", then |\(also |\(|\)|: ", text):
            if " > " in piece:
                chains.append(_norm_path(piece))
        chains.append(_norm_path(route.get("desc") or ""))
    return chains


def ungrounded_paths(trace: list[dict], response_text: str) -> list[str]:
    """Menu paths ("X > Y") the answer states that appear neither in the text
    of any tool result this turn (a citation's KB entries with their Where
    lines, a route card's steps) nor in any route or reference path. Reported for both sides of the
    2026-09-28 KB-in-prompt A/B, not asserted: a path the model knew from the
    KB but didn't cite counts as ungrounded, which is the point."""
    source = _norm_path(" ".join(json.dumps(c.get("output"), ensure_ascii=False) for c in trace))
    chains = _route_chains()
    out = []
    for m in _MENU_PATH.finditer(response_text or ""):
        # A path item may end a sentence ("... > Recording. In the pane ..."):
        # cut at the first ". " so the next sentence isn't read as part of it.
        found = re.split(r"\.\s", m.group(0), maxsplit=1)[0].rstrip(".")
        first, *rest = _norm_path(_TRAILING.sub("", found)).split(" > ")
        # The first item can swallow sentence words ("Go to File > ..."), so
        # every tail of it is tried: grounded if any version is.
        words = first.split()
        versions = [" > ".join([" ".join(words[k:])] + rest) for k in range(len(words))]
        if not any(v in source or any(v in ch for ch in chains) for v in versions):
            out.append(versions[-1])
    return out


def _membership_pass(expected, attached: list[str]) -> bool:
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


def _grade(expect: dict, actual: dict) -> list[dict]:
    """response_contains/response_not_contains grade the final response TEXT, not the
    trace -- added 2026-08-18 (Fable review) after two real bugs shipped through
    scenarios with no expect block at all: a fabricated UI detail riding alongside a
    correctly-attached walkthrough, and a response that silently dropped half of a
    two-part question. Case-insensitive substring match, not semantic -- brittle to
    paraphrasing, but strictly better than the zero coverage these two had before."""
    verdicts = []
    for key, expected in expect.items():
        if key == "walkthrough_destination":
            got = actual["attached_destinations"]
            ok = _membership_pass(expected, got)
        elif key == "walkthrough_destination_absent":
            # This route must NOT be on the card -- e.g. live state rules the
            # candidate out (ax_override_critical_test).
            # A list: none of them may be on it.
            got = actual["attached_destinations"]
            ok = not set(expected if isinstance(expected, list) else [expected]) & set(got)
        elif key == "walkthrough_solution":
            got = actual["attached_solutions"]
            ok = _membership_pass(expected, got)
        elif key == "action_calls":
            # null -> no action call at all (accepted or refused); a list ->
            # those calls, in order, as a subsequence of what was called.
            got = actual["action_calls"]
            ok = got == [] if expected is None else _ordered_subsequence(expected, got)
        elif key == "setting_calls":
            # open_setting calls (name + dropdown value, 2026-09-22), graded
            # like action_calls but kept apart from them: routes predate the
            # action tools and existing scenarios grade them by destination.
            got = actual["setting_calls"]
            ok = _ordered_subsequence(expected, got)
        elif key == "forbidden_setting_calls":
            # None of these may be among the open_setting calls (a refused call
            # only matches a spec that says ok: false).
            got = [g for g in actual["setting_calls"] if any(_action_call_matches(e, g) for e in expected)]
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


async def run_scenario(scenario: dict) -> dict:
    print(f"\n{'=' * 100}\nSCENARIO: {scenario['name']}")
    if scenario.get("note"):
        print(f"  note: {scenario['note']}")

    messages: list[dict] = []
    trace: list[dict] = []
    auto_run = None
    response_text: str = ""
    confidence_tier: str = ""
    sources: list = []
    scenario_usage: list[dict] = []
    verdicts: list[dict] = []
    for i, turn in enumerate(scenario["turns"], 1):
        messages.append({"role": "user", "content": turn["text"]})
        ax_fixture = turn.get("ax_fixture")
        # Free-text AX capture (client-v3's core.ax_capture) -- distinct from
        # ax_fixture above, which stays the curated key/value dict used by the
        # toggle_ax_key/value_ax_key gating tests. This exercises the real
        # per-turn mechanism a live client actually sends.
        ax_state = turn.get("ax_state")

        result = await respond(messages, ax_fixture=ax_fixture, ax_state=ax_state)
        messages = result.messages
        trace = result.trace  # graded against the scenario's top-level 'expect' below
        auto_run = getattr(result, "auto_run", None)
        response_text = result.text  # ditto, for response_contains/response_not_contains
        confidence_tier = result.confidence_tier
        sources = result.sources
        scenario_usage.extend(result.usage)

        print(f"\n  --- turn {i}: {turn['text']!r} ---")
        if ax_fixture:
            print(f"  [ax_fixture: {ax_fixture}]")
        if ax_state:
            print(f"  [ax_state: {ax_state[:200]}{'...' if len(ax_state) > 200 else ''}]")
        for call in result.trace:
            print(f"    tool call: {call['tool']}({call['input']})")
            out = call["output"]
            if call["tool"] in ("lookup_concept", "cite_kb"):
                print(f"      -> match={out.get('match')!r} problem={out.get('problem')!r} confidence={out.get('match_confidence')!r}")
                for sol in out.get("solutions", []):
                    print(f"         solution: {sol['name']!r} weight={sol.get('seed_weight')} "
                          f"action={sol.get('action')} distinguisher={sol.get('distinguisher')!r}")
            else:
                print(f"      -> {out}")
        print(f"  response: {result.text}")
        print(f"  walkthrough attached: {result.walkthrough_steps is not None}"
              + (f" -> {result.walkthrough_steps}" if result.walkthrough_steps else ""))
        print(f"  confidence_tier: {result.confidence_tier!r}, sources: {len(result.sources)}, "
              f"auto_run: {getattr(result, 'auto_run', None)!r}")
        acts = _action_calls(result.trace)
        if acts:
            print("  action calls: " + "; ".join(
                f"{c['tool']}({c['args']}){'' if c['ok'] else ' REFUSED'}" for c in acts))

        if "expect" in turn:
            turn_verdicts = _grade(turn["expect"], _actual_outcome(result.trace, result.text, result.confidence_tier, result.sources,
                                                            getattr(result, "auto_run", None)))
            for v in turn_verdicts:
                v["turn"] = i
            verdicts.extend(turn_verdicts)
            print(f"  GRADED (turn {i}):")
            for v in turn_verdicts:
                status = "PASS" if v["pass"] else "FAIL"
                print(f"    [{status}] {v['field']}: expected={v['expected']!r} actual={v['actual']!r}")

    if "expect" in scenario:
        final_verdicts = _grade(scenario["expect"], _actual_outcome(trace, response_text, confidence_tier, sources, auto_run))
        for v in final_verdicts:
            v["turn"] = "final"
        verdicts.extend(final_verdicts)
        print("\n  GRADED (final):")
        for v in final_verdicts:
            status = "PASS" if v["pass"] else "FAIL"
            print(f"    [{status}] {v['field']}: expected={v['expected']!r} actual={v['actual']!r}")

    return {"name": scenario["name"], "verdicts": verdicts, "usage": scenario_usage}


async def run_battery_once(scenarios: list[dict], run_label: str = "") -> dict:
    """Runs the given scenarios once, printing full transcripts/grading exactly as
    a single run always has, plus its own summary. Returns a per-scenario signature
    (each individual assertion's turn/field/pass, not just the aggregate pass/total
    count) so main() can compare across repeated runs -- pass-rate variance across
    runs of identical code has been observed and hand-compared all session
    (43/41/41/43/42 across five runs, logged 2026-08-05); this is what makes that
    comparison automatic instead of eyeballed. Storing the full signature rather
    than just (n_pass, n_total) matters for multi-assertion scenarios: two
    different assertions failing on two different runs can produce the identical
    count, which would hide that anything changed at all (found via Fable review,
    2026-08-06)."""
    label = f" ({run_label})" if run_label else ""
    results = []
    wall_start = time.monotonic()
    await db.connect()
    try:
        for i, scenario in enumerate(scenarios):
            results.append(await run_scenario(scenario))
            if i < len(scenarios) - 1:
                await asyncio.sleep(BETWEEN_SCENARIOS_DELAY_S)
    finally:
        await db.disconnect()
    wall_elapsed = time.monotonic() - wall_start

    print(f"\n{'=' * 100}\nSUMMARY{label}")
    total_assertions = 0
    total_pass = 0
    per_scenario: dict[str, tuple] = {}
    for r in results:
        if not r["verdicts"]:
            print(f"  {r['name']}: not graded (no 'expect' block)")
            continue
        n_pass = sum(1 for v in r["verdicts"] if v["pass"])
        n_total = len(r["verdicts"])
        total_assertions += n_total
        total_pass += n_pass
        # str(v["turn"]) -- "turn" is an int for per-turn asserts but the literal
        # string "final" for the scenario's own expect block, and Python can't
        # sort a mix of the two (found live, 2026-08-06, crashed the first real
        # multi-assertion scenario this ran against).
        per_scenario[r["name"]] = tuple(
            sorted((str(v["turn"]), v["field"], v["pass"]) for v in r["verdicts"])
        )
        flag = "" if n_pass == n_total else "  <-- has failing assertion(s)"
        print(f"  {r['name']}: {n_pass}/{n_total} assertions passed{flag}")
    print(f"\nTotal{label}: {total_pass}/{total_assertions} assertions passed across {len(results)} scenario(s).")

    all_usage = [u for r in results for u in r["usage"]]
    totals = {
        "input_tokens": sum(u["input_tokens"] for u in all_usage),
        "output_tokens": sum(u["output_tokens"] for u in all_usage),
        "cache_creation_input_tokens": sum(u["cache_creation_input_tokens"] for u in all_usage),
        "cache_read_input_tokens": sum(u["cache_read_input_tokens"] for u in all_usage),
    }
    print(f"\nUSAGE{label} (excludes the {BETWEEN_SCENARIOS_DELAY_S}s between-scenario throttle delay from the")
    print(f"wall-clock figure below -- {len(scenarios) - 1} delays, {(len(scenarios) - 1) * BETWEEN_SCENARIOS_DELAY_S}s total, subtracted):")
    print(f"  API calls: {len(all_usage)}")
    print(f"  input_tokens (non-cache): {totals['input_tokens']}")
    print(f"  cache_creation_input_tokens: {totals['cache_creation_input_tokens']}")
    print(f"  cache_read_input_tokens: {totals['cache_read_input_tokens']}")
    print(f"  output_tokens: {totals['output_tokens']}")
    throttle_s = (len(scenarios) - 1) * BETWEEN_SCENARIOS_DELAY_S if len(scenarios) > 1 else 0
    print(f"  wall time: {wall_elapsed:.1f}s total, {wall_elapsed - throttle_s:.1f}s excluding throttle delay")

    return {"total_pass": total_pass, "total_assertions": total_assertions, "per_scenario": per_scenario}


async def main() -> None:
    all_scenarios = json.loads(SCENARIOS_FILE.read_text())
    args = sys.argv[1:]
    runs = 1
    if "--runs" in args:
        idx = args.index("--runs")
        runs = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]
    requested = args
    scenarios = [s for s in all_scenarios if s["name"] in requested] if requested else all_scenarios
    if requested and len(scenarios) != len(requested):
        missing = set(requested) - {s["name"] for s in scenarios}
        print(f"WARNING: scenario(s) not found: {missing}", file=sys.stderr)

    run_summaries = []
    for run_i in range(1, runs + 1):
        label = f"run {run_i}/{runs}" if runs > 1 else ""
        run_summaries.append(await run_battery_once(scenarios, run_label=label))
        if run_i < runs:
            await asyncio.sleep(BETWEEN_SCENARIOS_DELAY_S)

    if runs > 1:
        print(f"\n{'=' * 100}\nMULTI-RUN SUMMARY ({runs} runs)")
        totals_per_run = [f"{s['total_pass']}/{s['total_assertions']}" for s in run_summaries]
        print(f"  Per-run totals: {', '.join(totals_per_run)}")
        pass_counts = [s["total_pass"] for s in run_summaries]
        print(
            f"  Total assertions passed -- min: {min(pass_counts)}, max: {max(pass_counts)}, "
            f"avg: {sum(pass_counts) / len(pass_counts):.1f} (out of "
            f"{run_summaries[0]['total_assertions']} each)"
        )
        # The actual flakiness signal is WHICH scenario's outcome changed, not just
        # the aggregate moving -- a fixed number of assertions passing across runs
        # could still be a different scenario failing each time.
        all_names = sorted({name for s in run_summaries for name in s["per_scenario"]})
        flaky = [
            (name, [s["per_scenario"].get(name) for s in run_summaries])
            for name in all_names
            if len({s["per_scenario"].get(name) for s in run_summaries if name in s["per_scenario"]}) > 1
        ]
        if flaky:
            print("  Flaky scenarios (which specific assertion failed changed across runs):")
            for name, signatures in flaky:
                print(f"    {name}:")
                for run_i, sig in enumerate(signatures, 1):
                    if sig is None:
                        print(f"      run {run_i}: not present in this run")
                        continue
                    n_pass = sum(1 for _, _, ok in sig if ok)
                    failing = [f"turn={turn} {field}" for turn, field, ok in sig if not ok]
                    detail = "all passed" if not failing else "failing: " + ", ".join(failing)
                    print(f"      run {run_i}: {n_pass}/{len(sig)} -- {detail}")
        else:
            print("  No scenario's pass/fail changed across runs.")


if __name__ == "__main__":
    asyncio.run(main())
