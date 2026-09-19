"""Track B mode-classification experiment (standalone, not part of the production
pipeline). Measures whether the decider model (MODEL, see app/config.py) can pick
a turn's behaviour -- silent_execute / paced_walkthrough / explain_only / clarify
-- from phrasing + context alone, with no user-set mode toggle.

Reuses the real SYSTEM_PROMPT (app/prompt.py) unmodified, plus an appended
instruction block (this file only) that defines the four modes and asks for a
bare JSON verdict. Same client/model construction pattern as app/pipeline.py and
app/research.py (AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY), model=MODEL) --
this is a read-only classification call, no tools, nothing executed against a
real Logic Pro project.

Cases come from scenarios/mode_cases.json (id, bucket, turns, expected_mode,
optional ax_state, note). A case's `turns` list is the literal conversation to
send: user turns are {"user": "..."}, and for the ~10 two-turn "flip" cases a
hand-written {"assistant": "..."} placeholder stands in for what the real
pipeline would have said on turn 1 (this script never calls app.pipeline.respond
-- it isolates the mode-classification question from the rest of the tool loop).

Run inside the app container:
    docker compose exec app python -m app.experiment_mode_classify [--runs 3]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from anthropic import AsyncAnthropic, APIStatusError

from app.config import CENTRAL_ANTHROPIC_KEY, MAX_TOKENS, MODEL
from app.prompt import SYSTEM_PROMPT

log = logging.getLogger(__name__)

CASES_FILE = Path(__file__).parent.parent / "scenarios" / "mode_cases.json"
RESULTS_DIR = Path(__file__).parent.parent / "scenarios" / "mode_results"

VALID_MODES = ("silent_execute", "paced_walkthrough", "explain_only", "clarify")
VALID_INTENTS = ("command", "question", "info")
VALID_CONFIDENCE = ("high", "medium", "low")

# Delay between individual model calls -- this script never touches Voyage/embed
# (see app/embed.py's [voyage_429] marker, which is about that rate limit, not
# this one), but 50 cases x N runs is still a real burst of Anthropic calls in a
# short window, so a small delay is cheap insurance against an ordinary Anthropic
# 429 rather than a documented requirement here.
BETWEEN_CALLS_DELAY_S = 1.5

# Appended to the real SYSTEM_PROMPT unmodified (never edited in place) -- this
# experiment's entire question is whether the PRODUCTION prompt plus a bolted-on
# instruction block is enough for the decider to pick the right mode, not whether
# a rewritten prompt could do better. Mirrors the four-mode definitions and
# examples from the testing plan verbatim so the model sees the same taxonomy a
# human grader is using.
_MODE_INSTRUCTIONS = """\


## Mode classification (test harness only -- not part of the real product prompt)

For THIS turn only, decide which of four behaviors the user actually wants, based \
on phrasing and whatever context is available. There is no user-set mode toggle -- \
you must infer it from the conversation alone.

- silent_execute: a direct command that the assistant should just carry out (after \
one Enter/confirm from the user) -- e.g. "open channel eq", "mute track 3", "set \
the low cut to 80hz".
- paced_walkthrough: the user asks to be shown or taught how to do something, step \
by step -- e.g. "can you show me how to open channel eq", "walk me through adding \
reverb".
- explain_only: an informational question with no action wanted -- e.g. "what does \
ratio do on a compressor".
- clarify: genuinely ambiguous (the target or scope is unclear) or risky/irreversible \
(e.g. "delete all tracks", "delete my entire project") -- ask or decline rather than \
act.

If an earlier turn set a direct/indirect/informational framing and the user's latest \
message changes it (e.g. asks to skip the explanation and just do it, or asks to stop \
and explain first), classify the LATEST turn's actual request, not the earlier one.

Answer with ONLY a single JSON object and nothing else -- no markdown fences, no \
commentary before or after it:
{"mode": "silent_execute" | "paced_walkthrough" | "explain_only" | "clarify", \
"intent": "command" | "question" | "info", "confidence": "high" | "medium" | "low", \
"why": "<one sentence>"}
"""

_client = AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY)


def _build_system(ax_state: str | None) -> list[dict]:
    """Mirrors app/pipeline.py's respond() system-block construction (see its
    ~947-1001): the real SYSTEM_PROMPT as one cached block, plus a second,
    separate "## Live state for this turn" block when ax_state is present --
    same heading and wording pipeline.py uses, so this experiment sees live
    state shaped exactly the way the production decider does."""
    system_text = SYSTEM_PROMPT + _MODE_INSTRUCTIONS
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
    if ax_state:
        system.append({
            "type": "text",
            "text": (
                "\n\n## Live state for this turn\n\n"
                "Values read directly from the running Logic Pro project via the "
                "Accessibility API -- not something the user said or you inferred. "
                "Authoritative for every control and value listed here; beats a stated "
                "claim or seed_weight for those items. It is partial: anything it doesn't "
                "list is not unknown -- read it from the attached screenshot. Labels are "
                "Logic's internal accessibility names, not the on-screen ones.\n\n"
                + ax_state
            ),
        })
    return system


def _build_messages(turns: list[dict]) -> list[dict]:
    messages = []
    for turn in turns:
        if "user" in turn:
            messages.append({"role": "user", "content": turn["user"]})
        elif "assistant" in turn:
            messages.append({"role": "assistant", "content": turn["assistant"]})
        else:
            raise ValueError(f"turn has neither 'user' nor 'assistant' key: {turn!r}")
    return messages


def _parse_verdict(raw_text: str) -> dict:
    """Robust-ish JSON parse: strip markdown fences if the model added them
    anyway, then try a straight json.loads, then fall back to the first
    {...} span in the text. Returns a dict with a "parse_error" key set (and
    the other fields defaulted to None) rather than raising, so one bad
    response doesn't crash a whole run."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                parsed = None

    if not isinstance(parsed, dict):
        return {
            "mode": None, "intent": None, "confidence": None,
            "why": None, "parse_error": f"could not parse JSON from: {raw_text[:300]!r}",
        }

    mode = parsed.get("mode")
    if mode not in VALID_MODES:
        return {
            "mode": None, "intent": parsed.get("intent"), "confidence": parsed.get("confidence"),
            "why": parsed.get("why"), "parse_error": f"invalid/missing mode: {mode!r}",
        }
    return {
        "mode": mode,
        "intent": parsed.get("intent") if parsed.get("intent") in VALID_INTENTS else parsed.get("intent"),
        "confidence": parsed.get("confidence") if parsed.get("confidence") in VALID_CONFIDENCE else parsed.get("confidence"),
        "why": parsed.get("why"),
        "parse_error": None,
    }


async def _classify_once(case: dict, attempt: int = 0) -> dict:
    system = _build_system(case.get("ax_state"))
    messages = _build_messages(case["turns"])
    try:
        resp = await _client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=messages,
        )
    except APIStatusError as e:
        if e.status_code == 429 and attempt < 4:
            backoff = 5.0 * (attempt + 1)
            log.warning("[anthropic_429] case=%r attempt=%d, backing off %.1fs", case["id"], attempt, backoff)
            await asyncio.sleep(backoff)
            return await _classify_once(case, attempt=attempt + 1)
        raise
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    verdict = _parse_verdict(text)
    verdict["raw_text"] = text
    return verdict


async def run(cases: list[dict], runs: int) -> dict:
    """Returns {case_id: {"case": ..., "runs": [verdict, ...]}}."""
    results: dict[str, dict] = {c["id"]: {"case": c, "runs": []} for c in cases}
    total_calls = len(cases) * runs
    call_i = 0
    for run_i in range(runs):
        for case in cases:
            call_i += 1
            print(f"  [{call_i}/{total_calls}] run {run_i + 1}/{runs} -- {case['id']}", flush=True)
            verdict = await _classify_once(case)
            results[case["id"]]["runs"].append(verdict)
            await asyncio.sleep(BETWEEN_CALLS_DELAY_S)
    return results


def _stable(runs: list[dict]) -> bool:
    modes = {r["mode"] for r in runs}
    return len(modes) == 1


def _print_case_table(results: dict, runs: int) -> None:
    print(f"\n{'=' * 120}\nPER-CASE RESULTS ({runs} run(s) each)")
    header = f"{'id':<38} {'bucket':<10} {'expected':<18} " + " ".join(f"run{i+1:<14}" for i in range(runs)) + " stable?"
    print(header)
    print("-" * len(header))
    for case_id, entry in results.items():
        case = entry["case"]
        got = [r["mode"] or f"ERR({r['parse_error'][:10]})" for r in entry["runs"]]
        stable = _stable(entry["runs"])
        row = f"{case_id:<38} {case['bucket']:<10} {case['expected_mode']:<18} "
        row += " ".join(f"{g:<15}" for g in got)
        row += "YES" if stable else "NO"
        print(row)


def _print_accuracy(results: dict) -> None:
    print(f"\n{'=' * 120}\nACCURACY")
    by_bucket: dict[str, list[bool]] = defaultdict(list)
    all_correct: list[bool] = []
    for entry in results.values():
        case = entry["case"]
        for r in entry["runs"]:
            correct = r["mode"] == case["expected_mode"]
            by_bucket[case["bucket"]].append(correct)
            all_correct.append(correct)
    for bucket in sorted(by_bucket):
        vals = by_bucket[bucket]
        n_correct = sum(vals)
        print(f"  {bucket:<12} {n_correct}/{len(vals)} ({100 * n_correct / len(vals):.1f}%)")
    n_correct = sum(all_correct)
    print(f"  {'OVERALL':<12} {n_correct}/{len(all_correct)} ({100 * n_correct / len(all_correct):.1f}%)")

    # Stability: fraction of cases whose mode was IDENTICAL across every run.
    stable_count = sum(1 for entry in results.values() if _stable(entry["runs"]))
    print(f"\n  Stable across all runs: {stable_count}/{len(results)} cases "
          f"({100 * stable_count / len(results):.1f}%)")


def _print_confusion(results: dict) -> None:
    print(f"\n{'=' * 120}\nCONFUSION (expected -> got, counted per individual run)")
    counts: Counter = Counter()
    for entry in results.values():
        case = entry["case"]
        for r in entry["runs"]:
            got = r["mode"] or "PARSE_ERROR"
            counts[(case["expected_mode"], got)] += 1
    for (expected, got), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        marker = "" if expected == got else "  <-- MISMATCH"
        print(f"  {expected:<18} -> {got:<18} : {n}{marker}")


def _print_wrong_or_unstable(results: dict) -> None:
    print(f"\n{'=' * 120}\nWRONG OR UNSTABLE CASES")
    any_flagged = False
    for case_id, entry in results.items():
        case = entry["case"]
        stable = _stable(entry["runs"])
        any_wrong = any(r["mode"] != case["expected_mode"] for r in entry["runs"])
        if stable and not any_wrong:
            continue
        any_flagged = True
        print(f"\n  {case_id} (bucket={case['bucket']}, expected={case['expected_mode']}, stable={stable})")
        print(f"    turns: {case['turns']}")
        for i, r in enumerate(entry["runs"], 1):
            flag = "OK" if r["mode"] == case["expected_mode"] else "WRONG"
            why = r.get("why") or r.get("parse_error") or "(no reason given)"
            print(f"    run {i}: [{flag}] got={r['mode']!r} confidence={r.get('confidence')!r} why={why!r}")
    if not any_flagged:
        print("  (none -- every case was correct and stable across all runs)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    cases = json.loads(CASES_FILE.read_text())
    print(f"Loaded {len(cases)} cases from {CASES_FILE}")
    print(f"Model: {MODEL}, runs: {args.runs}")

    start = time.monotonic()
    results = asyncio.run(run(cases, args.runs))
    elapsed = time.monotonic() - start
    print(f"\nDone in {elapsed:.1f}s ({len(cases) * args.runs} calls)")

    _print_case_table(results, args.runs)
    _print_accuracy(results)
    _print_confusion(results)
    _print_wrong_or_unstable(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{timestamp}.json"
    serializable = {
        case_id: {"case": entry["case"], "runs": entry["runs"]}
        for case_id, entry in results.items()
    }
    out_path.write_text(json.dumps({
        "timestamp": timestamp, "model": MODEL, "runs": args.runs,
        "results": serializable,
    }, indent=2))
    print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    main()
