"""v3 pipeline: query + context -> tool-calling loop -> response.

respond() takes the full running conversation (already ending with the newest
user turn) and returns the full conversation after this turn completes, so the
caller (the test harness) just appends the next user turn for the next call.
History is the only state — nothing is pinned server-side between turns.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Awaitable, Callable

from anthropic import AsyncAnthropic

from app import research, tools
from app.config import CENTRAL_ANTHROPIC_KEY, MAX_TOKENS, MODEL, RESEARCH_CALL_TIMEOUT_S, WRITER_MODEL
from app.prompt import SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT

OnChunk = Callable[[str], Awaitable[None]]
OnStatus = Callable[[str], Awaitable[None]]

log = logging.getLogger(__name__)

MAX_ITERATIONS = 6

# Decider-only request knobs, applied in _call_model to every call except the
# writer's; None omits the param. Module-level so run_latency_variants can swap
# them, along with MODEL, per variant. Effort "medium" (2026-09-24): vs the
# default "high", same graded quality over 3 full-battery passes, ~1s faster
# median, ~7% cheaper; "low" was no faster and failed across more scenarios.
DECIDER_EFFORT: str | None = "medium"
DECIDER_THINKING: dict | None = None

# End the turn right after an iteration whose tool calls were all successful
# card actions, skipping the decider's prose-only follow-up call (see the
# early-exit block in respond()). Direct actions only (2026-09-24): commands
# and "where is X" drop from ~5s to ~2.5-3s; diagnostic turns keep the call.
EARLY_EXIT_ON_ACTION = True

# Whether a "moderate" confidence tier also injects the "say you don't have a
# verified answer" instruction into the writer's facts (_finalize_answer).
# Off since 2026-09-12 together with the client badge -- see api.py's done
# payload for why. The tier itself is still computed.
HEDGE_MODERATE_TURNS = False

# tool_result handed to the decider in place of a web_research call the user
# said No to (resume="deny_research"). Code-guaranteed, like the lookup cap
# below: the model learns it can't research and must answer as unverified,
# rather than a prompt line asking it to.
_RESEARCH_DECLINED = {
    "error": (
        "The user declined web research for this turn. Do not call web_research "
        "again this turn. Answer from general Logic Pro knowledge if you can, "
        "framed clearly as unverified general guidance, not a confirmed answer -- "
        "or say plainly that you don't have a verified answer for this."
    )
}

# Hard, code-enforced cap on lookup_concept calls within one turn — a prompt
# instruction to "stop retrying" is a request, not a guarantee. Found live
# 2026-07-30: a genuinely absent topic ("wheres the monitor button") kept
# returning a *different* wrong match on every retried phrasing, so the model
# never hit a clean, repeated "nothing here" signal and just kept trying new
# wording until it hit the overall iteration ceiling with no answer at all.
# This intervenes explicitly before that happens, distinct from the overall
# MAX_ITERATIONS safety net below (which covers any tool, not just this one).
LOOKUP_ATTEMPT_LIMIT = 4
# What LOOKUP_ATTEMPT_LIMIT counts. Every lookup, as it always did: the
# 2026-09-21 "only unproductive ones" variant was measured on 09-22 and
# reverted -- an off-KB turn returned a DIFFERENT wrong bucket at moderate on
# every rephrasing, each one "productive" by that rule, so the cap never bound
# and the turn escalated to web_research (research_no_fire_* 3/3). The reason
# for the change is gone anyway: compound commands call action tools now and
# don't spend lookups. False counts only unproductive ones (kept for A/B).
COUNT_ALL_LOOKUPS = True

# First decider call on a turn that needs grounding (_needs_first_lookup):
# "any" = must call SOME tool, so a real question still can't be answered from
# memory with zero tool calls (the 2026-08-09 tool-skip gap), but a direct
# command can go straight to an action tool instead of paying a KB lookup for
# something that isn't in the KB. True restores forcing lookup_concept itself
# (the pre-2026-09-21 behaviour; kept for the harness's before/after).
FORCE_FIRST_LOOKUP = False

# One card per turn (2026-09-21). Every action -- called directly by the model
# or queued because a lookup landed on a solution that maps to it -- is a step
# on the same card, in call order. What commit-to-one was for stays: acting
# on a second candidate from the SAME problem bucket is an alternative fix,
# not a second request (2026-08-05: the last candidate silently won), and is
# refused. The cap bounds a confused turn, not a real one.
MAX_ATTACHES_PER_TURN = 6
_ATTACH_TOOLS = tools.ACTION_TOOLS

# A lookup that lands on ONE solution with a strong match queues that
# solution's action itself (2026-09-21): deciding whether to also attach was
# the single biggest failure point -- the model looked the answer up, said it,
# and never made the second call (09-21 baseline: most walkthrough misses).
# Weaker or multi-candidate results leave the choice to the model, which then
# has to act on one or ask (see _needs_pick).
# Moderate single matches get pick-or-ask too (not auto-attach): e.g. "how do
# i change the recording bit depth" lands on 'recording settings' at moderate,
# and was a 09-21 walkthrough miss. The confidence bands (0.40/0.60) were
# tuned at 8 problems / 21 solutions and never recalibrated, so a correct hit
# often reads moderate; a moderate COLLISION ("frame rate" -> sample rate) is
# why this asks the model to act, ask or look again rather than attaching.
PICK_ON_MODERATE_SINGLE = True

_ON_CARD_NOTE = (
    "This solution's action is already on the reply's card -- it waits for the user "
    "to press Run. Don't call it again, and don't describe it as done."
)
_PANE_ONLY_NOTE = (
    " It opens the setting's pane only, with no value chosen. If your answer recommends a "
    "value it can take, call open_setting for the same route with that value -- that "
    "replaces the queued step rather than adding one."
)
_PICK_NUDGE = (
    "A lookup this turn returned solutions that map to an action. If your answer "
    "recommends one, call its action so the user gets it on the card; if you can't tell "
    "which applies, ask the one question that would decide it (ask_clarifying_question); "
    "if the match isn't actually what they asked about, look up something more specific. "
    "If the candidate you recommend has no action of its own, call lookup_concept with its "
    "name -- don't queue a different candidate's action instead."
)

_ATTACH_CAP_REFUSAL = (
    f"this turn already has {MAX_ATTACHES_PER_TURN} steps queued, the limit for one "
    "card. Tell the user what's queued and that you'll do the rest when they ask next."
)
_DUPLICATE_ATTACH_REFUSAL = "this exact step is already queued this turn -- it runs once."

# Whether the card may run without the user pressing Run (client auto-run
# setting permitting). Decided in code, never by the model: an instruction
# naming what to do ("put a compressor on this", "can you add an eq") may
# auto-run; anything question-shaped ("how do I…", "what does…", "channel
# eq?", "explain … and add one"), broad in scope ("every track", "remove
# all"), or inferred from a KB lookup rather than asked for (see respond())
# waits for Run.
_QUESTION_START = re.compile(
    r"^(?:how|what|whats|why|where|wheres|when|which|who|is|are|does|did|should|"
    r"do (?:i|you|we)|can (?:i|we)|could (?:i|we)|explain|show me how|tell me)\b"
)
_QUESTION_ANYWHERE = re.compile(r"\b(?:how (?:do|can|would|should) (?:i|we|you)|how to|show me how|explain)\b")
_POLITE_REQUEST = re.compile(
    r"^(?:please\s+)?(?:can|could|would|will) you\b(?!\s+(?:explain|tell|show me how|describe|help me understand))"
)
_BULK_SIGNALS = (
    "every track", "all tracks", "all the tracks", "all my tracks", "every plugin", "all plugins",
    "all the plugins", "remove all", "delete all", "clear all", "reset all", "whole session",
    "entire session", "whole project", "entire project",
)


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].lower().replace("\u2019", "'").replace("'", "").strip()
    return ""


def _offered_text(messages: list[dict]) -> str:
    """The reply the newest user message answers -- see tools.TURN_OFFERED_TEXT.
    Options are often offered as a statement ("you can choose between X or Y"),
    so any reply counts; a first turn has none, so the model still can't pick
    a value there that the user didn't name."""
    i = _last_user_text_index(messages)
    for m in reversed(messages[:i] if i is not None else []):
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else " ".join(
            b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text") if isinstance(c, list) else ""
        if text.strip():
            return text.lower()
    return ""


def _is_question(text: str) -> bool:
    if _POLITE_REQUEST.match(text):
        return bool(_QUESTION_ANYWHERE.search(text))
    return text.endswith("?") or bool(_QUESTION_START.match(text) or _QUESTION_ANYWHERE.search(text))


def _auto_run_ok(messages: list[dict], card: list | None) -> bool:
    if not card:
        return False
    text = _last_user_text(messages)
    return bool(text) and not _is_question(text) and not any(sig in text for sig in _BULK_SIGNALS)


def _lookup_is_unproductive(output: dict, buckets_seen: set) -> bool:
    """See COUNT_ALL_LOOKUPS. A capped/errored call isn't a lookup result at all."""
    if "error" in output:
        return False
    if output.get("match") == "none":
        return True
    if str(output.get("match_confidence", "")).startswith("weak"):
        return True
    return output.get("problem") in buckets_seen


def _attach_key(name: str, inp: dict) -> str:
    return json.dumps([name, inp], sort_keys=True, default=str)


def _endorse_key(name: str, inp: dict) -> str:
    """What makes two calls 'the same action' for endorsement: the target, not
    every argument. A lookup queues open_plugin{plugin} and the model then calls
    open_plugin{plugin, track} -- that's the model agreeing with the queued
    step, not a second request, so the model's version replaces it and the card
    counts as model-called (it may auto-run)."""
    if name == "open_plugin":
        return f"open_plugin:{str(inp.get('plugin', '')).strip().lower()}"
    if name == "set_param":
        return f"set_param:{str(inp.get('plugin', '')).strip().lower()}:{str(inp.get('param', '')).strip().lower()}"
    if name == "open_setting":
        return f"open_setting:{str(inp.get('name', '')).strip().lower()}"
    return _attach_key(name, inp)


def _cand_key(name: str, inp: dict) -> str:
    """Which candidate a call is, for commit-to-one: an open_setting call is its
    route whatever value it picks (a lookup's candidate never carries one)."""
    if name == "open_setting":
        return _attach_key(name, {"name": inp.get("name")})
    return _attach_key(name, inp)


def _bucket_candidates(trace: list[dict]) -> dict[str, str]:
    """attach key of each candidate's action -> its problem bucket, for every
    multi-candidate ('problem') lookup this turn, plus moderate single matches
    (PICK_ON_MODERATE_SINGLE). What tells a second pick from the same bucket
    (an alternative, refused) apart from a separate request, and what
    _needs_pick checks."""
    out: dict[str, str] = {}
    for c in trace:
        o = c["output"] if c["tool"] == "lookup_concept" else {}
        conf = str(o.get("match_confidence", ""))
        moderate_single = (
            PICK_ON_MODERATE_SINGLE and o.get("match") == "single" and conf.startswith("moderate")
        )
        if o.get("match") != "problem" and not moderate_single:
            continue
        # A weak match is "nothing here really" (_lookup_is_unproductive says so
        # for the cap too): forcing a tool call on it pressures the model into
        # queuing whatever it half-matched, when the honest answer is often no
        # verified answer -- or web_research, which it can still choose freely.
        if conf.startswith("weak"):
            continue
        for sol in o.get("solutions") or []:
            for tool, inp in tools.action_calls(sol.get("action")):
                out.setdefault(_cand_key(tool, inp), o.get("problem"))
    return out


def _is_alternative_pick(name: str, inp: dict, trace: list[dict]) -> bool:
    """True if this action is a candidate from a bucket another of whose
    candidates is already on the card -- a fallback, not a second request."""
    cands = _bucket_candidates(trace)
    key = _cand_key(name, inp)
    problem = cands.get(key)
    if problem is None:
        return False
    return any(
        c["tool"] in _ATTACH_TOOLS and c["output"].get("attached")
        and cands.get(_cand_key(c["tool"], c["input"])) == problem
        and _cand_key(c["tool"], c["input"]) != key
        for c in trace
    )


def _replace_span(card_steps: list, start: int, end: int, new: list, *span_maps: dict) -> None:
    """Swap card_steps[start:end] for `new` and shift every recorded span after it."""
    card_steps[start:end] = new
    shift = len(new) - (end - start)
    if shift:
        for spans in span_maps:
            for k, (a, b) in list(spans.items()):
                if a >= end:
                    spans[k] = (a + shift, b + shift)


def _mark_replaced(trace: list[dict], name: str) -> None:
    """The earlier open_setting call(s) for this route no longer describe the
    card -- the writer (card_descriptions) and the grader stop counting them."""
    for c in trace:
        if c["tool"] == "open_setting" and c["input"].get("name") == name and c["output"].get("attached"):
            c["output"] = {**c["output"], "attached": False, "replaced": True}


def _needs_pick(trace: list[dict], card: list) -> bool:
    """A bucket lookup returned actionable candidates and the turn is about to
    end with nothing on the card and no clarifying question."""
    if card or any(c["tool"] == "ask_clarifying_question" for c in trace):
        return False
    return bool(_bucket_candidates(trace))

_client = AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY)


async def respond(messages: list[dict], **kwargs) -> "Result":
    """The turn loop (_respond), plus whether its card may auto-run -- decided
    here, once, from the user's own message for every return path."""
    result = await _respond(messages, **kwargs)
    # Any turn that consulted the KB waits for Run, whoever called the action:
    # the fix was inferred from a diagnosis, not asked for. Found live
    # 2026-09-22 (research_no_fire_generic_troubleshooting_miss): on a turn
    # whose own answer was "I don't have a verified fix", the model queued
    # "bypass control surfaces" off a moderate match and the card was cleared
    # to run by itself. Only actions the user actually asked for -- a direct
    # command with no lookup behind it -- may auto-run.
    inferred = result.card_from_lookup or any(c["tool"] == "lookup_concept" for c in result.trace)
    # Routes whose target depends on the user's selection (`wait_for_run`) always
    # wait, so the reply's "select it first" comes before anything runs.
    waits = any(c["tool"] == "open_setting" and tools.ROUTES.get((c.get("input") or {}).get("name"), {}).get("wait_for_run")
                for c in result.trace)
    result.auto_run = (not inferred) and not waits and _auto_run_ok(messages, result.walkthrough_steps)
    return result


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} seconds"
    minutes = seconds / 60
    if minutes == int(minutes):
        return f"{int(minutes)} minute{'s' if minutes != 1 else ''}"
    return f"{minutes:.1f} minutes"


# Status shown in the still-empty assistant bubble while a tool call this
# turn is in flight -- distinguishes the two real wait states client-v3 can't
# otherwise tell apart (a fast KB vector search vs. a nested Sonnet+
# web_search call, see app/research.py). Priority order below applies when an
# iteration's tool_uses contains more than one name; ask_clarifying_question
# is deliberately absent -- it ends the turn immediately, no wait to narrate.
# get_walkthrough is also absent -- an in-memory dict attach with no real I/O,
# found live 2026-09-04 to flash past too fast to read; leaving it unmapped
# means "Thinking..." (already showing from this iteration's top) just carries
# through the attach uninterrupted instead of a message appearing only to
# immediately vanish.
_TOOL_STATUS_PRIORITY = ["web_research", "lookup_concept"]


def _status_for_tools(tool_uses: list) -> str | None:
    names = {tu.name for tu in tool_uses}
    for name in _TOOL_STATUS_PRIORITY:
        if name not in names:
            continue
        if name == "web_research":
            return "Searching the web for a verified answer…"
        if name == "lookup_concept":
            return "Searching internal knowledge base…"
    return None

async def _ack_clarifying_question() -> dict:
    """No-op executor for tools.ASK_CLARIFYING_QUESTION_SCHEMA -- the tool call
    itself is the entire point (a hard, code-checkable signal that this turn
    makes no claim), nothing to actually do. Still needs a real tool_result
    round-trip like any other tool call, or the next turn's history replay
    would 400 (see _serialize_content's docstring on the same requirement)."""
    return {"acknowledged": True}


async def _now(result: dict) -> dict:
    """Wraps a synchronous executor result for the awaited _EXECUTORS call."""
    return result


_EXECUTORS = {
    "open_plugin": lambda inp, fixture: _now(tools.queue_open_plugin(inp)),
    "set_param": lambda inp, fixture: _now(tools.queue_set_param(inp)),
    "lookup_concept": lambda inp, fixture: tools.lookup_concept(inp["problem"]),
    "open_setting": lambda inp, fixture: _now(tools.queue_open_setting(inp, fixture)),
    "ask_clarifying_question": lambda inp, fixture: _ack_clarifying_question(),
    "web_research": lambda inp, fixture: research.web_research(inp["query"]),
}


@dataclass
class Result:
    text: str
    walkthrough_steps: list | None
    trace: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)
    # "strong" / "moderate" / "research" / "generic" (see _confidence_tier).
    # Drives the client's confidence badge; replaces the old literal
    # _HEDGE_PREFIX text that used to carry this signal inline.
    confidence_tier: str = "generic"
    # Set (and everything else left at its empty default) when a
    # research_confirm=True request reached a web_research call the user
    # hasn't approved yet: the turn is parked, `messages` carries the
    # transcript so far (ending in the pending tool_use) for the client to
    # hand back with resume="allow_research"/"deny_research". See respond().
    pending_research_query: str | None = None
    # Citations from this turn's web_research call(s), if any -- empty for
    # every other turn. Rendered as source chips client-side (message_widget.py
    # set_sources, built ahead of the tool itself on 2026-09-04).
    sources: list[dict] = field(default_factory=list)
    # True if this turn's card may run without the user pressing Run (the
    # client's own auto-run setting still has to be on). See _auto_run_ok.
    auto_run: bool = False
    # Some card step was queued by a lookup rather than called directly by the
    # model; such a card never auto-runs (see respond()).
    card_from_lookup: bool = False


# Bare "again" and "forgot" deliberately excluded, and "cant find it" removed
# after initially being added -- found via two rounds of Fable review,
# 2026-08-06. "again" collides with ordinary closing phrasing ("thanks again,
# that fixed it!"); "forgot" is the identical shape ("oh forgot to say -- that
# fixed it, thanks!"); "cant find it" (added when apostrophes were first
# stripped from both sides of the match, so "can't"/"cant" would both match)
# turned out to appear in 7 different battery turns once actually swept
# against the full battery, not just the one it was meant for -- a live
# comment here once claimed "only this gap's scenario matches" before that
# sweep was actually run; that was the exact circular-validation mistake this
# feature has repeatedly made. Every phrase kept here is a specific, low-
# frequency request to see something again, not a common word or filler.
_REASK_SIGNALS = (
    "remind me", "reminder", "one more time", "show me that",
    "show that again", "where was", "where is that", "closed the window",
    "lost the window",
)

# Plain acknowledgments/closers -- carved out of the iteration-0 forced
# lookup_concept call (see _needs_first_lookup below) so a "thanks!" turn
# doesn't force the model to invent a fake problem query just to satisfy a
# tool call it has nothing real to make. Exact substring match, same
# deliberately simple approach as _REASK_SIGNALS above -- hand-curated rather
# than pulled from an existing list (checked: Rasa's built-in "thankyou"
# intent is voice-transcript ASR training data full of filler noise like "uh
# thank you good bye", not a clean fit for exact matching). A typo'd close
# ("tahnks") just eats one wasted lookup call, not worth fuzzy-matching for.
_CLOSING_SIGNALS = (
    "thanks", "thank you", "thx", "ty",
    "got it", "sounds good", "cool", "perfect", "great", "awesome",
    "ok", "okay", "no more questions", "that's all", "im good", "i'm good",
    "bye", "goodbye", "see ya",
)

# Genuinely irreversible whole-project requests -- the prompt's own "Fail
# closed on execution" rule already says decline these outright, no
# walkthrough, but forcing a lookup_concept call on iteration 0 anyway
# (see _needs_first_lookup below) meant that rule never got a clean chance to
# apply: the forced call populated trace, and _needs_hedge (this function's
# pre-badge name; now _confidence_tier) then hedged a response that was
# actually a decline, not a claim
# -- found live 2026-09-02 via Fable review of the writer-split battery
# (destructive_probe: "Note: I couldn't verify this... so treat the
# following as general guidance" prepended to a safety refusal). Deliberately
# narrow and destructive-shaped (verb + whole-project scope), same curation
# discipline as _REASK_SIGNALS/_CLOSING_SIGNALS above -- NOT a bare "delete",
# which would wrongly carve out a legitimate "how do i delete a track"
# navigation question. "delete/erase everything" also deliberately excluded
# on its own (no bare form) -- found via direct testing 2026-09-03: matches
# "delete everything on this track" as a false positive, a single-track,
# undo-recoverable request, not the whole-project case this exists for;
# kept only project-scoped.
_IRREVERSIBLE_SIGNALS = (
    "delete my entire project", "delete the entire project",
    "delete my whole project", "delete the whole project",
    "erase my entire project", "erase the entire project",
    "erase my whole project", "erase the whole project",
    "delete everything in my project", "delete everything in the project",
    "erase everything in my project", "erase everything in the project",
    "wipe my entire project", "wipe the entire project",
)


# Bare greetings -- carved out of the iteration-0 forced lookup_concept call
# for the same reason as _CLOSING_SIGNALS: nothing real to look up on "hi",
# forcing one just makes the model invent a fake problem query. Found live
# 2026-09-04: no carve-out existed for openers at all, only closers, so every
# greeting forced a real KB search ("Searching internal knowledge base...").
# Deliberately matched as the WHOLE message (see _is_bare_greeting) rather
# than _last_user_message_matches's substring-anywhere check used for the
# other three signal lists -- "hi"/"yo" are common substrings of ordinary
# words ("this", "history", "yoke"), so containment matching here would
# wrongly skip the forced lookup on real questions.
_GREETING_SIGNALS = (
    "hi", "hey", "hello", "yo", "sup", "hiya", "howdy", "greetings",
    "hey there", "hi there", "what's up", "whats up", "good morning",
    "good afternoon", "good evening",
)


def _is_bare_greeting(messages: list[dict]) -> bool:
    """True if the most recent message is a user turn whose ENTIRE text
    (punctuation-stripped) is one of _GREETING_SIGNALS -- see that list's
    comment for why this can't reuse _last_user_message_matches's substring
    check."""
    if not messages or messages[-1].get("role") != "user":
        return False
    user_text = messages[-1].get("content")
    if not isinstance(user_text, str):
        return False
    normalized = user_text.lower().strip(" !.?").replace("'", "")
    return normalized in _GREETING_SIGNALS


def _last_user_message_matches(messages: list[dict], signals: tuple[str, ...]) -> bool:
    """True if the most recent message is a user turn whose text contains any of
    the given signal phrases (case/apostrophe-insensitive substring match)."""
    if not messages or messages[-1].get("role") != "user":
        return False
    user_text = messages[-1].get("content")
    if not isinstance(user_text, str):
        return False
    normalized = user_text.lower().replace("'", "")
    return any(sig in normalized for sig in signals)


def _needs_first_lookup(messages: list[dict]) -> bool:
    """Tool-skip gap fix: without this, the model is free to answer a real
    question with zero lookup_concept calls, and _confidence_tier() (formerly
    _needs_hedge()) deliberately falls through to "generic" on an empty trace
    (can't tell that apart from a normal closing turn) -- so those answers ship
    with full confidence and no KB grounding at all (found live 2026-08-09, the
    "how do i make a beat" query). Fix is to force lookup_concept on the turn's
    first iteration instead of leaving it to the model's judgment -- same
    "code, not a prompt request" lesson as _confidence_tier itself.

    Carved out for four different reasons, not one:
    - _REASK_SIGNALS: forcing a fresh lookup here would starve
      _backfill_walkthrough, which only fires on a turn with zero tool calls.
    - _CLOSING_SIGNALS: nothing real to look up on a plain "thanks"/"ok" turn;
      forcing one just makes the model invent a query to satisfy the tool.
    - _IRREVERSIBLE_SIGNALS: a genuinely irreversible request should be
      declined outright with no tool call at all -- forcing a lookup here
      populates trace for no reason and causes _confidence_tier to mark
      "moderate" what should be a plain decline (found live 2026-09-02, destructive_probe).
    - _GREETING_SIGNALS: same reasoning as _CLOSING_SIGNALS, opener instead
      of closer.
    """
    if _last_user_message_matches(messages, _REASK_SIGNALS):
        return False
    if _last_user_message_matches(messages, _CLOSING_SIGNALS):
        return False
    if _last_user_message_matches(messages, _IRREVERSIBLE_SIGNALS):
        return False
    if _is_bare_greeting(messages):
        return False
    return True


def _backfill_walkthrough(messages: list[dict], text: str) -> dict | None:
    """Commit-to-one is scoped per turn, so a turn that attaches nothing normally
    leaves the user with prose only -- correct when the model is just answering a
    new question, but wrong when the user asked to *see* a destination again and
    the model answered from memory instead of re-queuing the route (found
    live 2026-08-05, multiturn_same_destination_revisited_new_turn: inconsistent,
    sometimes recalls the tool, sometimes doesn't). Deterministic backfill instead
    of another prompt instruction -- prompt-only attempts at behaviors like this
    have repeatedly not held (hedge-on-navigation, toggle-gate force flag).

    Gated on the CURRENT user turn's own message containing explicit re-ask
    language (_REASK_SIGNALS), not on the model's response text alone. First
    version matched purely on the destination name appearing anywhere in the
    response; found live (2026-08-06) that's unreliable -- a turn can legitimately
    discuss the same general topic (e.g. "sample rate") for an unrelated reason
    (pointing the user at a different, non-Logic screen) without ever being asked
    to reattach anything, and the destination's own name is exactly the word most
    likely to recur regardless of intent. Requiring zero tool calls (this turn did
    no fresh work at all -- pure memory recall) plus explicit re-ask language in
    what the user actually typed is a direct signal of intent instead of an
    inference from response-content overlap.

    Checking this against the battery's own scenario texts is a weak signal, not
    real validation -- it only proves the gate doesn't misfire on the handful of
    conversations it was tuned against, not on real phrasing in general (this was
    stated here once before without actually re-running the sweep after a signal
    list change, and turned out to be false at the time -- re-verify by sweeping
    the battery, don't just trust this comment, if this list changes again). Real
    coverage of hostile phrasing (e.g. "thanks again, that fixed it" on a closing
    turn) lives in test_backfill.py's unit tests instead, which don't depend on
    guessing what a live model happens to say.

    Scans prior turns' message history (not this turn's own -- there's nothing to
    find there if this turn made no tool calls) for successful route
    attaches, identified by result shape (attached+destination+steps together, not
    by tool name -- avoids needing to correlate tool_use_id across blocks). If the
    re-ask check passes, reattaches the one destination whose name appears in this
    turn's response text (now just disambiguating *which* prior destination among
    possibly several, not deciding *whether* to backfill at all); zero or multiple
    matches are left alone rather than guessed.
    """
    if not _last_user_message_matches(messages, _REASK_SIGNALS):
        return None

    attached_by_destination: dict[str, list] = {}
    for msg in messages:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            try:
                result = json.loads(block["content"])
            except (TypeError, ValueError):
                continue
            for r in [result] + [o.get("output") or {} for o in result.get("on_card") or []]:
                # on_card: an action a lookup queued itself, the only place it is
                # recorded (see the auto-attach block in _respond).
                if r.get("attached") and r.get("destination") and r.get("steps"):
                    attached_by_destination[r["destination"]] = r["steps"]

    matches = [dest for dest in attached_by_destination if dest.lower() in text.lower()]
    if len(matches) != 1:
        return None
    return {"destination": matches[0], "steps": attached_by_destination[matches[0]]}



def _attach_is_strong(trace: list[dict], attached_solution: str | None) -> bool:
    """True if the get_walkthrough attach that set walkthrough_steps was itself
    backed by a strong-confidence lookup_concept match for that exact solution --
    confidence in a DESTINATION existing is not the same as confidence in the MATCH
    that picked it. Found live 2026-08-20/21 (collision_software_monitoring): a
    moderate-confidence match ("audio settings") attached a real, path-bearing
    destination and shipped fully unhedged, because the old check only asked
    whether *anything* attached, never at what confidence. A backfilled attach
    (2026-08-05 same-destination-revisit fix) has no lookup call this turn to check
    against -- treated as strong, since it's reattaching an earlier turn's
    already-resolved destination, not a fresh guess.

    Checks EVERY lookup_concept call that surfaced this solution, not just the
    first -- found live 2026-08-21 (ax_contradicts_user_claim): the model's first
    query landed moderate, a second, more specific query for the same solution
    landed strong, and the walkthrough correctly attached off the strong hit, but
    an earlier version of this function returned on the first (moderate) mention
    and hedged a response that was actually fully grounded."""
    if attached_solution is None:
        return True
    confidences = [
        c["output"].get("match_confidence")
        for c in trace
        if c["tool"] == "lookup_concept"
        for sol in c["output"].get("solutions") or []
        if sol.get("name") == attached_solution
    ]
    if not confidences:
        return True
    return "strong" in confidences


def _moderate_grounded_lookup_exists(trace: list[dict]) -> bool:
    """True if some lookup_concept call this turn hit at least 'moderate'
    match_confidence. A 'weak' top hit is explicitly documented (tools.py's
    _confidence_label) as "likely not actually relevant, don't treat this as
    a real match" -- collapsing it into the same "moderate" UI badge as a
    genuine partial match misrepresents a nonsensical/out-of-KB query (e.g.
    "ww") as "confirmed against the KB, just not confidently." Found live
    2026-09-04.

    startswith, not equality: _confidence_label's return value is the
    model-facing advisory sentence, not a clean tag -- only "strong" comes
    back bare with no suffix ("moderate ..."/"weak ..." both carry trailing
    advisory text, see tools.py)."""
    return any(
        c["tool"] == "lookup_concept"
        and str(c["output"].get("match_confidence", "")).startswith(("strong", "moderate"))
        for c in trace
    )


def _strong_grounded_lookup_exists(trace: list[dict]) -> bool:
    """True if some lookup_concept call this turn hit strong confidence AND
    surfaced at least one solution with real fix content (has_path). A correct,
    strong-confidence DIAGNOSIS whose every candidate solution has has_path=False
    is not grounding for a FIX -- there's nothing there to be confident about
    beyond the label itself. Found live 2026-08-20/21 (multiturn_clarify_then_
    resolve_thin_hollow): 'song sounds thin or hollow' matched at strong
    confidence, but every one of its solutions is diagnosis-only -- the model then
    fabricated an entire fix procedure from outside the KB, unhedged, because the
    old check only asked whether *some* lookup hit strong, never whether that
    match actually carried anything to be confident about."""
    return any(
        c["tool"] == "lookup_concept"
        and c["output"].get("match_confidence") == "strong"
        and any(s.get("action") for s in c["output"].get("solutions") or [])
        for c in trace
    )


def _successful_research_exists(trace: list[dict]) -> bool:
    """True if some web_research call this turn actually returned findings --
    not just was called. A timed-out/errored call (see app/research.py) or one
    that genuinely found nothing has no real content behind it and must fall
    through to the ordinary strong/moderate grounding check below, same as any
    other unsuccessful tool attempt -- "research was attempted" isn't the same
    claim as "research grounds this answer." Found live 2026-09-04: the
    research-battery's Mac-Studio-chip scenario hit web_research's own 90s
    timeout, and correctly still needs the ordinary hedge path, not a false
    "research" badge on an empty result."""
    return any(
        c["tool"] == "web_research" and not c["output"].get("error") and c["output"].get("findings")
        for c in trace
    )


def _confidence_tier(
    trace: list[dict],
    walkthrough_steps: list | None,
    attached_solution: str | None = None,
    is_clarifying_question: bool = False,
) -> str:
    """Deterministic backstop -- prompt-only hedge instructions held ~40-50% of the time
    across live testing, 0% on broad/indirect questions specifically (2026-08-11).
    Returns "moderate" exactly when this turn made a real attempt (a tool call) that hit
    at least a genuine 'moderate' lookup_concept match (see _moderate_grounded_lookup_exists)
    but never reached strong grounding -- a lookup whose only hit was 'weak' (tools.py:
    "likely not actually relevant") is treated the same as no real attempt, not a
    confirmed-but-uncertain one. Never "moderate" on zero tool calls -- can't
    distinguish a plain conversational close from the separate, not-yet-fixed
    tool-skip gap, so that case (and a claim-free clarifying question) falls
    through to "generic" instead.

    Was a bool (_needs_hedge) driving a guaranteed literal text prefix
    (_HEDGE_PREFIX) prepended to the response; replaced 2026-09-04 with this
    label driving a client-side confidence badge instead (v3-log.md). Same
    underlying condition, same reliability property that mattered originally
    (deterministic, code-computed, not dependent on the model choosing to phrase
    a caveat) -- a server-computed badge carries that guarantee at least as well
    as a forced text prefix did, without the prefix's "shows up verbatim at the
    top of every moderate answer" cost.

    "research" (added 2026-09-04, same day web_research shipped) is trusted the
    same way "strong" is -- a real citation-backed finding is not the same kind
    of claim as an unverified guess, and shouldn't get the "no verified answer"
    hedge facts _finalize_answer injects for "moderate" (confirmed live: before
    this branch existed, a fully cited Skrillex/FM8 answer and a real Mac-Studio-
    chip finding both opened with "I don't have a verified answer" anyway,
    because this function had no idea web_research had even run). Checked after
    the walkthrough branch, not before -- a turn that both attaches a real
    destination AND separately researched something else (see
    research_mixed_kb_and_research) keeps grading off the attach, its own
    already-correct primary claim, not the secondary research aside; sources
    still reach the client via Result.sources regardless of which tier wins.

    is_clarifying_question short-circuits ahead of all of that -- a turn that's
    ONLY a clarifying question (see tools.ASK_CLARIFYING_QUESTION_SCHEMA) makes no
    claim at all, so there's nothing to badge as uncertain regardless of what the
    underlying trace looks like. Found live 2026-09-02 (Fable review):
    muddy_ambiguous and thin_hollow_boundary are trace-identical to a real
    ungrounded claim (a tool call happened, nothing attached, confidence never
    hit strong) but the response is just a question.

    Known remaining gap, not fixed here: a strong, path-bearing match still only
    certifies the DESTINATION, not every specific claim layered on top of it in
    prose (e.g. what a control does, where it sits, what values a dropdown offers
    beyond what the tool result actually said) -- found live 2026-08-20/21
    (monitor_button_already_on, multiturn_evidence_arrives_later_turn). Catching
    that needs claim-level grounding, not a trace-level check; left to the
    widened prompt-level grounding rule instead. Same caveat now applies to
    "research" -- a cited finding grounds what the sources actually said, not
    every detail the model adds while translating it into a Logic Pro answer."""
    if is_clarifying_question:
        return "generic"
    if walkthrough_steps is not None:
        return "strong" if _attach_is_strong(trace, attached_solution) else "moderate"
    if _successful_research_exists(trace):
        return "research"
    if not trace:
        return "generic"
    if _strong_grounded_lookup_exists(trace):
        return "strong"
    return "moderate" if _moderate_grounded_lookup_exists(trace) else "generic"


def _collect_sources(trace: list[dict]) -> list[dict]:
    """Every source cited by a successful web_research call this turn, in call
    order, de-duplicated by URL. Empty for a turn that never called
    web_research or whose call(s) all errored/found nothing -- mirrors
    _successful_research_exists' own definition of "successful" rather than
    trusting whatever a failed call's output happens to contain."""
    sources: list[dict] = []
    seen: set[str] = set()
    for c in trace:
        if c["tool"] != "web_research" or c["output"].get("error"):
            continue
        for src in c["output"].get("sources") or []:
            url = src.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append(src)
    return sources


_FALLBACK_REFUSAL = (
    "another candidate for this same problem is already on the card this turn -- "
    "this is an alternative fix, not a second request. Mention it in prose as the "
    "next thing to try, and only queue it if the user comes back and says the "
    "first one didn't work."
)

def _plain_history(messages: list[dict]) -> list[dict]:
    """Full decider transcript -> plain user/assistant text turns only, for the
    writer call (see _write_response). The writer never sees tool_use/tool_result
    blocks -- those are exactly the vocabulary this split exists to keep out of its
    context (v3-log.md 2026-08-22/24). Drops a turn entirely if it carries no text
    (a pure tool-call assistant turn, or a pure tool-result user turn) rather than
    passing an empty message the API would reject."""
    plain = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                plain.append({"role": msg["role"], "content": content})
            continue
        if isinstance(content, list):
            text = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if text:
                plain.append({"role": msg["role"], "content": text})
    return plain


async def _write_response(
    facts: str,
    prior_messages: list[dict],
    on_chunk: OnChunk | None,
) -> tuple[str, dict]:
    """The 'writer' call -- rephrases the decider's already-decided, already-
    correct synthesis into plain-language prose with none of the decider's tool
    vocabulary, prototyped standalone 2026-08-22/24 (0/9 jargon leaks across 3
    case shapes). Deliberately a different, cheaper model (WRITER_MODEL) and a
    minimal prompt that never mentions tools/results/confidence at all -- the
    decider (Sonnet) keeps sole ownership of *deciding* (tool loop, evidence
    weighing, the hedge/attach guards above); this call only ever *phrases* what
    was already decided, via the facts block below, never reasons over raw trace
    data itself.
    """
    system_text = (
        WRITER_SYSTEM_PROMPT + "\n\n## What to say this turn\n\n" + facts
    )
    system = [{"type": "text", "text": system_text}]
    history = _plain_history(prior_messages)
    resp = await _call_model(system, None, history, on_chunk, model=WRITER_MODEL)
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, _usage_dict(resp.usage)


def card_descriptions(trace: list[dict]) -> list[str]:
    """One plain line per queued step, from the calls themselves -- what the
    writer is told the card holds."""
    out = []
    for c in trace:
        if c["tool"] not in _ATTACH_TOOLS or not c["output"].get("attached"):
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
                line += " (opens the pane only -- no value is chosen; the user picks there)"
            out.append(line)
    return out


async def _finalize_answer(
    trace: list[dict],
    walkthrough_steps: list | None,
    text: str,
    prior_messages: list[dict],
    on_chunk: OnChunk | None,
    usage: list[dict],
    attached_solution: str | None = None,
    is_clarifying_question: bool = False,
    live_state: str | None = None,
    had_screenshots: bool = False,
    has_actions: bool = False,
) -> tuple[str, str]:
    """Called once this turn's decider text is fully known (either the
    no-more-tool-calls return or the MAX_ITERATIONS safety net). Tier decision
    stays exactly as before -- deterministic, code-enforced (_confidence_tier) --
    but as of 2026-09-04 no longer prepends a literal _HEDGE_PREFIX to the
    streamed/returned text; the client renders the tier as a confidence badge
    instead (source_tier in the /v3/chat "done" payload -- see api.py). The
    badge is at least as reliable a carrier of this signal as the old forced
    text prefix was (still server-computed, still not dependent on the model
    choosing to phrase a caveat), without the prefix's cost of showing up
    verbatim at the top of every moderate-confidence answer.

    The writer still gets told plainly, in its facts input, when a turn has no
    verified answer -- so its own prose doesn't overclaim a specific unverified
    path as fact -- but nothing about that instruction is load-bearing for the
    confidence signal reaching the user anymore; that's the badge's job now.
    Returns (response_text, confidence_tier)."""
    tier = _confidence_tier(trace, walkthrough_steps, attached_solution, is_clarifying_question)
    facts = text
    # Queued actions run on the CLIENT, after this reply, when the user presses
    # Run. Found live 2026-09-18: the writer said "Loaded ValhallaSupermassive
    # onto Audio 2" before anything ran, and the next turn treated its own
    # claim as fact. Code-guaranteed framing, not a prompt hope. (Phase 2 of
    # the typed-tools plan replaces this with a sentence built in code.)
    if has_actions:
        # Built in code, not taken from the decider's prose: on a card a lookup
        # queued, the decider learned of the step only after the fact and may
        # never mention it, leaving the writer to invent a description of a card
        # the user can see (Fable review, 2026-09-21).
        queued = "\n".join(f"- {line}" for line in card_descriptions(trace))
        facts = (
            "The reply has a card attached, holding exactly these steps, which the user "
            "runs with one press:\n" + queued + "\n\nNONE of them has happened yet. Say in "
            "one short sentence what the card will do, as about to happen (\"This adds X to "
            "the selected track\" / \"This opens Y\"), never as done, and don't list manual "
            "steps for it -- the card does it. Don't mention anything not listed above.\n\n"
            + text
        )
    # Only "moderate" gets the hedge facts injected below -- "research" is
    # trusted the same as "strong" (see _confidence_tier's 2026-09-04 note) and
    # deliberately falls straight through to facts = text, unhedged.
    # Gated off 2026-09-12 (HEDGE_MODERATE_TURNS): same over-firing as the
    # badge -- the tier is a trace rule, so a correct read of live state
    # ("which tracks are muted") got the writer told to open with "I don't
    # have a verified answer". Re-enable once the tier can tell an
    # observation from a recommendation.
    if tier == "moderate" and HEDGE_MODERATE_TURNS:
        facts = (
            "No verified, confirmed answer was established for this turn -- no "
            "confident, specific menu path, setting, or fix was found. Say plainly "
            "that you don't have a verified answer for this; if you offer anything "
            "further, frame it clearly as unconfirmed general guidance, not a "
            "confirmed fix.\n\n" + text
        )
    # The writer only ever sees text (see _plain_history), never the turn's
    # screenshots, so left to itself it describes its own vantage point --
    # found live 2026-09-12: asked "can you see my screen?", it answered "No,
    # I'm reading from Logic Pro's accessibility data", because the live-state
    # block below was the only context it had. Tell it what the decider saw.
    if had_screenshots:
        facts = (
            "The facts below were worked out from screenshots of the user's open "
            "Logic Pro windows (plus live project state where noted). Speak as "
            "someone looking at their screen; never say you can't see it.\n\n"
            + facts
        )
    # Code-guaranteed, not dependent on the decider's own text having restated
    # it (see respond()'s comment on live_state) -- appended last so it can't be
    # buried or dropped by whichever branch above ran. Framed as a partial set
    # of live values, not "ground truth": it's authoritative for the items it
    # lists, silent on everything else (a 2026-09-12 finding -- track-header
    # mute state wasn't in it at all, and the old wording made both models
    # treat that silence as "can't tell").
    if live_state:
        facts += (
            "\n\n## Live values read from the Logic Pro project this turn (not the "
            "user's claim). Authoritative for the items listed here -- if anything "
            "above contradicts one of these values, this is correct. Labels are "
            "Logic's internal accessibility names, not the on-screen ones:\n\n"
            + live_state
        )
    resp_text, writer_usage = await _write_response(facts, prior_messages, on_chunk)
    usage.append(writer_usage)
    return resp_text, tier


def _serialize_content(content: list) -> list[dict]:
    """SDK response content blocks -> plain request-shape dicts.

    resp.content items are response objects (TextBlock, ToolUseBlock, ...)
    carrying response-only fields (e.g. TextBlock.parsed_output) that the
    request-side schema doesn't accept. Appending them raw into msgs works
    fine *within* a single respond() call (never re-serialized before the
    next .create()), but Result.messages leaves the process as JSON for the
    client to round-trip as the next turn's history -- at that point the
    extra field is a literal dict key, and the API rejects it outright
    ("Extra inputs are not permitted") on the following request. Found live
    2026-08-08: turn 1 (fresh) succeeds, turn 2 ("show me that again",
    replaying turn 1's text block from history) 400s. Normalizing here,
    once, at the source keeps msgs safe for both same-turn reuse and
    cross-request replay.
    """
    out = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        elif b.type == "thinking":
            # Dropped, not persisted -- found live 2026-09-04 right after bumping
            # MODEL to claude-sonnet-5: unlike Sonnet 4.6, Sonnet 5 runs adaptive
            # thinking by default even with no `thinking` param sent, so every
            # decider call now produces one of these. api.py's _validate_history
            # only allows "text"/"tool_use" for assistant blocks -- replaying an
            # untouched "thinking" block in a later request's history 422s the
            # whole turn (broke every multi-turn conversation, not just research
            # ones). Safe to drop: this is a standard, non-interleaved manual
            # tool loop (no interleaved-thinking beta), so each _call_model
            # invocation thinks fresh regardless of what a prior iteration did.
            continue
        else:
            out.append(b.model_dump())
    return out


def _usage_dict(u) -> dict:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


async def _call_model(
    system: list[dict],
    tools_param: list[dict] | None,
    msgs: list[dict],
    on_chunk: OnChunk | None,
    tool_choice: dict | None = None,
    model: str | None = None,
):
    """One messages.create call, or the streamed equivalent when on_chunk is given.

    Streaming path forwards every text delta from every iteration (not just the
    final answer) — a tool-calling turn's own text_stream only ever carries its
    text blocks, never tool_use input, so a turn that also calls a tool doesn't
    accidentally leak partial JSON to the client. When on_chunk is None (every
    caller except api.py — the battery runner, drive_turn, and the mocked unit
    tests) this still calls _client.messages.create directly, unchanged, so
    existing mocks (which patch that exact method) keep working.

    tool_choice forces a specific tool instead of leaving the model free to
    answer with no tool call at all -- used for the iteration-0 forced
    lookup_concept call (see _needs_first_lookup). Only meaningful alongside
    tools_param; unset on every other call site.

    model defaults to the decider's MODEL; the writer call (_write_response)
    passes WRITER_MODEL instead -- the one place this loop's model varies.
    Resolved at call time, not bound as the default, so a runtime MODEL swap
    reaches every decider call. DECIDER_EFFORT / DECIDER_THINKING apply to
    decider calls only.
    """
    decider = model is None
    model = MODEL if decider else model
    kwargs = dict(model=model, max_tokens=MAX_TOKENS, system=system, messages=msgs)
    if decider and DECIDER_EFFORT is not None:
        kwargs["output_config"] = {"effort": DECIDER_EFFORT}
    if decider and DECIDER_THINKING is not None:
        kwargs["thinking"] = DECIDER_THINKING
    if tools_param is not None:
        kwargs["tools"] = tools_param
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    t0 = time.monotonic()
    if on_chunk is None:
        resp = await _client.messages.create(**kwargs)
    else:
        async with _client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                await on_chunk(text)
            resp = await stream.get_final_message()
    # Per-call timing + cache accounting, so the within-turn cache breakpoints
    # (see respond()'s _with_screenshots / live-state block) can be verified
    # from prod logs rather than a one-off replay. WARNING level to match the
    # other timing markers here (web_research_timing); same getattr guards as
    # _usage_dict since the mocked unit tests return bare SimpleNamespace usage.
    u = getattr(resp, "usage", None)
    log.warning(
        "[model_call] %s %.1fs in=%s cache_read=%s cache_write=%s out=%s forced=%s",
        model, time.monotonic() - t0,
        getattr(u, "input_tokens", "?"),
        getattr(u, "cache_read_input_tokens", 0) or 0,
        getattr(u, "cache_creation_input_tokens", 0) or 0,
        getattr(u, "output_tokens", "?"),
        tool_choice is not None,
    )
    return resp


def _media_type_for_b64(b64: str) -> str:
    """Sniff a screenshot's media type from its base64 prefix.

    v0.3.0 clients send PNG; v0.3.1+ send JPEG (client-v3 window_capture.py,
    after a 1568px PNG of a brushed-metal plugin window blew past the
    per-image cap live, 2026-09-09). The request shape carries no media type,
    so sniff the magic bytes rather than break old clients: base64 of
    `\\x89PNG` starts "iVBOR", base64 of the JPEG SOI marker `\\xff\\xd8\\xff`
    starts "/9j/". Unknown falls back to PNG, the pre-0.3.1 behavior.
    """
    if b64.startswith("/9j/"):
        return "image/jpeg"
    return "image/png"


def _last_user_text_index(msgs: list[dict]) -> int | None:
    """Index of the newest plain-string user message -- the turn boundary
    (same definition api.py's _trim_history snaps to)."""
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return i
    return None


def _prior_turn_looked_up(msgs: list[dict]) -> bool:
    """Whether the turn before the current one called lookup_concept. A
    follow-up that adds evidence mid-diagnosis ("spotify went silent too")
    tends to skip its own lookup because the last one is still in history --
    so for EARLY_EXIT_ON_ACTION it counts as a diagnostic turn, not a command."""
    cur = _last_user_text_index(msgs)
    if cur is None:
        return False
    prev = _last_user_text_index(msgs[:cur])
    for m in msgs[(prev or 0):cur]:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            if any(b.get("type") == "tool_use" and b.get("name") == "lookup_concept" for b in m["content"]):
                return True
    return False


def _pending_tool_uses(msgs: list[dict]) -> list:
    """tool_use blocks of a parked transcript's final assistant message, as
    objects with the .name/.input/.id the dispatch loop reads off SDK blocks.
    Empty unless the transcript really ends that way (no results after it)."""
    if not msgs or msgs[-1].get("role") != "assistant" or not isinstance(msgs[-1].get("content"), list):
        return []
    return [
        SimpleNamespace(type="tool_use", name=b.get("name"), input=b.get("input") or {}, id=b.get("id"))
        for b in msgs[-1]["content"]
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]


def _restore_turn_state(tail: list[dict]) -> dict:
    """Rebuild what respond() had accumulated when it parked the turn --
    trace, decider prose, a walkthrough attach, the lookup count -- from the
    completed tool-loop iterations between the user message and the pending
    assistant message. Nothing is stored server-side, so this is the only
    way an attach made before the research prompt survives into the answer
    (research_mixed_kb_and_research is a real scenario shape)."""
    trace: list[dict] = []
    text_parts: list[str] = []
    lookups_counted = 0
    buckets_seen: set = set()
    card_steps: list = []
    attach_keys: set = set()
    queued_by_lookup: dict[str, tuple[int, int]] = {}
    route_spans: dict[str, tuple[int, int]] = {}
    calls: dict[str, dict] = {}
    for m in tail:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if m.get("role") == "assistant":
                if b.get("type") == "text" and b.get("text"):
                    text_parts.append(b["text"])
                elif b.get("type") == "tool_use":
                    calls[b.get("id")] = {"tool": b.get("name"), "input": b.get("input") or {}}
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in calls:
                call = calls[b["tool_use_id"]]
                try:
                    output = json.loads(b.get("content") or "{}")
                except (TypeError, ValueError):
                    output = {"raw": b.get("content")}
                if not isinstance(output, dict):
                    output = {"raw": output}
                entry = {"tool": call["tool"], "input": call["input"], "output": output}
                trace.append(entry)
                if call["tool"] == "lookup_concept":
                    if COUNT_ALL_LOOKUPS or _lookup_is_unproductive(output, buckets_seen):
                        lookups_counted += 1
                    buckets_seen.add(output.get("problem"))
                    # Actions this lookup queued itself (see _auto_attach) ride
                    # in its own result, the only place they're recorded.
                    for auto in output.get("on_card") or []:
                        trace.append({"tool": auto["tool"], "input": auto["input"],
                                      "output": auto["output"], "auto_from": output.get("problem")})
                        if not auto["output"].get("attached"):
                            continue
                        start = len(card_steps)
                        card_steps.extend(auto["output"].get("steps") or [])
                        attach_keys.add(_attach_key(auto["tool"], auto["input"]))
                        queued_by_lookup[_endorse_key(auto["tool"], auto["input"])] = (start, len(card_steps))
                if call["tool"] in _ATTACH_TOOLS and output.get("attached"):
                    key = _endorse_key(call["tool"], call["input"])
                    new = output.get("steps") or []
                    if output.get("already_queued") or output.get("replaces_earlier"):
                        # Endorsed a lookup-queued step (it's model-called now) or
                        # re-visited a setting: the new version replaces the old
                        # span, as it did live.
                        span = queued_by_lookup.pop(key, None) or route_spans.get(key)
                        if span:
                            _replace_span(card_steps, *span, new, queued_by_lookup, route_spans)
                            if call["tool"] == "open_setting":
                                route_spans[key] = (span[0], span[0] + len(new))
                        if output.get("replaces_earlier"):
                            _mark_replaced(trace[:-1], call["input"].get("name"))
                        attach_keys.add(_attach_key(call["tool"], call["input"]))
                        continue
                    start = len(card_steps)
                    card_steps.extend(new)
                    attach_keys.add(_attach_key(call["tool"], call["input"]))
                    if call["tool"] == "open_setting":
                        route_spans[key] = (start, len(card_steps))
    return {
        "trace": trace, "text_parts": text_parts, "lookups_counted": lookups_counted,
        "buckets_seen": buckets_seen, "card_steps": card_steps, "attach_keys": attach_keys,
        "queued_by_lookup": queued_by_lookup, "route_spans": route_spans,
    }


async def _respond(
    messages: list[dict],
    ax_fixture: dict | None = None,
    on_chunk: OnChunk | None = None,
    on_status: OnStatus | None = None,
    screenshots_b64: list[str] | None = None,
    ax_state: str | None = None,
    research_confirm: bool = False,
    resume: str | None = None,
) -> Result:
    """research_confirm: the client wants to approve web research before it
    runs. When True and the decider calls web_research, this returns early
    with Result.pending_research_query set (nothing executed, no answer) so
    the client can ask the user. Installed clients that don't send it get
    the pre-2026-09-13 behaviour: research runs whenever the decider picks it.

    resume: "allow_research" or "deny_research" -- `messages` is then the
    transcript a prior call parked (ending in the assistant's pending
    tool_use(s)), and this call picks the turn up by executing those instead
    of calling the model first. Deny swaps the web_research result for
    _RESEARCH_DECLINED; allow runs it. Either way research is never prompted
    for again within this call. Stateless like everything else here: the
    parked transcript lives in the client's history round-trip, not on the
    server.
    """
    msgs = list(messages)
    trace: list[dict] = []
    pending_tool_uses: list | None = None
    research_authorized = resume == "allow_research"
    research_denied = resume == "deny_research"
    if resume:
        # `messages` (used below by _needs_first_lookup, _backfill_walkthrough
        # and the writer's plain history) must end at this turn's user
        # message like it does on a fresh call -- the parked tool-loop tail
        # is replayed into msgs/trace/text_parts instead.
        turn_start = _last_user_text_index(msgs)
        if turn_start is None:
            raise ValueError("resume: no user turn in history")
        messages = msgs[: turn_start + 1]
        pending_tool_uses = _pending_tool_uses(msgs)
        if not pending_tool_uses:
            raise ValueError("resume: history does not end in a pending tool call")
    # What open_setting checks "the user named this value" against.
    tools.TURN_USER_TEXT.set(_last_user_text(messages))
    tools.TURN_OFFERED_TEXT.set(_offered_text(messages))

    # Screenshots are fresh, per-turn context (see api.py's ChatRequest) --
    # spliced into the newest user turn only for this turn's own model calls
    # via _with_screenshots below, never into `msgs` itself. `msgs` is what
    # Result.messages returns as next turn's `history`, and that has to stay
    # plain text: the client's history validator has no "image" block type at
    # all (api.py's _validate_history), so an image surviving into history
    # would 400 the very next request. Keeping it out of msgs also means a
    # screenshot is never re-sent on every later turn -- only the turn it
    # actually arrived with pays for it.
    # The turn's user message is msgs[-1] on a fresh call; on a resume it's
    # further back (the parked tool-loop tail follows it), hence the search.
    screenshot_idx: int | None = None
    if screenshots_b64 and msgs:
        screenshot_idx = _last_user_text_index(msgs)
    if screenshot_idx is not None:
        screenshot_msg = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": _media_type_for_b64(b64), "data": b64}}
                for b64 in screenshots_b64
            ] + [{"type": "text", "text": msgs[screenshot_idx]["content"]}],
        }

    # Index of this turn's user message whether or not screenshots came with
    # it -- the within-turn cache breakpoint below goes on it either way.
    turn_idx: int | None = _last_user_text_index(msgs) if msgs else None

    def _with_screenshots(base: list[dict]) -> list[dict]:
        """Per-call copy of `base` with (a) the screenshot version of this
        turn's user message swapped in, and (b) a cache_control breakpoint on
        that message's last block. Every decider iteration of a turn resends
        the same prefix (system, tools, live state, prior history, this turn's
        images + question), so marking the user message lets iterations two
        onward read all of that from cache instead of re-processing it --
        measured 2026-09-18: ~9k uncached input tokens per call dropped to
        under 2.5k, roughly 1-1.5s per call, on a two-screenshot turn.
        Breakpoint count is three (static prompt, live-state block, this
        message), under the API's limit of four. Content is byte-identical
        either way; caching changes what's re-computed and billed, never
        what the model reads.

        The marker lives ONLY on this per-call copy. `msgs` is what
        Result.messages hands back to the client as next turn's history, and
        api.py's _validate_history rejects any extra key on a text block, so
        a cache_control that leaked into msgs would 422 the next request."""
        if turn_idx is None:
            return base
        out = list(base)
        if screenshot_idx is not None:
            content = list(screenshot_msg["content"])
        else:
            content = [{"type": "text", "text": base[turn_idx]["content"]}]
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
        out[turn_idx] = {"role": "user", "content": content}
        return out


    usage: list[dict] = []
    attached_solution: str | None = None   # no longer set; kept for _confidence_tier until tiers go
    lookups_counted = 0          # toward LOOKUP_ATTEMPT_LIMIT; see COUNT_ALL_LOOKUPS
    buckets_seen: set = set()
    card_steps: list = []        # every attach this turn, in call order (one card)
    attach_keys: set = set()
    queued_by_lookup: dict[str, tuple[int, int]] = {}   # endorse key -> its slice of card_steps
    route_spans: dict[str, tuple[int, int]] = {}        # model-called open_setting -> its slice
    pick_forced = False          # the one pick-or-ask re-call per turn (see _needs_pick)
    # Text the model writes alongside a tool call (not just the final,
    # tool-free iteration) is a real part of the answer, not scratch
    # thinking -- found live 2026-08-20 (hedge_mixed_grounded_ungrounded):
    # the model wrote the full, correct recording narration in the same
    # iteration as a get_walkthrough call, then wrote fresh export-only text
    # in the next (final) iteration, and only that last iteration's text was
    # ever returned -- the recording narration was silently discarded despite
    # the model having said it. Accumulating every iteration's text (in
    # order) instead of only the terminal one's fixes this at the root,
    # rather than papering over it with a post-hoc content check.
    text_parts: list[str] = []
    if resume:
        restored = _restore_turn_state(msgs[len(messages):-1])
        trace.extend(restored["trace"])
        text_parts.extend(restored["text_parts"])
        lookups_counted = restored["lookups_counted"]
        buckets_seen = restored["buckets_seen"]
        card_steps = restored["card_steps"]
        attach_keys = restored["attach_keys"]
        queued_by_lookup = restored["queued_by_lookup"]
        route_spans = restored["route_spans"]

    # Pushed, not pulled: live state (when known for this turn) is handed to the
    # model automatically rather than waiting on it to decide to call a tool for
    # it -- a decision point it's already been observed skipping under real
    # conditions (found live 2026-08-04, the monitor-button hallucination).
    system_text = SYSTEM_PROMPT
    # ax_fixture (dict) is the test battery's hand-authored, deterministic-gating
    # mechanism (see tools.py's toggle_ax_key/value_ax_key) -- never populated by
    # a real client request. ax_state (str) is the real thing: a live text dump
    # of whatever Logic windows/dialogs are actually open (client-v3's
    # core.ax_capture), same "pushed, not pulled" reasoning, just not shaped as
    # named key/value pairs since it's a passive tree read, not curated fixture
    # data. Both compose into the same instruction block when present.
    state_blocks = []
    if ax_fixture:
        state_blocks.append("\n".join(f"- {k}: {v}" for k, v in ax_fixture.items()))
    if ax_state:
        state_blocks.append(ax_state)
    # Also handed to _finalize_answer below (as `live_state`) so the writer call
    # gets it verbatim in its `facts` input -- the decider seeing this block in
    # its own system prompt is not enough on its own. Found live 2026-09-04
    # (battery Fable review): the writer's system prompt is built fresh from
    # WRITER_SYSTEM_PROMPT + facts (see _write_response) and never includes this
    # block, so a correction only reached the final answer when the decider's
    # own generated text happened to restate it -- confirmed failing on a direct
    # user-claim contradiction (ax_contradicts_user_claim, ax_state_contradicts_
    # user_claim both wrong in the 2026-09-04 full-battery run). Appending it to
    # facts is a code-guaranteed backstop, not a hope that the decider restates
    # it -- same "code over prompt" reasoning as everything else pushed state
    # relies on in this file.
    live_state = "\n\n".join(state_blocks) if state_blocks else None
    # Cached as its own block: identical across every iteration of this turn's
    # loop (system+tools resent unchanged on each one -- measured 2026-08-05:
    # 66 calls, 207,869 uncached input tokens across a 24-scenario battery).
    # The safety-net call below appends its own instruction as a second,
    # uncached block so it still hits this same cache entry.
    #
    # The live-state section is deliberately its OWN block after the cached
    # one, never concatenated into it: it changes every turn, and while it
    # was part of the same text (2026-09-04 .. 2026-09-12) the cache prefix
    # changed every turn too, so the static prompt never hit across turns.
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
    if state_blocks:
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
                + live_state
            ),
            # Its own breakpoint: constant across this turn's iterations (only
            # changes between turns), so iterations two onward read the AX dump
            # from cache rather than re-parsing up to ~3.5k tokens of it each
            # call. Second of the three breakpoints per call (see
            # _with_screenshots for the third and the measured effect).
            "cache_control": {"type": "ephemeral"},
        })

    # Tool-skip gap fix: force lookup_concept on this turn's first iteration
    # (see _needs_first_lookup) instead of leaving "call a tool at all" up to
    # the model's judgment. Decided once, up front, against the original
    # `messages` param -- same turn-defining message _backfill_walkthrough
    # itself checks.
    force_first_lookup = _needs_first_lookup(messages)

    for i in range(MAX_ITERATIONS):
        # The decider's own text never reaches the client directly anymore --
        # only the writer call (see _finalize_answer) streams -- so every
        # decider call runs non-streaming regardless of whether this respond()
        # call was itself given an on_chunk. Status text takes the same shape:
        # "Thinking..." while this call is in flight (we don't know yet
        # whether -- or which -- tool it'll pick), overwritten below once
        # tool_uses is known, and reset back to "Thinking..." here on every
        # later iteration too (e.g. after tool results come back and the
        # decider is composing on top of them).
        if i == 0 and pending_tool_uses:
            # Resumed turn: the model already made this iteration's call (it's
            # the last message in msgs); execute what it asked for.
            tool_uses = pending_tool_uses
            resp = None
        else:
            if on_status:
                await on_status("Thinking…")
            tool_choice = None
            if i == 0 and force_first_lookup:
                tool_choice = (
                    {"type": "tool", "name": "lookup_concept"} if FORCE_FIRST_LOOKUP
                    else {"type": "any"}
                )
            resp = await _call_model(system, tools.TOOLS, _with_screenshots(msgs), None, tool_choice)
            usage.append(_usage_dict(resp.usage))
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            iter_text = "".join(b.text for b in resp.content if b.type == "text")
            if not tool_uses and not pick_forced and _needs_pick(trace, card_steps):
                # About to end on prose after a bucket lookup whose candidates
                # all map to actions: once per turn, re-ask with a tool call
                # required, so the pick the answer makes reaches the card (or
                # becomes a clarifying question). The prose-only reply is
                # discarded -- its replacement is made with the same context.
                pick_forced = True
                resp = await _call_model(
                    system + [{"type": "text", "text": "\n\n" + _PICK_NUDGE}],
                    tools.TOOLS, _with_screenshots(msgs), None, {"type": "any"},
                )
                usage.append(_usage_dict(resp.usage))
                tool_uses = [b for b in resp.content if b.type == "tool_use"]
                iter_text = "".join(b.text for b in resp.content if b.type == "text")
            if iter_text:
                text_parts.append(iter_text)

        if not tool_uses:
            text = "\n\n".join(text_parts)
            msgs.append({"role": "assistant", "content": _serialize_content(resp.content)})
            if not trace:
                backfilled = _backfill_walkthrough(messages, text)
                if backfilled:
                    card_steps = list(backfilled["steps"])
                    # Re-showing an earlier card is not a fresh instruction --
                    # it waits for Run like any lookup-queued card.
                    queued_by_lookup = {"backfill": (0, len(card_steps))}
                    trace.append({
                        "tool": "open_setting",
                        "input": {"name": backfilled["destination"]},
                        "output": {
                            "attached": True,
                            "destination": backfilled["destination"],
                            "steps": backfilled["steps"],
                            "backfilled": True,
                        },
                    })
            text, tier = await _finalize_answer(
                trace, (list(card_steps) or None), text, messages, on_chunk, usage,
                attached_solution, live_state=live_state, had_screenshots=screenshot_idx is not None,
                has_actions=bool(card_steps),
            )
            return Result(text=text, walkthrough_steps=(list(card_steps) or None),
                          trace=trace, messages=msgs, usage=usage, confidence_tier=tier,
                          sources=_collect_sources(trace), card_from_lookup=bool(queued_by_lookup))

        if resp is not None:
            msgs.append({"role": "assistant", "content": _serialize_content(resp.content)})
        # Park the turn before executing ANY of this iteration's tools when
        # one of them is an unapproved web_research: the client asks the
        # user and resumes with the whole batch, so every tool_use in the
        # assistant message still gets exactly one tool_result (a partial set
        # is rejected by the API).
        if research_confirm and not research_authorized and not research_denied:
            pending = [tu for tu in tool_uses if tu.name == "web_research"]
            if pending:
                log.warning("[research_prompt] parking turn for user approval, query=%r", pending[0].input.get("query"))
                return Result(
                    text="", walkthrough_steps=None, trace=trace, messages=msgs, usage=usage,
                    confidence_tier="generic", pending_research_query=str(pending[0].input.get("query", "")),
                )
        if on_status:
            status_text = _status_for_tools(tool_uses)
            if status_text:
                await on_status(status_text)
        tool_results = []
        for tu in tool_uses:
            if tu.name == "lookup_concept" and lookups_counted >= LOOKUP_ATTEMPT_LIMIT:
                result = {
                    "error": (
                        "Too many lookups without a clear answer. "
                        "Stop searching now — answer from general Logic Pro knowledge if you're "
                        "genuinely confident, or tell the user you don't have a verified answer "
                        "for this. Do not call lookup_concept again this turn."
                    )
                }
            elif tu.name in _ATTACH_TOOLS and _endorse_key(tu.name, tu.input) in queued_by_lookup:
                # The model called the action a lookup had already queued: that's
                # agreement, not a second request. Its version (which may name a
                # track) replaces the queued step, and the step stops counting as
                # lookup-queued, so an instruction-shaped turn can auto-run.
                result = await _EXECUTORS[tu.name](tu.input, ax_fixture)
                if result.get("attached"):
                    # Popped only on success: a refused version (a dropdown value
                    # the user didn't name) leaves the queued step as it was, so
                    # a corrected call still replaces it instead of joining it.
                    key = _endorse_key(tu.name, tu.input)
                    start, end = queued_by_lookup.pop(key)
                    new = result.get("steps") or []
                    _replace_span(card_steps, start, end, new, queued_by_lookup, route_spans)
                    if tu.name == "open_setting":
                        route_spans[key] = (start, start + len(new))
                    attach_keys.add(_attach_key(tu.name, tu.input))
                    result = {**result, "already_queued": True}
            elif tu.name in _ATTACH_TOOLS and _attach_key(tu.name, tu.input) in attach_keys:
                result = {"attached": False, "reason": _DUPLICATE_ATTACH_REFUSAL}
            elif tu.name == "open_setting" and _endorse_key(tu.name, tu.input) in route_spans:
                # One visit per setting per card, and the later call wins: two
                # values for one dropdown both ran, and "first stands" put
                # Rhythmic on the card while the reply said Monophonic (both
                # found in the 2026-09-22 battery, calls in the same reply). The
                # later call is the correction the prose follows.
                result = await _EXECUTORS[tu.name](tu.input, ax_fixture)
                if result.get("attached"):
                    key = _endorse_key(tu.name, tu.input)
                    start, end = route_spans[key]
                    new = result.get("steps") or []
                    _replace_span(card_steps, start, end, new, queued_by_lookup, route_spans)
                    route_spans[key] = (start, start + len(new))
                    _mark_replaced(trace, tu.input.get("name"))
                    attach_keys.add(_attach_key(tu.name, tu.input))
                    result = {**result, "replaces_earlier": True}
            elif tu.name in _ATTACH_TOOLS and len(attach_keys) >= MAX_ATTACHES_PER_TURN:
                result = {"attached": False, "reason": _ATTACH_CAP_REFUSAL}
            elif tu.name in _ATTACH_TOOLS and _is_alternative_pick(tu.name, tu.input, trace):
                # Commit-to-one for alternatives: a second candidate for the
                # SAME problem is a fallback, not a second request (found live
                # 2026-08-05 -- the last candidate silently won, contradicting
                # whichever one the response led with). A walkthrough for a
                # different problem is a second thing the user asked for and
                # joins the card (see MAX_ATTACHES_PER_TURN).
                result = {"attached": False, "reason": _FALLBACK_REFUSAL}
            elif tu.name == "web_research" and research_denied:
                result = dict(_RESEARCH_DECLINED)
            else:
                executor = _EXECUTORS.get(tu.name)
                result = {"error": f"unknown tool {tu.name!r}"} if executor is None else await executor(tu.input, ax_fixture)
            # web_research is the only executor that makes its own real API call
            # (a nested Sonnet + web_search request, see app/research.py) --
            # pop its usage back out here so the turn's real token spend still
            # gets counted (api.py sums Result.usage into the logged event),
            # without that bookkeeping key ever reaching the model inside the
            # tool_result content below. No-op for every other tool (none of
            # them set "_usage").
            call_usage = result.pop("_usage", None) if isinstance(result, dict) else None
            if call_usage:
                usage.append(call_usage)
            on_card: list[dict] = []
            if (
                tu.name == "lookup_concept" and result.get("match") == "single"
                and str(result.get("match_confidence", "")).startswith("strong")
            ):
                sol = (result.get("solutions") or [{}])[0]
                for tool, inp in tools.action_calls(sol.get("action")):
                    key = _attach_key(tool, inp)
                    if (key in attach_keys or len(attach_keys) >= MAX_ATTACHES_PER_TURN
                            or (tool == "open_setting" and _endorse_key(tool, inp) in route_spans)):
                        continue
                    if _is_alternative_pick(tool, inp, trace):
                        # Same commit-to-one rule as a model-made pick: don't let
                        # a by-name lookup of a second candidate ("sample rate"
                        # after buffer size) slip a competing fix onto the card.
                        on_card.append({"tool": tool, "input": inp,
                                        "output": {"attached": False, "reason": _FALLBACK_REFUSAL}})
                        continue
                    out = await _EXECUTORS[tool](inp, ax_fixture)
                    on_card.append({"tool": tool, "input": inp, "output": out})
                    if out.get("attached"):
                        start = len(card_steps)
                        card_steps.extend(out.get("steps") or [])
                        attach_keys.add(key)
                        queued_by_lookup[_endorse_key(tool, inp)] = (start, len(card_steps))
                if on_card:
                    refused = [o["output"].get("reason") for o in on_card if not o["output"].get("attached")]
                    note = _ON_CARD_NOTE
                    if any(o["tool"] == "open_setting" and o["output"].get("note") for o in on_card):
                        note += _PANE_ONLY_NOTE
                    result = {**result, "on_card": on_card,
                              "on_card_note": note if not refused else
                              "This solution's action was NOT queued: " + "; ".join(r for r in refused if r)}
            trace.append({"tool": tu.name, "input": tu.input, "output": result})
            trace.extend({**o, "auto_from": result.get("problem")} for o in on_card)
            if tu.name == "lookup_concept" and "error" not in result:
                if COUNT_ALL_LOOKUPS or _lookup_is_unproductive(result, buckets_seen):
                    lookups_counted += 1
                buckets_seen.add(result.get("problem"))
            if (tu.name in _ATTACH_TOOLS and result.get("attached")
                    and not result.get("already_queued") and not result.get("replaces_earlier")):
                start = len(card_steps)
                card_steps.extend(result.get("steps") or [])
                attach_keys.add(_attach_key(tu.name, tu.input))
                if tu.name == "open_setting":
                    route_spans[_endorse_key(tu.name, tu.input)] = (start, len(card_steps))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })
        msgs.append({"role": "user", "content": tool_results})

        # Early exit (A/B, 2026-09-24): an iteration that called only action
        # tools, all attached, has nothing left to decide -- the next decider
        # call would only write prose the writer then rewrites (~2.5s median).
        # Any refusal or error keeps looping so the decider can react to it.
        # Direct actions only: after a lookup, that next call is where the
        # decider explains the diagnosis and next steps, and skipping it lost
        # them (battery review, 2026-09-24). Same for a follow-up to a turn
        # that looked something up (see _prior_turn_looked_up).
        if (EARLY_EXIT_ON_ACTION and tool_uses
                and all(tu.name in _ATTACH_TOOLS for tu in tool_uses)
                and all(json.loads(tr["content"]).get("attached") for tr in tool_results)
                and not any(t["tool"] == "lookup_concept" for t in trace)
                and not _prior_turn_looked_up(messages)):
            text = "\n\n".join(text_parts)
            text, tier = await _finalize_answer(
                trace, (list(card_steps) or None), text, messages, on_chunk, usage,
                attached_solution, live_state=live_state, had_screenshots=screenshot_idx is not None,
                has_actions=bool(card_steps),
            )
            return Result(text=text, walkthrough_steps=(list(card_steps) or None),
                          trace=trace, messages=msgs, usage=usage, confidence_tier=tier,
                          sources=_collect_sources(trace), card_from_lookup=bool(queued_by_lookup))

        # ask_clarifying_question ends the turn immediately, same as the
        # no-tool-uses path above -- it's not "not the final answer, keep
        # looping" like every other tool, it IS the final answer (a claim-free
        # question). No backfill check here: reaching this branch means trace
        # is non-empty (this tool call itself is in it), and backfill only
        # ever applies to a turn with zero tool calls at all.
        if any(tu.name == "ask_clarifying_question" for tu in tool_uses):
            text = "\n\n".join(text_parts)
            text, tier = await _finalize_answer(
                trace, (list(card_steps) or None), text, messages, on_chunk, usage,
                attached_solution, is_clarifying_question=True, live_state=live_state,
                had_screenshots=screenshot_idx is not None, has_actions=bool(card_steps),
            )
            return Result(text=text, walkthrough_steps=(list(card_steps) or None),
                          trace=trace, messages=msgs, usage=usage, confidence_tier=tier,
                          sources=_collect_sources(trace), card_from_lookup=bool(queued_by_lookup))

    # Safety net: never return a truly empty response, regardless of why the
    # loop didn't converge on its own. Force one final tools-off call so the
    # model must synthesize whatever it already learned into a real answer.
    # Non-streaming, same reasoning as every decider call above -- only the
    # writer call streams.
    final = await _call_model(
        system + [{
            "type": "text",
            "text": "\n\nAnswer now with your best available information — no more tool calls.",
        }],
        None,
        _with_screenshots(msgs),
        None,
    )
    usage.append(_usage_dict(final.usage))
    final_text = "".join(b.text for b in final.content if b.type == "text")
    if final_text:
        text_parts.append(final_text)
    text = "\n\n".join(text_parts)
    msgs.append({"role": "assistant", "content": _serialize_content(final.content)})
    # No backfill check here (unlike the early-return path above): reaching this
    # point requires every one of MAX_ITERATIONS loop passes to have had at least
    # one tool call, each of which unconditionally appends to trace -- so trace
    # can never be empty here, and _backfill_walkthrough only ever fires on a
    # turn with zero tool calls. A backfill check here was dead code (found via
    # Fable review, 2026-08-06) and has been removed rather than left in place.
    text, tier = await _finalize_answer(
        trace, (list(card_steps) or None), text, messages, on_chunk, usage,
        attached_solution, live_state=live_state, had_screenshots=screenshot_idx is not None,
        has_actions=bool(card_steps),
    )
    return Result(text=text, walkthrough_steps=(list(card_steps) or None),
                  trace=trace, messages=msgs, usage=usage, confidence_tier=tier,
                  sources=_collect_sources(trace), card_from_lookup=bool(queued_by_lookup))
