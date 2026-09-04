"""v3 pipeline: query + context -> tool-calling loop -> response.

respond() takes the full running conversation (already ending with the newest
user turn) and returns the full conversation after this turn completes, so the
caller (the test harness) just appends the next user turn for the next call.
History is the only state — nothing is pinned server-side between turns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from anthropic import AsyncAnthropic

from app import research, tools
from app.config import CENTRAL_ANTHROPIC_KEY, MAX_TOKENS, MODEL, RESEARCH_CALL_TIMEOUT_S, WRITER_MODEL
from app.prompt import SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT

OnChunk = Callable[[str], Awaitable[None]]
OnStatus = Callable[[str], Awaitable[None]]

MAX_ITERATIONS = 6

# Hard, code-enforced cap on lookup_concept calls within one turn — a prompt
# instruction to "stop retrying" is a request, not a guarantee. Found live
# 2026-07-30: a genuinely absent topic ("wheres the monitor button") kept
# returning a *different* wrong match on every retried phrasing, so the model
# never hit a clean, repeated "nothing here" signal and just kept trying new
# wording until it hit the overall iteration ceiling with no answer at all.
# This intervenes explicitly before that happens, distinct from the overall
# MAX_ITERATIONS safety net below (which covers any tool, not just this one).
LOOKUP_ATTEMPT_LIMIT = 4

_client = AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY)


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
            return f"Searching the web for a verified answer (may take up to {_format_duration(RESEARCH_CALL_TIMEOUT_S)})…"
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


_EXECUTORS = {
    "lookup_concept": lambda inp, fixture: tools.lookup_concept(inp["problem"]),
    "get_walkthrough": lambda inp, fixture: tools.get_walkthrough(inp["solution"], ax_fixture=fixture),
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
    # Citations from this turn's web_research call(s), if any -- empty for
    # every other turn. Rendered as source chips client-side (message_widget.py
    # set_sources, built ahead of the tool itself on 2026-09-04).
    sources: list[dict] = field(default_factory=list)


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
    the model answered from memory instead of re-calling get_walkthrough (found
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
    find there if this turn made no tool calls) for successful get_walkthrough
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
            if result.get("attached") and result.get("destination") and result.get("steps"):
                attached_by_destination[result["destination"]] = result["steps"]

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
        and any(s.get("has_path") for s in c["output"].get("solutions") or [])
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
    Returns "moderate" exactly when this turn made a real attempt (a tool call) but
    never reached strong grounding. Never "moderate" on zero tool calls -- can't
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
    return "strong" if _strong_grounded_lookup_exists(trace) else "moderate"


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


def _solution_to_problem(trace: list[dict]) -> dict[str, str | None]:
    """Maps each solution name surfaced by this turn's lookup_concept calls to
    the problem bucket it was returned under, so a refused second get_walkthrough
    can tell whether it's an alternative candidate for the SAME problem (a
    fallback) or a solution from a DIFFERENT problem entirely (a separate,
    unrelated request) -- see hedge_mixed_grounded_ungrounded below."""
    mapping: dict[str, str | None] = {}
    for c in trace:
        if c["tool"] != "lookup_concept":
            continue
        problem = c["output"].get("problem")
        for sol in c["output"].get("solutions") or []:
            mapping.setdefault(sol["name"], problem)
    return mapping


_FALLBACK_REFUSAL = (
    "a walkthrough was already attached this turn -- only one attaches per "
    "turn. Mention a fallback candidate in prose instead, and only attach it "
    "if the user comes back and says the first one didn't work."
)

# hedge_mixed_grounded_ungrounded (v3-log.md 2026-08-18, reproduced 3/3): when
# the refused solution is from a DIFFERENT problem bucket than the one that
# already attached, it isn't a fallback candidate for that attach -- it's an
# unrelated second request sharing the same turn. The generic fallback
# refusal above was found to make the model frame it as "next thing to try,"
# which consistently crowded out narrating the walkthrough that actually won.
_UNRELATED_REFUSAL = (
    "a walkthrough was already attached this turn for a DIFFERENT, unrelated "
    "request -- only one attaches per turn, but this is not a fallback for "
    "that attach. Cover this solution's own steps directly in your prose "
    "instead of attaching it. Also make sure your response still fully "
    "narrates the walkthrough that DID attach -- don't let this one crowd it out."
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
) -> tuple[str, str]:
    """Called once this turn's decider text is fully known (either the
    no-more-tool-calls return or the MAX_ITERATIONS safety net). Tier decision
    stays exactly as before -- deterministic, code-enforced (_confidence_tier) --
    but as of 2026-09-04 no longer prepends a literal _HEDGE_PREFIX to the
    streamed/returned text; the client renders the tier as a confidence badge
    instead (source_tier in the /v1/chat "done" payload -- see api.py). The
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
    # Only "moderate" gets the hedge facts injected below -- "research" is
    # trusted the same as "strong" (see _confidence_tier's 2026-09-04 note) and
    # deliberately falls straight through to facts = text, unhedged.
    if tier == "moderate":
        facts = (
            "No verified, confirmed answer was established for this turn -- no "
            "confident, specific menu path, setting, or fix was found. Say plainly "
            "that you don't have a verified answer for this; if you offer anything "
            "further, frame it clearly as unconfirmed general guidance, not a "
            "confirmed fix.\n\n" + text
        )
    # Code-guaranteed, not dependent on the decider's own text having restated
    # it (see respond()'s comment on live_state) -- appended last so it can't be
    # buried or dropped by whichever branch above ran.
    if live_state:
        facts += (
            "\n\n## Ground truth for this turn, from Logic Pro's live Accessibility "
            "state (not the user's claim) -- if anything above conflicts with this, "
            "this is correct:\n\n" + live_state
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
    model: str = MODEL,
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
    """
    kwargs = dict(model=model, max_tokens=MAX_TOKENS, system=system, messages=msgs)
    if tools_param is not None:
        kwargs["tools"] = tools_param
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if on_chunk is None:
        return await _client.messages.create(**kwargs)
    async with _client.messages.stream(**kwargs) as stream:
        async for text in stream.text_stream:
            await on_chunk(text)
        return await stream.get_final_message()


async def respond(
    messages: list[dict],
    ax_fixture: dict | None = None,
    on_chunk: OnChunk | None = None,
    on_status: OnStatus | None = None,
    screenshots_b64: list[str] | None = None,
    ax_state: str | None = None,
) -> Result:
    msgs = list(messages)
    trace: list[dict] = []

    # Screenshots are fresh, per-turn context (see api.py's ChatRequest) --
    # spliced into the newest user turn only for this turn's own model calls
    # via _with_screenshots below, never into `msgs` itself. `msgs` is what
    # Result.messages returns as next turn's `history`, and that has to stay
    # plain text: the client's history validator has no "image" block type at
    # all (api.py's _validate_history), so an image surviving into history
    # would 400 the very next request. Keeping it out of msgs also means a
    # screenshot is never re-sent on every later turn -- only the turn it
    # actually arrived with pays for it.
    screenshot_idx: int | None = None
    if screenshots_b64 and msgs and msgs[-1].get("role") == "user" and isinstance(msgs[-1].get("content"), str):
        screenshot_idx = len(msgs) - 1
        screenshot_msg = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
                for b64 in screenshots_b64
            ] + [{"type": "text", "text": msgs[screenshot_idx]["content"]}],
        }

    def _with_screenshots(base: list[dict]) -> list[dict]:
        if screenshot_idx is None:
            return base
        out = list(base)
        out[screenshot_idx] = screenshot_msg
        return out


    usage: list[dict] = []
    walkthrough_steps: list | None = None
    attached_solution: str | None = None
    lookup_attempts = 0
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
    if state_blocks:
        system_text += (
            "\n\n## Live state for this turn\n\n"
            "Read directly from the running Logic Pro project via the Accessibility "
            "API -- ground truth, not something the user said or you inferred. This "
            "outranks stated claims, seed_weight, and anything read from a screenshot "
            "when they conflict.\n\n"
            + live_state
        )
    # Cached as its own block: identical across every iteration of this turn's
    # loop (system+tools resent unchanged on each one -- measured 2026-08-05:
    # 66 calls, 207,869 uncached input tokens across a 24-scenario battery).
    # The safety-net call below appends its own instruction as a second,
    # uncached block so it still hits this same cache entry.
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

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
        if on_status:
            await on_status("Thinking…")
        tool_choice = None
        if i == 0 and force_first_lookup:
            tool_choice = {"type": "tool", "name": "lookup_concept"}
        resp = await _call_model(system, tools.TOOLS, _with_screenshots(msgs), None, tool_choice)
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
                    walkthrough_steps = backfilled["steps"]
                    trace.append({
                        "tool": "get_walkthrough",
                        "input": {"solution": None},
                        "output": {
                            "attached": True,
                            "destination": backfilled["destination"],
                            "steps": backfilled["steps"],
                            "backfilled": True,
                        },
                    })
            text, tier = await _finalize_answer(
                trace, walkthrough_steps, text, messages, on_chunk, usage, attached_solution,
                live_state=live_state,
            )
            return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs,
                           usage=usage, confidence_tier=tier, sources=_collect_sources(trace))

        msgs.append({"role": "assistant", "content": _serialize_content(resp.content)})
        if on_status:
            status_text = _status_for_tools(tool_uses)
            if status_text:
                await on_status(status_text)
        tool_results = []
        for tu in tool_uses:
            if tu.name == "lookup_concept":
                lookup_attempts += 1
            if tu.name == "lookup_concept" and lookup_attempts > LOOKUP_ATTEMPT_LIMIT:
                result = {
                    "error": (
                        f"Too many lookup attempts ({lookup_attempts}) without a clear answer. "
                        "Stop searching now — answer from general Logic Pro knowledge if you're "
                        "genuinely confident, or tell the user you don't have a verified answer "
                        "for this. Do not call lookup_concept again this turn."
                    )
                }
            elif tu.name == "get_walkthrough" and walkthrough_steps is not None:
                # Commit-to-one: only the first successful attach in a turn
                # reaches the user (found live 2026-08-05 -- multiple attaches
                # in one turn meant the *last* one silently won, contradicting
                # whichever candidate the response text actually led with).
                # Which refusal wording applies depends on whether this is an
                # alternative candidate for the SAME problem as the winner (a
                # real fallback) or a solution from a DIFFERENT problem (an
                # unrelated second request) -- see _solution_to_problem.
                sol_to_problem = _solution_to_problem(trace)
                same_bucket = (
                    sol_to_problem.get(tu.input.get("solution"))
                    == sol_to_problem.get(attached_solution)
                )
                result = {
                    "attached": False,
                    "reason": _FALLBACK_REFUSAL if same_bucket else _UNRELATED_REFUSAL,
                }
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
            trace.append({"tool": tu.name, "input": tu.input, "output": result})
            if tu.name == "get_walkthrough" and result.get("attached"):
                walkthrough_steps = result.get("steps")
                attached_solution = tu.input.get("solution")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })
        msgs.append({"role": "user", "content": tool_results})

        # ask_clarifying_question ends the turn immediately, same as the
        # no-tool-uses path above -- it's not "not the final answer, keep
        # looping" like every other tool, it IS the final answer (a claim-free
        # question). No backfill check here: reaching this branch means trace
        # is non-empty (this tool call itself is in it), and backfill only
        # ever applies to a turn with zero tool calls at all.
        if any(tu.name == "ask_clarifying_question" for tu in tool_uses):
            text = "\n\n".join(text_parts)
            text, tier = await _finalize_answer(
                trace, walkthrough_steps, text, messages, on_chunk, usage,
                attached_solution, is_clarifying_question=True, live_state=live_state,
            )
            return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs,
                           usage=usage, confidence_tier=tier, sources=_collect_sources(trace))

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
        trace, walkthrough_steps, text, messages, on_chunk, usage, attached_solution,
        live_state=live_state,
    )
    return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs,
                   usage=usage, confidence_tier=tier, sources=_collect_sources(trace))
