"""Code-side reads of the user's own message: is it a question, an undo request, a
greeting or closer, may its card auto-run, is it a go-ahead that forces a tool call.
Each phrase list is deliberately narrow; why each phrase is in or out: README.md
(Heuristics). Re-sweep the battery whenever a list changes."""
import json
import re

from app.pipeline.transcript import last_user_text_index

# Question-shaped or bulk requests never auto-run.
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

# Typed undo/revert is declined (the card's Revert button does it). Only an
# instruction to Conductor counts, not "how do i undo a cut in logic".
_UNDO_REQUEST = re.compile(
    r"\b(?:undo|revert|(?:put|change|set|switch|turn) (?:it|that|this|them|those) back)\b"
)

# Asking to see an earlier destination again (drives backfill.py).
_REASK_SIGNALS = (
    "remind me", "reminder", "one more time", "show me that",
    "show that again", "where was", "where is that", "closed the window",
    "lost the window",
)

# Plain acknowledgments: nothing real to look up.
_CLOSING_SIGNALS = (
    "thanks", "thank you", "thx", "ty",
    "got it", "sounds good", "cool", "perfect", "great", "awesome",
    "ok", "okay", "no more questions", "that's all", "im good", "i'm good",
    "bye", "goodbye", "see ya",
)

# Whole-project destructive requests: declined outright, never forced to call a tool.
_IRREVERSIBLE_SIGNALS = (
    "delete my entire project", "delete the entire project",
    "delete my whole project", "delete the whole project",
    "erase my entire project", "erase the entire project",
    "erase my whole project", "erase the whole project",
    "delete everything in my project", "delete everything in the project",
    "erase everything in my project", "erase everything in the project",
    "wipe my entire project", "wipe the entire project",
)

# Matched against the WHOLE message, not as substrings ("hi" is in "this").
_GREETING_SIGNALS = (
    "hi", "hey", "hello", "yo", "sup", "hiya", "howdy", "greetings",
    "hey there", "hi there", "what's up", "whats up", "good morning",
    "good afternoon", "good evening",
)


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].lower().replace("’", "'").replace("'", "").strip()
    return ""


def offered_text(messages: list[dict]) -> str:
    """The reply the newest user message answers (see tools.TURN_OFFERED_TEXT).
    Any reply counts, since options are often offered as a statement; a first
    turn has none, so the model still can't pick a value the user didn't name."""
    i = last_user_text_index(messages)
    for m in reversed(messages[:i] if i is not None else []):
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else " ".join(
            b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text") if isinstance(c, list) else ""
        if text.strip():
            return text.lower()
    return ""


def is_question(text: str) -> bool:
    if _POLITE_REQUEST.match(text):
        return bool(_QUESTION_ANYWHERE.search(text))
    return text.endswith("?") or bool(_QUESTION_START.match(text) or _QUESTION_ANYWHERE.search(text))


def is_undo_request(text: str) -> bool:
    return bool(text) and not is_question(text) and bool(_UNDO_REQUEST.search(text))


def auto_run_ok(messages: list[dict], card: list | None) -> bool:
    """Whether the user's own message is a direct instruction (not a question,
    not bulk). respond.py adds the other auto-run conditions."""
    if not card:
        return False
    text = last_user_text(messages)
    return bool(text) and not is_question(text) and not any(sig in text for sig in _BULK_SIGNALS)


def last_user_message_matches(messages: list[dict], signals: tuple[str, ...]) -> bool:
    """True if the newest message is a user turn containing any of the phrases
    (case/apostrophe-insensitive substring match)."""
    if not messages or messages[-1].get("role") != "user":
        return False
    user_text = messages[-1].get("content")
    if not isinstance(user_text, str):
        return False
    normalized = user_text.lower().replace("'", "")
    return any(sig in normalized for sig in signals)


def is_reask(messages: list[dict]) -> bool:
    return last_user_message_matches(messages, _REASK_SIGNALS)


def is_bare_greeting(messages: list[dict]) -> bool:
    """True if the newest user message, punctuation stripped, is exactly a greeting."""
    if not messages or messages[-1].get("role") != "user":
        return False
    user_text = messages[-1].get("content")
    if not isinstance(user_text, str):
        return False
    normalized = user_text.lower().strip(" !.?").replace("'", "")
    return normalized in _GREETING_SIGNALS


def follows_a_card(messages: list[dict]) -> bool:
    """Whether the reply before this turn's user message queued a card."""
    cur = last_user_text_index(messages)
    if cur is None:
        return False
    prev = last_user_text_index(messages[:cur])
    for m in messages[(prev or 0):cur]:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                try:
                    out = json.loads(b.get("content") or "{}") if isinstance(b, dict) else {}
                except (TypeError, ValueError):
                    continue
                if isinstance(out, dict) and (out.get("attached") or any(
                        (o.get("output") or {}).get("attached") for o in out.get("on_card") or [])):
                    return True
    return False


def force_first_call(messages: list[dict]) -> bool:
    """The one case the turn's first decider call must call a tool: a go-ahead
    ("yes please, do it") after a reply that queued a card. Unforced, it got "press
    Run" about the earlier card. Questions, re-asks, closers, irreversible requests
    and greetings are never forced (README: Forced first call)."""
    text = last_user_text(messages)
    return (bool(text) and follows_a_card(messages) and not is_question(text)
            and not last_user_message_matches(messages, _REASK_SIGNALS)
            and not last_user_message_matches(messages, _CLOSING_SIGNALS)
            and not last_user_message_matches(messages, _IRREVERSIBLE_SIGNALS)
            and not is_bare_greeting(messages))
