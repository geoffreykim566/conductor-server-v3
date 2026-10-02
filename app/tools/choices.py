"""Dropdown routes: matching a model-sent value to an option, and checking the user
actually named it. Rules: README.md (Dropdown routes)."""
import re

from app.tools.routes import RELATIVE_VALUES


def pane_only_ask(choice: dict) -> str:
    """What the reply should do when a dropdown route stops at the pane: ask which
    option, naming them exactly, so the answer can set it next turn (TURN_OFFERED_TEXT)."""
    options = choice.get("options") or []
    if len(options) < 2:
        return "the user picks there"
    return ("ask the user which one they want, naming the options exactly: " + " | ".join(options)
            + " -- or, if your answer recommends one, name it and offer to set it")


def _number(text: str) -> float | None:
    m = re.search(r"\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None


def _match_option(value: str, options: list[str], shows: dict | None = None) -> str | None:
    """The option a model-sent value means: exact (any case), what the control
    displays for it ("beats" -> "On + Align Bars and Beats"), same number
    ("256 samples" -> "256", "48000" -> "48 kHz"), or a unique prefix
    ("mono" -> "Monophonic")."""
    v = value.strip().lower()
    exact = next((o for o in options if o.lower() == v), None) or next(
        (o for o, d in (shows or {}).items() if d.lower() == v and o in options), None)
    if exact:
        return exact
    n = _number(v)
    if n is not None:
        for cand in (n, n / 1000):
            hits = [o for o in options if _number(o) == cand]
            if len(hits) == 1:
                return hits[0]
        return None
    hits = [o for o in options if o.lower().startswith(v)]
    return hits[0] if len(hits) == 1 else None


def _user_named(option: str, user_text: str, shown: str | None = None) -> bool:
    """Whether the user's own message names this option ("set buffer to 256",
    "48k", "44100", "slicing"), or what the control displays for it ("beats")."""
    if shown and re.search(rf"\b{re.escape(shown.lower())}\b", user_text):
        return True
    n = _number(option)
    if n is not None:
        forms = {f"{n:g}"} | ({f"{n * 1000:g}"} if "khz" in option.lower() else set())
        return any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d.])", user_text) for f in forms)
    word = _words(re.sub(r"\s*\(.*?\)", "", option))
    return bool(word) and re.search(rf"\b{re.escape(word)}\b", _words(user_text)) is not None


def _words(text: str) -> str:
    """Letters and digits only, so an option's punctuation needn't be typed
    ("flex on smart tempo on" names "Flex On, Smart Tempo On"). Logic
    abbreviates Smart Tempo as "ST" in the longer import-default options."""
    return re.sub(r"\bsmart tempo\b", "st", " ".join(re.findall(r"[a-z0-9]+", text.lower())))


def resolve_choice(value: str, choice: dict, user_text: str,
                    offered_text: str = "") -> tuple[str | None, str | None]:
    """(option to choose, None) or (None, refusal reason for the model)."""
    options = choice.get("options") or []
    ordered = bool(choice.get("ordered"))
    if not options:
        return None, ("this setting's options aren't verified yet -- call it without value to open "
                      "the pane, and let the user pick there")
    if value.lower() in RELATIVE_VALUES:
        if not ordered:
            return None, (f"{value!r} only works on a setting with ordered sizes -- pass one of "
                          f"{', '.join(options)}, or no value to just open the pane")
        return value.lower(), None
    shows = choice.get("shows") or {}
    pick = _match_option(value, options, shows)
    if pick is None:
        # An ambiguous value ("flex on" fits four options): make the reply ask.
        fits = [o for o in options if _words(o).startswith(_words(value))] if _words(value) else []
        if len(fits) > 1:
            return None, (f"{value!r} could be any of: {' | '.join(fits)} -- don't choose one for them: call "
                          "it without value to open the pane, and ask the user which they want, naming "
                          "these exactly")
        return None, (f"{value!r} isn't one of this setting's options ({', '.join(options)}) -- pass one "
                      "of those" + (", or larger / smaller" if ordered else "") + ", or no value to "
                      "open the pane and ask the user which one they want, naming the options exactly")
    if (pick not in (choice.get("model_may_pick") or [])
            and not any(_user_named(pick, t, shows.get(pick)) for t in (user_text, offered_text) if t)):
        alt = "pass larger or smaller" if ordered else "call it without value to open the pane"
        return None, (f"the user didn't name {pick!r} -- don't choose an exact value for them here; "
                      f"{alt}, or ask them which they want")
    return pick, None
