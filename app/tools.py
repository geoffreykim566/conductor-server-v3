"""v3 tools: lookup_concept (knowledge) and the action tools (open_plugin,
set_param, open_setting) that queue steps onto the reply's card.

lookup_concept queries problems and solutions TOGETHER, in one ranked pass, and
lets whichever is actually closest by distance win — same fix as the v2 pilot's
combined-retrieval bug (a type checked first unconditionally starves a better
match; found live 2026-07-30). If the top hit is a solution that belongs to a
problem bucket, it always resolves as that bucket, never in isolation —
otherwise raw embedding-distance noise can let a bucket member win the top-1
slot and hide its siblings entirely (found live 2026-07-31,
ax_override_critical_test resolving three different ways across three runs).

Since 2026-09-21 the KB holds knowledge only. A solution with a fix carries
the action call it maps to (solutions.action); navigation paths live once, in
seed/routes.json, reached only through open_setting -- never embedded, never
matched against. get_walkthrough is gone: picking a solution is what queues
its action (see pipeline.py's auto-attach).
"""
import asyncio
import json
import re
from contextvars import ContextVar
from pathlib import Path

from app import db
from app.embed import embed
from app.walkthrough import path_to_walkthrough_steps

# Distance bands are advisory only — informing the model's own judgment, not a
# hard filter. The old hard `RELEVANCE_FLOOR` SQL cutoff was a leftover from v1,
# inconsistent with this project's own design direction (leave judgment calls to
# the model, not a hand-tuned number) and directly implicated in a real bug: it
# admitted a different wrong match on almost every retried phrasing for a topic
# with no real KB coverage, which kept the model retrying instead of ever
# cleanly recognizing "nothing relevant here." Found live 2026-07-30.
_STRONG_MATCH = 0.40
_MODERATE_MATCH = 0.60


def _confidence_label(distance: float) -> str:
    if distance <= _STRONG_MATCH:
        return "strong"
    if distance <= _MODERATE_MATCH:
        return "moderate — treat with more skepticism, consider asking or trying a different phrasing"
    return "weak — likely not actually relevant; don't treat this as a real match"


def _confidence_short(distance: float) -> str:
    """Plain strong/moderate/weak label for query_log — _confidence_label's
    return value is the model-facing advisory sentence, not a clean value to
    group/filter on."""
    if distance <= _STRONG_MATCH:
        return "strong"
    if distance <= _MODERATE_MATCH:
        return "moderate"
    return "weak"


async def _embed_with_retry(texts: list[str], input_type: str) -> list[list[float]]:
    """Voyage rate-limits on bursty test runs — retry/backoff rather than letting
    one transient 429 kill an entire scenario battery run. embed.py now also
    proactively spaces calls to avoid triggering 429s in the first place; this is
    the safety net for whatever gets through anyway. Widened 2026-08-06 after two
    full-battery runs both exhausted the old 4-attempt/60s-total budget and died
    mid-run -- 6 attempts, longer steps, ~4.5min total budget before giving up."""
    for attempt in range(6):
        try:
            return await embed(texts, input_type=input_type)
        except Exception as exc:
            if attempt == 5:
                raise
            wait = 15 * (attempt + 1)
            print(f"  [retry] embed() failed ({exc}); retrying in {wait}s")
            await asyncio.sleep(wait)

LOOKUP_CONCEPT_SCHEMA = {
    "name": "lookup_concept",
    "description": (
        "Call this before diagnosing any Logic Pro problem or naming any menu path, "
        "shortcut, or settings location — never state one from memory. Results include "
        "a match_confidence ('strong'/'moderate'/'weak') — this is advisory, not a "
        "filter; judge for yourself whether a 'moderate' or 'weak' result is actually "
        "relevant rather than treating it as grounded. If you already know the exact "
        "name of a specific setting or destination (from a solution's own fix text, "
        "from a bucket you were just given, or from your own knowledge), query with "
        "that name directly rather than paraphrasing the user's original wording."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "problem": {
                "type": "string",
                "description": (
                    "Your own short (2-8 word) query — either a normalized description of the "
                    "user's symptom (not their literal wording; recognize the underlying issue "
                    "and phrase it the way this KB names things), or the exact name of a "
                    "specific setting/destination you already know you want."
                ),
            }
        },
        "required": ["problem"],
        "additionalProperties": False,
    },
}


# Pure structural signal, no side effect -- exists so "this turn makes no
# claim" is a hard fact in trace, not something inferred from response text.
# Found live 2026-09-02 (Fable review of the writer-split battery): the
# deterministic hedge gate can't currently tell a genuine claim-free
# clarifying question (muddy_ambiguous, thin_hollow_boundary) apart from an
# ungrounded claim -- both look identical at the trace level (a tool call
# happened, nothing attached, no strong grounding). A text heuristic ("ends
# with a question mark") was considered and rejected: it would wrongly
# suppress the hedge on hedge_indirect_beat_from_scratch, whose entire
# response is confident unhedged claims that also happen to end with a
# trailing offer to go deeper. Same reasoning as every other ambiguous-
# behavior fix in this file (commit-to-one, backfill, tool-skip): make the
# model take an explicit, code-checkable action instead of inferring intent
# from prose.
ASK_CLARIFYING_QUESTION_SCHEMA = {
    "name": "ask_clarifying_question",
    "description": (
        "Call this when your entire response for this turn is a clarifying question "
        "and nothing else -- no diagnosis, guidance, or claim alongside it. Marks the "
        "turn as making no claim, so it won't be hedged as an unconfirmed answer. Only "
        "for the genuine-toss-up case this prompt already describes (a 'problem' result "
        "with no distinguishing evidence yet). Never call this if you're also offering "
        "any guidance, even tentative guidance, in the same response -- that response "
        "should stand as a real (possibly hedged) answer, not a claim-free question. "
        "Never call this alongside an action tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

# Executor lives in app/research.py (its own nested Sonnet + web_search call,
# not a DB lookup like the tools above) -- kept out of this file the same way
# v1 split router.py's glue from research.py's actual call. Model-decided
# trigger, not a deterministic gate: the description below is the only thing
# telling the model when to call this (v3-log.md 2026-09-04 scoping note) --
# a code-forced call on every weak/no-hit lookup_concept result was
# considered and rejected as the costlier, more eager option.
WEB_RESEARCH_SCHEMA = {
    "name": "web_research",
    "description": (
        "Search the web for information outside this KB's coverage -- specific artist/producer "
        "techniques, gear, current Logic Pro features/changes, or anything else lookup_concept has "
        "no real match for (its match_confidence came back 'weak' or there was no match at all). "
        "Call this instead of answering from pretrained knowledge when a question asks for "
        "something you'd otherwise have to guess at. Don't call it for ordinary troubleshooting or "
        "navigation questions lookup_concept already covers -- this is for genuinely out-of-KB "
        "information, not a first resort."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused web-search query capturing exactly what needs researching.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

# Typed action tools (2026-09-21). An action the client runs on the user's Mac
# when they press Run -- not a KB row reached by embedding distance. On v040
# these were template rows ("open plugin" / "set plugin parameter") behind
# lookup_concept + get_walkthrough(args): the model had to FIND them before it
# could call them, their two-word names collided with plugin vocabulary
# ("plugin manager", "multipressor"), and it reached them by name from the
# prompt anyway (agentic-design-review-2026-09-20.md). As tools, the model
# picks them directly, no lookup first; the wire step shapes are unchanged, so
# the client executor is untouched.
#
# `value` is a string, parsed server-side (_parse_param_value): an anyOf
# number-or-"on"/"off" union was probed under strict mode on 2026-09-21 and
# the model still returned "-18" as a string, i.e. the union wasn't held.
OPEN_PLUGIN_SCHEMA = {
    "name": "open_plugin",
    "description": (
        "Queue an action that loads an audio-effect plugin onto a track by name, or opens its "
        "window if that plugin is already on the track. It runs on the user's Mac after your "
        "reply, when they press Run -- it has NOT happened when you write your reply. Works for "
        "any installed plugin, Apple or third-party, whether or not the KB mentions it. Use it "
        "for a direct request to add / put / load / open a plugin; no lookup_concept call is "
        "needed first. Several action calls in one reply run in call order as one card."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "plugin": {
                "type": "string",
                "description": "Plugin name as Logic lists it (Channel EQ, Compressor, Space Designer, Phat FX, ValhallaSupermassive).",
            },
            "track": {
                "type": "string",
                "description": "Track name from the live state, only when the user names or clearly means one; omit for the selected track.",
            },
            "new_instance": {
                "type": "boolean",
                "description": "true only when the user clearly wants a second copy of a plugin that's already on the track.",
            },
        },
        "required": ["plugin"],
        "additionalProperties": False,
    },
}

SET_PARAM_SCHEMA = {
    "name": "set_param",
    "description": (
        "Queue an action that sets one control of a plugin on a track to an exact value. It runs "
        "on the user's Mac after your reply, when they press Run -- it has NOT happened when you "
        "write your reply. The plugin must be on the track: if the live state doesn't show it, "
        "call open_plugin for it first in the same reply. Setting a value on a band that's "
        "switched off (an EQ's Low Cut, say) switches the band on as part of the same action -- "
        "no separate call or question needed."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "plugin": {"type": "string", "description": "Plugin name, as for open_plugin."},
            "param": {
                "type": "string",
                "description": "The control's label as the plugin shows it (Low Cut Frequency, Threshold, Ratio, Mix).",
            },
            "value": {
                "type": "string",
                "description": "A plain number in the control's displayed unit (\"80\", \"-18\", \"4\"), or \"on\" / \"off\" for a switch.",
            },
            "track": {
                "type": "string",
                "description": "Track name from the live state, only when the user names or clearly means one; omit for the selected track.",
            },
        },
        "required": ["plugin", "param", "value"],
        "additionalProperties": False,
    },
}

# Approved navigation routes (2026-09-21): name -> verified path, loaded once.
# The model only ever names a route; the path it runs is always this file's.
# Model-written menu paths are the failure this codebase has hit most
# (confidently wrong, and wrong in ways the user can't catch).
ROUTES: dict[str, dict] = json.loads(
    (Path(__file__).parent.parent / "seed" / "routes.json").read_text()
)

# A route that ends on a dropdown carries a `choice` block: the dropdown's
# options, whether they're ordered (so larger/smaller means something), and
# which options the model may pick on its own from a diagnosis -- any other
# value has to be in the user's own message, checked here in code. Found live
# 2026-09-22: these routes used to end by opening the dropdown and stopping,
# and the open menu swallowed the next step's keystrokes ("open buffer size
# then open compressor": the plugin search never got focus). Now a dropdown
# route either picks a value or stops at the pane -- it never leaves a menu
# open. An empty `options` list means not verified yet: pane only. `shows`
# maps an option to the shorter text the control displays once picked
# (Region Smart Tempo: "On + Align Bars" reads "Bars"), for the client's read-back.
RELATIVE_VALUES = ("larger", "smaller")

# The current turn's user message (lowercased, see pipeline._last_user_text),
# set by pipeline._respond -- what "the user named this value" is checked against.
TURN_USER_TEXT: ContextVar[str] = ContextVar("turn_user_text", default="")
# The reply the user is answering (lowercased): an option it offered counts as
# the user's pick once they answer ("yes", "the last one") -- they shouldn't have
# to type the option out (user call 2026-09-28). Set with TURN_USER_TEXT.
TURN_OFFERED_TEXT: ContextVar[str] = ContextVar("turn_offered_text", default="")


def _route_line(name: str, route: dict) -> str:
    line = f"- {name}: {route['desc']}"
    choice = route.get("choice")
    if choice is None:
        return line
    options = choice.get("options") or []
    if not options:
        return line + " [opens the pane only; no value]"
    values = " | ".join(options) + (" | larger | smaller" if choice.get("ordered") else "")
    free = choice.get("model_may_pick") or []
    who = (f"you may choose {' or '.join(free)} yourself; any other value only if the user named it"
           if free else "an exact value only if the user named it")
    if choice.get("ordered"):
        who += "; larger/smaller move one step from its current value"
    return line + f" [value: {values} -- {who}]"


OPEN_SETTING_SCHEMA = {
    "name": "open_setting",
    "description": (
        "Queue an action that takes the user to a Logic Pro setting, window or feature by one of "
        "the approved routes below -- the route's verified steps run on the user's Mac after your "
        "reply, when they press Run (it has NOT happened when you write your reply). Use it for a "
        "direct request to open or go to one of these; name must be one of the routes listed. If "
        "what the user wants isn't listed, don't pick the nearest one -- say you don't have a "
        "verified route for it. A route marked [value: ...] ends on a dropdown: pass value to "
        "choose one of its options, or omit it to open the pane with the current value showing "
        "and let the user pick.\n\nRoutes:\n"
        + "\n".join(_route_line(name, route) for name, route in sorted(ROUTES.items()))
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": sorted(ROUTES), "description": "One of the approved routes."},
            "value": {
                "type": "string",
                "description": "Only for a route marked [value: ...]: one of its listed options, or "
                               "larger / smaller where listed. Omit to just open the pane.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    # Cache breakpoint for the whole tools array -- every schema is static
    # across every call in a run, but gets resent unchanged on every loop
    # iteration and every turn otherwise (measured 2026-08-05: 66 calls,
    # 207,869 uncached input tokens across a 24-scenario battery). Must sit
    # on the LAST schema in the array for the cache breakpoint to cover all
    # of them.
    "cache_control": {"type": "ephemeral"},
}

TOOLS = [LOOKUP_CONCEPT_SCHEMA, ASK_CLARIFYING_QUESTION_SCHEMA, WEB_RESEARCH_SCHEMA,
         OPEN_PLUGIN_SCHEMA, SET_PARAM_SCHEMA, OPEN_SETTING_SCHEMA]
ACTION_TOOLS = {"open_plugin", "set_param", "open_setting"}


def _content_summary(name: str, content: dict) -> str:
    lines = [name]
    for key, label in (
        ("symptom", "Symptom"), ("cause", "Cause"), ("fix", "Fix"),
        ("zone", "Zone"), ("remove_when", "Remove when"), ("remove_move", "Remove move"),
        ("add_when", "Add when"), ("add_move", "Add move"), ("do_not", "Do not"),
        ("opens", "Opens"),
    ):
        if content.get(key):
            lines.append(f"{label}: {content[key]}")
    return "\n".join(lines)


async def lookup_concept(problem: str, _vec: list[float] | None = None) -> dict:
    # _vec lets a caller that already embedded this exact text (probe_lookup.py's
    # raw-top8 pass) skip a second, redundant Voyage call for the same query --
    # found live 2026-08-18, probe runs were double-embedding every query and
    # hitting far more 429s than the battery ever does for the same query count.
    # Never passed by the real pipeline (model tool calls always start from raw text).
    vec = _vec if _vec is not None else (await _embed_with_retry([problem], input_type="query"))[0]
    rows = await db.pool().fetch(
        """
        select * from (
            select 'problem'::text as kind, id, name, null::jsonb as content,
                   null::jsonb as action, note,
                   (embedding <=> $1) as distance
            from problems
            union all
            select 'solution'::text as kind, id, name, content,
                   action, null::text as note,
                   (embedding <=> $1) as distance
            from solutions
        ) combined
        order by distance
        limit 5
        """,
        vec,
    )
    top_results = [
        {"kind": r["kind"], "name": r["name"], "distance": float(r["distance"])}
        for r in rows
    ]
    if not rows:
        await db.insert_query_log(query=problem, confidence="none", top_results=[])
        return {"match": "none"}

    top = rows[0]
    confidence = _confidence_label(top["distance"])
    await db.insert_query_log(
        query=problem, confidence=_confidence_short(top["distance"]), top_results=top_results
    )

    problem_id = problem_name = problem_note = None
    if top["kind"] == "problem":
        problem_id, problem_name, problem_note = top["id"], top["name"], top["note"]
    else:
        # A solution can belong to more than one bucket (e.g. `sample rate
        # mismatch` is a candidate for both `song sounds slowed` and
        # `crackling during playback`) -- when the top hit is the solution
        # itself rather than either parent, pick deterministically by which
        # bucket it's most strongly weighted in, not by unordered row order.
        link = await db.pool().fetchrow(
            """
            select p.id, p.name, p.note
            from problem_solutions ps
            join problems p on p.id = ps.problem_id
            where ps.solution_id = $1
            order by ps.seed_weight desc nulls last
            limit 1
            """,
            top["id"],
        )
        if link:
            problem_id, problem_name, problem_note = link["id"], link["name"], link["note"]

    if problem_id is not None:
        links = await db.pool().fetch(
            """
            select s.name, s.content, s.action, ps.seed_weight, ps.distinguisher
            from problem_solutions ps
            join solutions s on s.id = ps.solution_id
            where ps.problem_id = $1
            order by ps.seed_weight desc nulls last
            """,
            problem_id,
        )
        solutions = [
            {
                "name": r["name"],
                "seed_weight": r["seed_weight"],
                "distinguisher": r["distinguisher"],
                "summary": _content_summary(r["name"], r["content"] or {}),
                "action": r["action"],
            }
            for r in links
        ]
        return {
            "match": "problem",
            "problem": problem_name,
            "match_confidence": confidence,
            "solutions": solutions,
            "note": problem_note or "seed_weight is a population prior — override it on direct evidence.",
        }

    return {
        "match": "single",
        "problem": top["name"],
        "match_confidence": confidence,
        "solutions": [{
            "name": top["name"],
            "seed_weight": None,
            "distinguisher": None,
            "summary": _content_summary(top["name"], top["content"] or {}),
            "action": top["action"],
        }],
        "note": None,
    }


def _is_truthy(value) -> bool:
    """Real (non-fixture) AX data may not always arrive as a JSON boolean --
    accept common truthy shapes rather than silently bypassing the toggle
    gate on them. Found via adversarial review, 2026-08-05: the original
    strict `is True` check would pass right through a "true"/1/"yes" value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, int):
        return value == 1
    return False


_MAX_NAME = 64
_ON = ("on", "true", "enable", "enabled", "yes")
_OFF = ("off", "false", "disable", "disabled", "no")
_NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:[a-z%]+)?\s*$", re.IGNORECASE)


def _clean_name(value, what: str) -> tuple[str | None, str | None]:
    name = str(value or "").strip()
    if not name:
        return None, f"{what} is empty"
    if len(name) > _MAX_NAME:
        return None, f"{what} is too long to be a {what} name"
    return name, None


def _parse_param_value(value) -> str | None:
    """'on'/'off' for a switch, else the plain number as a string ("80hz" ->
    "80", "-18 dB" -> "-18", "4:1" is not parsed). None if it's neither --
    the client's writers take float(value) or an on/off word, nothing else."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    text = str(value or "").strip().lower()
    if text in _ON:
        return "on"
    if text in _OFF:
        return "off"
    m = _NUMBER.match(text)
    return m.group(1) if m else None


def queue_open_plugin(inp: dict) -> dict:
    """Executor for open_plugin: validate, then emit the client's wire step.
    Pure -- the action runs client-side when the user presses Run."""
    plugin, err = _clean_name(inp.get("plugin"), "plugin")
    if err:
        return {"attached": False, "reason": err}
    step = {"ax_open_plugin": plugin}
    if inp.get("new_instance") is True:
        step["new"] = True
    if inp.get("track"):
        track, err = _clean_name(inp["track"], "track")
        if err:
            return {"attached": False, "reason": err}
        step["track"] = track
    return {"attached": True, "action": "open_plugin", "steps": [step]}


def queue_set_param(inp: dict) -> dict:
    """Executor for set_param: validate, then emit the client's wire step."""
    plugin, err = _clean_name(inp.get("plugin"), "plugin")
    if err:
        return {"attached": False, "reason": err}
    param, err = _clean_name(inp.get("param"), "param")
    if err:
        return {"attached": False, "reason": err}
    value = _parse_param_value(inp.get("value"))
    if value is None:
        return {
            "attached": False,
            "reason": (f"value {inp.get('value')!r} isn't a plain number or on/off -- pass the "
                       "number in the control's displayed unit (e.g. \"80\", \"-18\") or \"on\"/\"off\""),
        }
    spec = {"plugin": plugin, "param": param, "value": value}
    if inp.get("track"):
        track, err = _clean_name(inp["track"], "track")
        if err:
            return {"attached": False, "reason": err}
        spec["track"] = track
    return {"attached": True, "action": "set_param", "steps": [{"ax_set_param": spec}]}


def queue_open_setting(inp: dict, ax_fixture: dict | None = None) -> dict:
    """Executor for open_setting: an approved route's verified steps."""
    name = str(inp.get("name") or "").strip()
    route = ROUTES.get(name)
    if route is None:
        return {"attached": False, "reason": f"{name!r} isn't an approved route -- say you don't have a verified route for it"}
    # Toggle gate (moved here from the solution row, 2026-08-05): the route
    # flips something the live state says is already in its target state, so
    # running it would switch it AWAY. Refused rather than attached.
    key = route.get("toggle_ax_key")
    if key and ax_fixture and _is_truthy(ax_fixture.get(key)):
        return {
            "attached": False,
            "reason": (f"already in the target state ({key} is already true this turn) -- running this "
                       "would switch it away; tell the user it's already there instead"),
        }
    steps = path_to_walkthrough_steps(route["path"])
    if not steps:
        return {"attached": False, "reason": "route has no executable steps"}
    choice = route.get("choice")
    value = str(inp.get("value") or "").strip()
    chosen = None
    if value and choice is None:
        return {"attached": False, "reason": f"{name!r} has no value to choose -- call it without value"}
    if choice is not None:
        if not value:
            dropped, steps = steps[-1], steps[:-1]   # the pane only: never click the dropdown open
            # The dropped row is the kept last step's `expect`. A disclosure click
            # (Region Inspector's "Region") toggles: the row showing means it's
            # already open, so the client skips the click. A menu: the client
            # verifies on the row, not the tab name ("Audio" matched the Project
            # Settings window once), and skips the menu when the row already shows.
            if steps and "click_value_of" in dropped and ("click_text" in steps[-1] or "menu_path" in steps[-1]):
                steps[-1] = {**steps[-1], "expect": [dropped["click_value_of"]]}
        else:
            chosen, reason = _resolve_choice(value, choice, TURN_USER_TEXT.get(), TURN_OFFERED_TEXT.get())
            if chosen is None:
                return {"attached": False, "reason": reason}
            # `reopen`: the route's own steps, so the client's revert can get
            # back to this dropdown and pick the old value again.
            pick = {"choose": chosen, "reopen": list(steps)}
            if choice.get("shows"):   # the control's shorter display, for the read-back
                pick["shows"] = dict(choice["shows"])
            steps = steps + [pick]
    # `destination` keeps the result shape pipeline._backfill_walkthrough
    # recognises (attached + destination + steps).
    out = {"attached": True, "action": "open_setting", "destination": name, "steps": steps}
    if chosen:
        out["chooses"] = chosen
    elif choice is not None:
        out["note"] = "opens the pane only -- no value is chosen; " + pane_only_ask(choice)
    return out


def pane_only_ask(choice: dict) -> str:
    """What the reply does when a dropdown route stops at the pane: ask which
    option, naming them exactly, so the user's answer can set it on the next
    card (TURN_OFFERED_TEXT) -- user call 2026-09-29. A route with fewer than
    two verified options (threads) can't be asked about: the user picks there."""
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


def _resolve_choice(value: str, choice: dict, user_text: str,
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
        # "flex on" fits four import-default options: say so, so the reply asks
        # instead of naming an option that doesn't exist (found live 2026-09-29).
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


def action_calls(action) -> list[tuple[str, dict]]:
    """A solution's stored action ({"open_setting": "x"} / {"open_plugin": {...}}
    / a list of those) as (tool name, tool input) pairs -- the same shape a
    model's own call would have, so an auto-attached action and a direct call
    are indistinguishable downstream."""
    out = []
    for call in (action if isinstance(action, list) else [action] if action else []):
        for tool, arg in call.items():
            if tool == "open_setting":
                out.append((tool, {"name": arg}))
            elif tool in ACTION_TOOLS and isinstance(arg, dict):
                out.append((tool, dict(arg)))
    return out
