"""The tool definitions sent to the decider. What each tool is for, and why the action
tools are typed tools rather than KB rows: README.md."""
from app.tools.routes import ROUTES, route_line

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

# No side effect: it makes "this turn makes no claim" a fact in the trace rather
# than something guessed from the reply text.
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

# Executor: app/research. The description is the only trigger; nothing forces a call.
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

# `value` is a string parsed server-side: a number-or-on/off union wasn't held
# by the model even under strict mode.
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
        + "\n".join(route_line(name, route) for name, route in sorted(ROUTES.items()))
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
    # Cache breakpoint for the whole tools array; must sit on the LAST schema.
    "cache_control": {"type": "ephemeral"},
}

TOOLS = [LOOKUP_CONCEPT_SCHEMA, ASK_CLARIFYING_QUESTION_SCHEMA, WEB_RESEARCH_SCHEMA,
         OPEN_PLUGIN_SCHEMA, SET_PARAM_SCHEMA, OPEN_SETTING_SCHEMA]
ACTION_TOOLS = {"open_plugin", "set_param", "open_setting"}
