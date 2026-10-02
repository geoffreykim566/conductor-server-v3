"""Fixed text the pipeline hands the decider: tool-result refusals, card notes and
nudges. Code decides when each applies; this file only holds the wording."""
from app.pipeline import settings

# In place of a web_research call the user declined.
RESEARCH_DECLINED = {
    "error": (
        "The user declined web research for this turn. Do not call web_research "
        "again this turn. Answer from general Logic Pro knowledge if you can, "
        "framed clearly as unverified general guidance, not a confirmed answer -- "
        "or say plainly that you don't have a verified answer for this."
    )
}

TOOL_NUDGE = (
    "Call a tool before you answer: the action the user just agreed to (each turn gets its own "
    "card, so queue it again even if an earlier reply offered it), or ask_clarifying_question "
    "if you can't tell what they want run."
)

ON_CARD_NOTE = (
    "This solution's action is already on the reply's card -- it waits for the user "
    "to press Run. Don't call it again, and don't describe it as done."
)
PANE_ONLY_NOTE = (
    " It opens the setting's pane only, with no value chosen. If your answer recommends a "
    "value it can take, call open_setting for the same route with that value -- that "
    "replaces the queued step rather than adding one. Otherwise ask the user which option "
    "they want, naming the options exactly."
)
PICK_NUDGE = (
    "You cited a problem whose candidate fixes map to actions. If your answer "
    "recommends one, call its action so the user gets it on the card; if you can't tell "
    "which applies, ask the one question that would decide it (ask_clarifying_question). "
    "If the candidate you recommend has no action of its own, cite it with cite_kb "
    "-- don't queue a different candidate's action instead."
)
FINAL_ANSWER_NUDGE = "\n\nAnswer now with your best available information — no more tool calls."

ATTACH_CAP_REFUSAL = (
    f"this turn already has {settings.MAX_ATTACHES_PER_TURN} steps queued, the limit for one "
    "card. Tell the user what's queued and that you'll do the rest when they ask next."
)
DUPLICATE_ATTACH_REFUSAL = "this exact step is already queued this turn -- it runs once."
FALLBACK_REFUSAL = (
    "another candidate for this same problem is already on the card this turn -- "
    "this is an alternative fix, not a second request. Mention it in prose as the "
    "next thing to try, and only queue it if the user comes back and says the "
    "first one didn't work."
)
