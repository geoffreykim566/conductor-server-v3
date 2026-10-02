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

LOOKUP_CAP_ERROR = (
    "Too many lookups without a clear answer. "
    "Stop searching now — answer from general Logic Pro knowledge if you're "
    "genuinely confident, or tell the user you don't have a verified answer "
    "for this. Do not call lookup_concept again this turn."
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
    "A lookup this turn returned solutions that map to an action. If your answer "
    "recommends one, call its action so the user gets it on the card; if you can't tell "
    "which applies, ask the one question that would decide it (ask_clarifying_question); "
    "if the match isn't actually what they asked about, look up something more specific. "
    "If the candidate you recommend has no action of its own, call lookup_concept with its "
    "name -- don't queue a different candidate's action instead."
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
