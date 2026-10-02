"""What a turn returns to api/routes/chat.py and the evals."""
from dataclasses import dataclass, field


@dataclass
class Result:
    text: str
    walkthrough_steps: list | None  # the card's steps, in run order
    trace: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # next turn's history
    usage: list[dict] = field(default_factory=list)
    confidence_tier: str = "generic"  # strong / moderate / research / generic
    # Set on a turn parked for research approval; `messages` then ends in the
    # pending tool_use, for the client to hand back with resume=...
    pending_research_query: str | None = None
    sources: list[dict] = field(default_factory=list)  # web_research citations
    auto_run: bool = False  # card may run without Run (client setting permitting)
    card_from_lookup: bool = False  # some step was queued by a citation; never auto-runs
