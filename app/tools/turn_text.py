"""Per-turn text the dropdown checks compare against, set by pipeline/respond.py."""
from contextvars import ContextVar

# The current turn's user message, lowercased (pipeline.heuristics.last_user_text).
TURN_USER_TEXT: ContextVar[str] = ContextVar("turn_user_text", default="")
# The reply the user is answering, lowercased: an option it offered counts as the
# user's pick once they answer ("yes", "the last one").
TURN_OFFERED_TEXT: ContextVar[str] = ContextVar("turn_offered_text", default="")
