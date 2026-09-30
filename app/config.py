"""v3 configuration. Secrets come from the environment, never the repo.

Trimmed from the real server's config.py — no budget/rate-limit settings yet
(see api.py's module docstring for what's still deferred).
"""
import os

CENTRAL_ANTHROPIC_KEY = os.environ.get("CENTRAL_ANTHROPIC_KEY", "")
CONDUCTOR_ID_SECRET = os.environ.get("CONDUCTOR_ID_SECRET", "")
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")
REGISTER_RATE_LIMIT = os.environ.get("REGISTER_RATE_LIMIT", "5/day")

# Global circuit breaker: once trailing-24h spend (event_costs) crosses this,
# /v3/chat refuses new requests with 503 until spend rolls out of the window.
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "10"))
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://conductor:conductor@localhost:5432/conductor"
)
# Decider model. Sonnet 5.5 (2026-09-28 A/B, log.md): same graded quality as
# Sonnet 5 with forced tool_choice, ~30% faster decider calls. It rejects forced
# tool_choice, so the pipeline asks with auto and re-asks once when the first
# call makes no tool call (pipeline.FORCED_TOOL_CHOICE / RETRY_NO_TOOL_FIRST_CALL).
MODEL = "claude-sonnet-5-5"
# web_research's nested call: Sonnet 5.5 too (2026-09-29, user call). On Sonnet 5
# it timed out at 150 s on 4 of ~8 battery research calls (09-29 merged run).
RESEARCH_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 3072

# web_research tool's own nested Sonnet + web_search call (see app/research.py).
# Started at 90.0 (v1's own default) but raised 2026-09-04: live testing hit
# the 90s ceiling on every one of 3 real research-battery calls in one run
# (max_uses=3, up to 2 pause_turn continuations for a thorough multi-source
# synthesis routinely runs past it) -- a call that times out can never earn
# the trusted "research" confidence tier (see pipeline.py::_confidence_tier),
# so a too-tight budget silently defeats the whole point of trusting it.
RESEARCH_CALL_TIMEOUT_S = float(os.environ.get("RESEARCH_CALL_TIMEOUT_S", "150.0"))

# The "writer" call (see pipeline.py::_write_response) rephrases the decider's
# already-decided facts into user-facing prose with zero tool vocabulary --
# deliberately a separate, cheaper model, since it only phrases, never weighs
# evidence (see v3-log.md 2026-08-22/24).
WRITER_MODEL = "claude-haiku-4-5-20251001"

FREE_LIMIT = int(os.environ.get("FREE_LIMIT", "50"))

# /admin dashboard basic auth. Unset ADMIN_PASSWORD fails closed (see
# admin.py::_require_admin) -- never falls through to comparing against "".
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

# Message-count cap on round-tripped history, the v3 equivalent of the old
# client-side _MAX_CONTEXT_MESSAGES trim -- unbounded history was a known gap
# once /v3/chat started round-tripping it opaquely (see api.py).
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "60"))
