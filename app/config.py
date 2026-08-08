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
# /v1/chat refuses new requests with 503 until spend rolls out of the window.
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "10"))
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://conductor:conductor@localhost:5432/conductor"
)
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 3072

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
VOYAGE_MODEL = "voyage-3.5"  # 1024-dim; index and query must stay symmetric

# Cosine distance ceiling for retrieval — same calibrated value as the real server
# (raised 0.5 -> 0.65, see kb-log.md 2026-07-28), kept here for continuity, not
# re-derived from scratch for this tiny seed set.
RELEVANCE_FLOOR = float(os.environ.get("RELEVANCE_FLOOR", "0.65"))

FREE_LIMIT = int(os.environ.get("FREE_LIMIT", "50"))

# Message-count cap on round-tripped history, the v3 equivalent of the old
# client-side _MAX_CONTEXT_MESSAGES trim -- unbounded history was a known gap
# once /v1/chat started round-tripping it opaquely (see api.py).
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "60"))
