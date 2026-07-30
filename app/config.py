"""v3 configuration. Secrets come from the environment, never the repo.

Trimmed from the real server's config.py — no free-tier/budget/rate-limit
settings, no HTTP-facing config at all, since this pipeline is client-less
(direct calls, no FastAPI route) for now.
"""
import os

CENTRAL_ANTHROPIC_KEY = os.environ.get("CENTRAL_ANTHROPIC_KEY", "")
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
