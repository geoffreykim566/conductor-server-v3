"""Server configuration. Secrets come from the environment (.env.local locally), never the repo.
Why each value is what it is: app/core/README.md."""
import os

CENTRAL_ANTHROPIC_KEY = os.environ.get("CENTRAL_ANTHROPIC_KEY", "")
CONDUCTOR_ID_SECRET = os.environ.get("CONDUCTOR_ID_SECRET", "")
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")
REGISTER_RATE_LIMIT = os.environ.get("REGISTER_RATE_LIMIT", "5/day")

# Trailing-24h spend (event_costs) above this makes /v3/chat return 503.
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "10"))
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://conductor:conductor@localhost:5432/conductor"
)

# The decider. Sonnet 5.5 rejects forced tool_choice; see pipeline/settings.py.
MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 3072
# The writer only phrases already-decided facts, so it runs on a cheaper model.
WRITER_MODEL = "claude-haiku-4-5-20251001"
# web_research's nested call (Sonnet 5 timed out at 150s on half the battery's calls).
RESEARCH_MODEL = "claude-sonnet-5-5"
# 90s timed out every real research call.
RESEARCH_CALL_TIMEOUT_S = float(os.environ.get("RESEARCH_CALL_TIMEOUT_S", "150.0"))

FREE_LIMIT = int(os.environ.get("FREE_LIMIT", "50"))

# /admin basic auth. An unset password locks everyone out.
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

# Cap on round-tripped history (api/history.py trims to this).
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "60"))
