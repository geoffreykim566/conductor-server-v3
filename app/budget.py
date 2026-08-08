"""Global daily-spend circuit breaker for the free tier.

Ported from server/app/budget.py (v1), same design: the console spend cap on
the central key is the hard backstop, but tripping it takes the key -- and
the whole beta -- down until it resets. This breaker trips first: when the
trailing 24h of event_costs crosses DAILY_BUDGET_USD, chat is refused with
503 and resumes on its own as spend rolls out of the window. Cached so it
costs one aggregate query per minute, not per message.
"""
import time

from app import db
from app.config import DAILY_BUDGET_USD

_CACHE_TTL_S = 60.0
_checked_at: float = 0.0  # monotonic; 0 forces a check on first request
_spent_usd: float = 0.0


async def over_daily_budget() -> bool:
    global _checked_at, _spent_usd
    now = time.monotonic()
    if _checked_at == 0.0 or now - _checked_at > _CACHE_TTL_S:
        _spent_usd = await db.spend_last_24h_usd()
        _checked_at = now
    return _spent_usd >= DAILY_BUDGET_USD
