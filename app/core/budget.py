"""Daily-spend circuit breaker: /v3/chat returns 503 while trailing-24h spend is over
DAILY_BUDGET_USD. Trips before the console cap would take the key down. See README.md."""
import time

from app.core import db
from app.core.config import DAILY_BUDGET_USD

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
