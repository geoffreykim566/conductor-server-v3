"""over_daily_budget: the spend threshold and its 60s cache (db spend query mocked)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from app.core import budget


def _reset_cache() -> None:
    budget._checked_at = 0.0
    budget._spent_usd = 0.0


async def test_under_budget_returns_false() -> None:
    _reset_cache()
    with patch("app.core.budget.DAILY_BUDGET_USD", 10.0), \
         patch("app.core.budget.db.spend_last_24h_usd", AsyncMock(return_value=5.0)):
        assert await budget.over_daily_budget() is False, "spend below the cap must not trip the breaker"


async def test_at_or_over_budget_returns_true() -> None:
    _reset_cache()
    with patch("app.core.budget.DAILY_BUDGET_USD", 10.0), \
         patch("app.core.budget.db.spend_last_24h_usd", AsyncMock(return_value=10.0)):
        assert await budget.over_daily_budget() is True, "spend at the cap (>=, not >) must trip the breaker"


async def test_result_cached_within_ttl() -> None:
    """A second call inside the 60s TTL must reuse the cached value, not
    re-query -- the whole point of caching is one aggregate query per minute,
    not per message (see budget.py's own docstring)."""
    _reset_cache()
    spend_mock = AsyncMock(return_value=5.0)
    with patch("app.core.budget.DAILY_BUDGET_USD", 10.0), \
         patch("app.core.budget.db.spend_last_24h_usd", spend_mock), \
         patch("app.core.budget.time.monotonic", side_effect=[100.0, 105.0]):
        first = await budget.over_daily_budget()
        spend_mock.return_value = 20.0  # would trip the breaker if actually re-queried
        second = await budget.over_daily_budget()
    assert first is False and second is False, "the cached (stale) value must be reused within the TTL"
    assert spend_mock.await_count == 1, f"expected exactly one real query within the TTL, got {spend_mock.await_count}"


async def test_cache_expires_after_ttl() -> None:
    _reset_cache()
    spend_mock = AsyncMock(side_effect=[5.0, 20.0])
    with patch("app.core.budget.DAILY_BUDGET_USD", 10.0), \
         patch("app.core.budget.db.spend_last_24h_usd", spend_mock), \
         patch("app.core.budget.time.monotonic", side_effect=[100.0, 200.0]):
        first = await budget.over_daily_budget()
        second = await budget.over_daily_budget()
    assert first is False, "first check should reflect the initial (under-budget) spend"
    assert second is True, "a call past the TTL must re-query and reflect the new (over-budget) spend"
    assert spend_mock.await_count == 2, f"expected a fresh query after the TTL elapsed, got {spend_mock.await_count}"
