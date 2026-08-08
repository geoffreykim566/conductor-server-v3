"""Unit tests for budget.py's over_daily_budget -- threshold + 60s cache TTL.
db.spend_last_24h_usd is mocked, no real DB connection needed. Run inside the
app container:
    docker compose exec app python -m app.test_budget
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app import budget


def _reset_cache() -> None:
    budget._checked_at = 0.0
    budget._spent_usd = 0.0


async def test_under_budget_returns_false() -> None:
    _reset_cache()
    with patch("app.budget.DAILY_BUDGET_USD", 10.0), \
         patch("app.budget.db.spend_last_24h_usd", AsyncMock(return_value=5.0)):
        assert await budget.over_daily_budget() is False, "spend below the cap must not trip the breaker"
    print("PASS: spend below DAILY_BUDGET_USD does not trip the breaker.")


async def test_at_or_over_budget_returns_true() -> None:
    _reset_cache()
    with patch("app.budget.DAILY_BUDGET_USD", 10.0), \
         patch("app.budget.db.spend_last_24h_usd", AsyncMock(return_value=10.0)):
        assert await budget.over_daily_budget() is True, "spend at the cap (>=, not >) must trip the breaker"
    print("PASS: spend at (not just above) the cap trips the breaker.")


async def test_result_cached_within_ttl() -> None:
    """A second call inside the 60s TTL must reuse the cached value, not
    re-query -- the whole point of caching is one aggregate query per minute,
    not per message (see budget.py's own docstring)."""
    _reset_cache()
    spend_mock = AsyncMock(return_value=5.0)
    with patch("app.budget.DAILY_BUDGET_USD", 10.0), \
         patch("app.budget.db.spend_last_24h_usd", spend_mock), \
         patch("app.budget.time.monotonic", side_effect=[100.0, 105.0]):
        first = await budget.over_daily_budget()
        spend_mock.return_value = 20.0  # would trip the breaker if actually re-queried
        second = await budget.over_daily_budget()
    assert first is False and second is False, "the cached (stale) value must be reused within the TTL"
    assert spend_mock.await_count == 1, f"expected exactly one real query within the TTL, got {spend_mock.await_count}"
    print("PASS: a second call within the 60s TTL reuses the cached value.")


async def test_cache_expires_after_ttl() -> None:
    _reset_cache()
    spend_mock = AsyncMock(side_effect=[5.0, 20.0])
    with patch("app.budget.DAILY_BUDGET_USD", 10.0), \
         patch("app.budget.db.spend_last_24h_usd", spend_mock), \
         patch("app.budget.time.monotonic", side_effect=[100.0, 200.0]):
        first = await budget.over_daily_budget()
        second = await budget.over_daily_budget()
    assert first is False, "first check should reflect the initial (under-budget) spend"
    assert second is True, "a call past the TTL must re-query and reflect the new (over-budget) spend"
    assert spend_mock.await_count == 2, f"expected a fresh query after the TTL elapsed, got {spend_mock.await_count}"
    print("PASS: the cache expires after 60s and re-queries real spend.")


async def main() -> None:
    await test_under_budget_returns_false()
    await test_at_or_over_budget_returns_true()
    await test_result_cached_within_ttl()
    await test_cache_expires_after_ttl()


if __name__ == "__main__":
    asyncio.run(main())
