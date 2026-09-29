"""Postgres access — the only module that runs SQL connection setup."""
import asyncio
import json
import uuid

import asyncpg
import pgvector.asyncpg

from app.config import DATABASE_URL, FREE_LIMIT

_pool: asyncpg.Pool | None = None


async def _init_conn(conn: asyncpg.Connection) -> None:
    await pgvector.asyncpg.register_vector(conn)
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def connect(retries: int = 30, delay: float = 1.0) -> None:
    global _pool
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            _pool = await asyncpg.create_pool(
                DATABASE_URL, min_size=1, max_size=10, init=_init_conn
            )
            return
        except (OSError, asyncpg.PostgresError) as e:
            last_err = e
            await asyncio.sleep(delay)
    raise RuntimeError(f"could not connect to Postgres after {retries} tries: {last_err}")


async def disconnect() -> None:
    if _pool is not None:
        await _pool.close()


def pool() -> asyncpg.Pool:
    assert _pool is not None, "DB pool not initialised"
    return _pool


async def get_or_create_user(conductor_id: uuid.UUID) -> asyncpg.Record:
    """Upsert the user by id, bump last_seen, return the full row.

    conductor_id is already verified by the time it reaches here (see
    deps.py's current_user, which checks the HMAC signature first) — the
    upsert just lets a row lost to a DB reset self-heal.
    """
    return await pool().fetchrow(
        """
        insert into users (id, free_limit)
        values ($1, $2)
        on conflict (id) do update set last_seen_at = now()
        returning *
        """,
        conductor_id,
        FREE_LIMIT,
    )


async def claim_free_message(user_id: uuid.UUID) -> asyncpg.Record | None:
    """Atomically consume one free-tier message; None once the cap is spent."""
    return await pool().fetchrow(
        """
        update users set free_used = free_used + 1
        where id = $1 and free_used < free_limit
        returning *
        """,
        user_id,
    )


async def insert_event(
    *,
    user_id: uuid.UUID,
    tokens_in: int,
    tokens_out: int,
    model: str | None = None,
    source_tier: str | None = None,
    latency_ms: int | None = None,
    prompt: str | None = None,
    response: str | None = None,
) -> uuid.UUID:
    row = await pool().fetchrow(
        """
        insert into events (user_id, tokens_in, tokens_out,
                             model, source_tier, latency_ms, prompt, response)
        values ($1, $2, $3, $4, $5, $6, $7, $8)
        returning id
        """,
        user_id, tokens_in, tokens_out, model, source_tier, latency_ms, prompt, response,
    )
    return row["id"]


async def spend_last_24h_usd() -> float:
    """Trailing-24h spend across all users, from the derived event_costs view."""
    return float(await pool().fetchval(
        """
        select coalesce(sum(usd_cost), 0)
        from event_costs
        where created_at > now() - interval '24 hours'
        """
    ))


async def set_rating(event_id: uuid.UUID, user_id: uuid.UUID, rating: int) -> bool:
    """rating is 1 (up), -1 (down), or 0 (undo). Returns False if no such event
    belongs to this user."""
    result = await pool().execute(
        """
        update events
           set rating   = nullif($1, 0),
               rated_at = case when $1 = 0 then null else now() end
         where id = $2 and user_id = $3
        """,
        rating, event_id, user_id,
    )
    return result != "UPDATE 0"


async def update_profile(user_id: uuid.UUID, experience: str | None, role: str | None) -> None:
    await pool().execute(
        """
        update users set
            experience = coalesce($2, experience),
            role       = coalesce($3, role)
        where id = $1
        """,
        user_id, experience, role,
    )


async def mark_uninstalled(user_id: uuid.UUID) -> None:
    await pool().execute(
        "update users set uninstalled_at = now() where id = $1 and uninstalled_at is null",
        user_id,
    )
