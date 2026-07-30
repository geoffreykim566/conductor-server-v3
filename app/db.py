"""Postgres access — the only module that runs SQL connection setup.

Trimmed from the real server's db.py — no users/events/free-tier/rating
functions, since this schema doesn't have those tables at all.
"""
import asyncio
import json

import asyncpg
import pgvector.asyncpg

from app.config import DATABASE_URL

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
