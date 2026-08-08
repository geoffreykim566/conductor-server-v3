"""Identity — the only place auth is decided.

X-Conductor-Id carries a server-signed token (`<uuid>.<sig>`, minted by
/v1/register — see signing.py). A bare or tampered uuid fails HMAC
verification before ever touching the DB; a valid token is upserted lazily
so a row lost to a DB reset self-heals.
"""
import asyncpg
from fastapi import Header, HTTPException

from app import db, signing


async def current_user(x_conductor_id: str = Header(...)) -> asyncpg.Record:
    cid = signing.verify(x_conductor_id)
    if cid is None:
        raise HTTPException(status_code=401, detail="invalid_conductor_id")
    return await db.get_or_create_user(cid)
