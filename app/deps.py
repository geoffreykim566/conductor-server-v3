"""Identity — the only place auth is decided.

No signature verification yet (deferred by explicit choice, see api.py's
module docstring) — X-Conductor-Id is trusted as whatever the client sent,
parsed as a bare uuid. A malformed header is rejected before touching the DB;
a well-formed one is upserted lazily so a row lost to a DB reset self-heals.
"""
import uuid

import asyncpg
from fastapi import Header, HTTPException

from app import db


async def current_user(x_conductor_id: str = Header(...)) -> asyncpg.Record:
    try:
        cid = uuid.UUID(x_conductor_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid_conductor_id")
    return await db.get_or_create_user(cid)
