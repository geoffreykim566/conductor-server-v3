"""POST /v3/ratings: thumbs up/down on a reply."""
import uuid

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import current_user
from app.core import db

router = APIRouter()


class RatingIn(BaseModel):
    event_id: uuid.UUID
    rating: int = Field(..., ge=-1, le=1)


@router.post("/v3/ratings")
async def set_rating(r: RatingIn, user: asyncpg.Record = Depends(current_user)):
    ok = await db.set_rating(r.event_id, user["id"], r.rating)
    if not ok:
        raise HTTPException(status_code=404, detail="event_not_found")
    return {"ok": True}
