"""POST /v3/register: mint a signed conductor id."""
import uuid

from fastapi import APIRouter, HTTPException, Request

from app.core import db, signing
from app.core.config import REGISTER_RATE_LIMIT
from app.core.ratelimit import limiter

router = APIRouter()


@router.post("/v3/register")
@limiter.limit(REGISTER_RATE_LIMIT)
async def register(request: Request):
    cid = uuid.uuid4()
    try:
        # Sign before inserting: without the secret, refuse rather than mint ids nothing can verify.
        token = signing.sign(cid)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="registration_unavailable")
    await db.get_or_create_user(cid)
    return {"conductor_id": token}
