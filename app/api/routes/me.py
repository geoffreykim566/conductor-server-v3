"""/v3/me: the caller's free-message count, profile, and uninstall marker."""
import asyncpg
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import current_user
from app.core import db

router = APIRouter()


class Profile(BaseModel):
    experience: str | None = Field(None, max_length=50)
    role: str | None = Field(None, max_length=50)


@router.get("/v3/me")
async def get_me(user: asyncpg.Record = Depends(current_user)):
    return {
        "free_used": user["free_used"],
        "free_limit": user["free_limit"],
        "remaining": max(user["free_limit"] - user["free_used"], 0),
    }


@router.put("/v3/me")
async def update_me(p: Profile, user: asyncpg.Record = Depends(current_user)):
    await db.update_profile(user["id"], p.experience, p.role)
    return {"ok": True}


@router.delete("/v3/me")
async def delete_me(user: asyncpg.Record = Depends(current_user)):
    await db.mark_uninstalled(user["id"])
    return {"ok": True}
