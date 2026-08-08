"""HTTP/SSE layer for client-v3.

Real per-user free-tier counting, ratings, and streaming are wired now.
Deliberately still deferred (see the Fable scope-out run 2026-08-08 for a
plan on these): HMAC signing of conductor_id (X-Conductor-Id is trusted
as-sent, parsed as a bare uuid — see deps.py), IP-based rate limiting on
/v1/register and /v1/chat, a daily-cost circuit breaker, an oversized-body
guard, and admin/analytics routes. Not production-hardened yet.

Conversation continuity is opaque round-tripping, not a server-side
session store — chosen because pipeline.respond() is explicitly designed
stateless ("history is the only state," see pipeline.py's own docstring),
and a server-side store would contradict that. The client sends back
exactly the `history` a prior `/v1/chat` call returned (the full raw
message list respond() produced, including tool_use/tool_result blocks)
plus the new user turn; nothing is kept here between requests. This is
what makes the battery's verified multi-turn behaviors (backfill,
fallback continuity) apply to live traffic too, not just the harness.
Trimmed to MAX_HISTORY_MESSAGES before each call (_trim_history below).

pipeline.respond() streams the final answer's text via an on_chunk
callback (see pipeline.py's _call_model) — /v1/chat forwards each piece as
its own SSE "chunk" event as soon as it arrives, not one lump at the end.
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import asyncpg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import db
from app.config import MAX_HISTORY_MESSAGES
from app.deps import current_user
from app.pipeline import respond


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    try:
        yield
    finally:
        await db.disconnect()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    # Opaque — exactly what a prior /v1/chat call's "done" event sent as
    # "history". None (or omitted) starts a fresh conversation.
    history: list[dict] | None = None


def _json_default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    raise TypeError(f"not serializable: {type(o)}")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, default=_json_default)}\n\n"


def _trim_history(messages: list[dict]) -> list[dict]:
    """Keep the newest MAX_HISTORY_MESSAGES, snapped forward to the next real
    turn boundary (a plain-string user message) so a trim never starts on a
    dangling tool_result whose tool_use got cut — that shape is rejected by
    the Anthropic API outright. If no such boundary exists in the window
    (a pathologically long single tool-loop), don't truncate at all rather
    than risk sending a broken message list.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    window = messages[-MAX_HISTORY_MESSAGES:]
    for i, msg in enumerate(window):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return window[i:]
    return messages


@app.post("/v1/register")
async def register():
    cid = uuid.uuid4()
    await db.get_or_create_user(cid)
    return {"conductor_id": str(cid)}


@app.get("/v1/me")
async def get_me(user: asyncpg.Record = Depends(current_user)):
    return {
        "free_used": user["free_used"],
        "free_limit": user["free_limit"],
        "remaining": max(user["free_limit"] - user["free_used"], 0),
    }


class Profile(BaseModel):
    experience: str | None = Field(None, max_length=50)
    role: str | None = Field(None, max_length=50)


@app.put("/v1/me")
async def update_me(p: Profile, user: asyncpg.Record = Depends(current_user)):
    await db.update_profile(user["id"], p.experience, p.role)
    return {"ok": True}


@app.delete("/v1/me")
async def delete_me(user: asyncpg.Record = Depends(current_user)):
    await db.mark_uninstalled(user["id"])
    return {"ok": True}


class RatingIn(BaseModel):
    event_id: uuid.UUID
    rating: int = Field(..., ge=-1, le=1)


@app.post("/v1/ratings")
async def set_rating(r: RatingIn, user: asyncpg.Record = Depends(current_user)):
    ok = await db.set_rating(r.event_id, user["id"], r.rating)
    if not ok:
        raise HTTPException(status_code=404, detail="event_not_found")
    return {"ok": True}


@app.post("/v1/chat")
async def chat(req: ChatRequest, user: asyncpg.Record = Depends(current_user)):
    # Claimed before any model call — a request that never gets a real answer
    # still cost a real API call if claimed after, so claim first and fail
    # closed on the cap rather than risk free messages that don't decrement.
    claimed = await db.claim_free_message(user["id"])
    if claimed is None:
        raise HTTPException(
            status_code=402,
            detail={"error": "free_limit_reached", "limit": user["free_limit"]},
        )
    remaining = max(claimed["free_limit"] - claimed["free_used"], 0)

    async def stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_chunk(text: str) -> None:
            await queue.put(("chunk", text))

        async def run() -> None:
            try:
                history = _trim_history(req.history or [])
                messages = history + [{"role": "user", "content": req.message}]
                result = await respond(messages, on_chunk=on_chunk)
                await queue.put(("result", result))
            except Exception as e:
                await queue.put(("error", e))

        task = asyncio.create_task(run())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "chunk":
                    yield _sse({"type": "chunk", "text": payload})
                elif kind == "error":
                    yield _sse({"type": "error", "message": f"{type(payload).__name__}: {payload}"})
                    return
                else:  # "result"
                    result = payload
                    tokens_in = sum(u["input_tokens"] for u in result.usage)
                    tokens_out = sum(u["output_tokens"] for u in result.usage)
                    event_id = await db.insert_event(
                        user_id=user["id"], tokens_in=tokens_in, tokens_out=tokens_out
                    )
                    yield _sse({
                        "type": "done",
                        "event_id": str(event_id),
                        "remaining": remaining,
                        "source_tier": "",
                        "sources": [],
                        "intent": "",
                        "locate_type": "",
                        "element": "",
                        "search_term": "",
                        "walkthrough_steps": result.walkthrough_steps or [],
                        "history": result.messages,
                    })
                    return
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream")
