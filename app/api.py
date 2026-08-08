"""HTTP/SSE layer for client-v3.

Real per-user free-tier counting, ratings, streaming, HMAC-signed conductor
ids (see signing.py/deps.py), history shape validation (ChatRequest below),
an oversized-body guard, and a disabled /docs are wired now. Deliberately
still deferred (see the Fable scope-out run 2026-08-08 for a plan on these):
IP-based rate limiting on /v1/register and /v1/chat, a daily-cost circuit
breaker, and admin/analytics routes. Not production-hardened yet.

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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app import db, signing
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


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

# Traefik/uvicorn read the whole body before FastAPI's pydantic validation
# runs, so an oversized-body cap has to live ahead of that, in a middleware
# (same shape as v1's server/app/main.py). Cap is far smaller than v1's 45MB:
# v3's traffic is text-only (no image blocks anywhere in this pipeline), and
# even a maxed-out history (300 messages, ChatRequest's own per-field caps)
# tops out in the low single-digit MB.
_MAX_BODY_BYTES = 2 * 1024 * 1024


@app.middleware("http")
async def reject_oversized_bodies(request: Request, call_next):
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > _MAX_BODY_BYTES:
                return JSONResponse({"detail": "request_too_large"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "invalid_content_length"}, status_code=400)
    return await call_next(request)


# The client is not a trust boundary: `history` is round-tripped opaquely
# (see module docstring) and replayed straight into messages.create(), so a
# tampered payload could smuggle fields the API would honor (e.g. inflating
# cost) or malformed shapes that 400 mid-turn. Whitelist exactly the shapes
# pipeline.py's _serialize_content actually produces — plain-string user
# turns, assistant text/tool_use blocks, user tool_result blocks — unlike
# v1's whitelist (server/app/routes/chat.py), which never had to deal with
# tool_use/tool_result at all.
_MAX_HISTORY_LEN = 300  # sanity ceiling ahead of _trim_history's real trim to MAX_HISTORY_MESSAGES
_MAX_MESSAGE_CHARS = 4_000
_MAX_TEXT_CHARS = 20_000  # generous vs real walkthrough/tool-result JSON sizes
_MAX_TOOL_INPUT_CHARS = 4_000
_MAX_BLOCKS_PER_MSG = 10
_ALLOWED_ROLES = {"user", "assistant"}
_ALLOWED_MSG_KEYS = {"role", "content"}
_ALLOWED_TEXT_BLOCK_KEYS = {"type", "text"}
_ALLOWED_TOOL_USE_KEYS = {"type", "id", "name", "input"}
_ALLOWED_TOOL_RESULT_KEYS = {"type", "tool_use_id", "content"}


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=_MAX_MESSAGE_CHARS)
    # Opaque — exactly what a prior /v1/chat call's "done" event sent as
    # "history". None (or omitted) starts a fresh conversation.
    history: list[dict] | None = None

    @field_validator("history")
    @classmethod
    def _validate_history(cls, msgs: list[dict] | None) -> list[dict] | None:
        if msgs is None:
            return msgs
        if len(msgs) > _MAX_HISTORY_LEN:
            raise ValueError(f"too many history messages (max {_MAX_HISTORY_LEN})")
        for msg in msgs:
            if not isinstance(msg, dict) or set(msg) - _ALLOWED_MSG_KEYS:
                raise ValueError("unexpected keys in message")
            role = msg.get("role")
            if role not in _ALLOWED_ROLES:
                raise ValueError("invalid role")
            content = msg.get("content")
            if isinstance(content, str):
                if role != "user":
                    raise ValueError("only user turns may have plain-string content")
                if len(content) > _MAX_TEXT_CHARS:
                    raise ValueError("message text too long")
                continue
            if not isinstance(content, list):
                raise ValueError("invalid content")
            if len(content) > _MAX_BLOCKS_PER_MSG:
                raise ValueError(f"too many content blocks (max {_MAX_BLOCKS_PER_MSG})")
            for block in content:
                if not isinstance(block, dict):
                    raise ValueError("invalid content block")
                btype = block.get("type")
                if role == "assistant":
                    if btype == "text":
                        if set(block) - _ALLOWED_TEXT_BLOCK_KEYS:
                            raise ValueError("unexpected keys in text block")
                        if not isinstance(block.get("text", ""), str):
                            raise ValueError("invalid text block")
                        if len(block.get("text", "")) > _MAX_TEXT_CHARS:
                            raise ValueError("message text too long")
                    elif btype == "tool_use":
                        if set(block) - _ALLOWED_TOOL_USE_KEYS:
                            raise ValueError("unexpected keys in tool_use block")
                        if not isinstance(block.get("id"), str) or not isinstance(block.get("name"), str):
                            raise ValueError("invalid tool_use block")
                        if len(json.dumps(block.get("input", {}))) > _MAX_TOOL_INPUT_CHARS:
                            raise ValueError("tool_use input too large")
                    else:
                        raise ValueError(f"unsupported assistant block type: {btype!r}")
                else:  # user
                    if btype != "tool_result":
                        raise ValueError(f"unsupported user block type: {btype!r}")
                    if set(block) - _ALLOWED_TOOL_RESULT_KEYS:
                        raise ValueError("unexpected keys in tool_result block")
                    if not isinstance(block.get("tool_use_id"), str):
                        raise ValueError("invalid tool_result block")
                    # _serialize_content always emits a json.dumps() string here
                    # (pipeline.py), never a nested block list -- a list would let
                    # a smuggled image/cache_control block ride inside tool_result,
                    # past the block-level checks above that only look at content's
                    # own top-level type.
                    if not isinstance(block.get("content", ""), str):
                        raise ValueError("invalid tool_result content")
                    if len(block.get("content", "")) > _MAX_TEXT_CHARS:
                        raise ValueError("tool_result content too long")
        return msgs


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
    try:
        # Sign before inserting: a deploy without the secret must refuse
        # registration, not mint identities nothing will ever verify.
        token = signing.sign(cid)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="registration_unavailable")
    await db.get_or_create_user(cid)
    return {"conductor_id": token}


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
