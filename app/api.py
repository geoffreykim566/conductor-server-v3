"""HTTP/SSE layer — dev wiring for client-v3.

First cut, deliberately minimal (per the wiring scope-out): only the two
endpoints the client can't function without at all, `/v1/register` and
`/v1/chat`. Everything client-v3 treats as optional (ratings, /v1/me,
control-map) is left unbuilt — the client already degrades gracefully
without them. No auth verification, no rate limits, no per-user state:
`X-Conductor-Id` is accepted but ignored, and /v1/register mints a plain
unsigned uuid. Not production-hardened — that's explicitly out of scope
for wiring up a first real end-to-end test.

Conversation continuity is opaque round-tripping, not a server-side
session store — chosen because pipeline.respond() is explicitly designed
stateless ("history is the only state," see pipeline.py's own docstring),
and a server-side store would contradict that. The client sends back
exactly the `history` a prior `/v1/chat` call returned (the full raw
message list respond() produced, including tool_use/tool_result blocks)
plus the new user turn; nothing is kept here between requests. This is
what makes the battery's verified multi-turn behaviors (backfill,
fallback continuity) apply to live traffic too, not just the harness.

No history length/cost cap yet (the old client-side _MAX_CONTEXT_MESSAGES
trim doesn't have a v3 equivalent) — acceptable for dev testing, a real
gap before this could take real traffic.

pipeline.respond() is non-streaming (a multi-call tool-use loop), so this
only ever emits one "chunk" with the full text followed by "done" — real
token-level streaming of the final synthesis call is a later improvement.
"""
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import db
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


@app.post("/v1/register")
async def register():
    return {"conductor_id": str(uuid.uuid4())}


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    async def stream():
        try:
            messages = list(req.history or []) + [{"role": "user", "content": req.message}]
            result = await respond(messages)
            yield _sse({"type": "chunk", "text": result.text})
            yield _sse({
                "type": "done",
                "event_id": "",
                "remaining": -1,
                "source_tier": "",
                "sources": [],
                "intent": "",
                "locate_type": "",
                "element": "",
                "search_term": "",
                "walkthrough_steps": result.walkthrough_steps or [],
                "history": result.messages,
            })
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="text/event-stream")
