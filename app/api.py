"""HTTP/SSE layer — dev wiring for client-v3.

First cut, deliberately minimal (per the wiring scope-out): only the two
endpoints the client can't function without at all, `/v1/register` and
`/v1/chat`. Everything client-v3 treats as optional (ratings, /v1/me,
control-map) is left unbuilt — the client already degrades gracefully
without them. No auth verification, no rate limits, no per-user state:
`X-Conductor-Id` is accepted but ignored, and /v1/register mints a plain
unsigned uuid. Not production-hardened — that's explicitly out of scope
for wiring up a first real end-to-end test.

Conversation history is flattened to plain {role, content: str} pairs
(images dropped, any structured content reduced to its text). This means
the model-driven backfill/multi-turn behaviors verified in the battery
(which persist real tool_use/tool_result blocks between turns) do NOT
carry over to live client traffic yet — accepted gap, not an oversight,
tracked in v3-log.md as a follow-up design item (server-side session
store vs. client round-tripping opaque message blocks).

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
    messages: list[dict]


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _text_of(content) -> str:
    """Flatten a v1-shaped message content field (str, or a list of text/image
    blocks) down to its plain text. Images are dropped — server-v3 has no
    vision path."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _to_plain_messages(raw: list[dict]) -> list[dict]:
    return [{"role": m.get("role"), "content": _text_of(m.get("content"))} for m in raw]


@app.post("/v1/register")
async def register():
    return {"conductor_id": str(uuid.uuid4())}


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    async def stream():
        try:
            messages = _to_plain_messages(req.messages)
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
            })
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="text/event-stream")
