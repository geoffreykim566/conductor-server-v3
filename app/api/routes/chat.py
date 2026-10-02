"""POST /v3/chat: one turn, streamed as SSE (status, chunk, then done /
research_prompt / error). Event shapes and the free-message rules: README.md."""
import asyncio
import logging
import time

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import current_user
from app.api.history import pending_research_query, trim_history
from app.api.schemas import ChatRequest
from app.api.sse import sse
from app.core import budget, db
from app.core.config import MODEL, RATE_LIMIT
from app.core.ratelimit import limiter
from app.pipeline import respond

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v3/chat")
@limiter.limit(RATE_LIMIT)
async def chat(request: Request, req: ChatRequest, user: asyncpg.Record = Depends(current_user)):
    # Budget check before claim, so a refused request never burns a free message.
    if await budget.over_daily_budget():
        raise HTTPException(
            status_code=503,
            detail={"error": "free_tier_paused", "message": "Free tier is busy — try again later."},
        )
    if req.resume:
        # Resume: check the parked shape, don't charge again (README: Research approval).
        if pending_research_query(req.history) is None:
            log.warning("[resume_422] resume=%s with no pending web_research in history", req.resume)
            raise HTTPException(
                status_code=422,
                detail={"error": "no_pending_research", "message": "Nothing to resume."},
            )
        remaining = max(user["free_limit"] - user["free_used"], 0)
    else:
        # Claim before any model call, failing closed on the cap.
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

        async def on_status(text: str) -> None:
            await queue.put(("status", text))

        async def run() -> None:
            try:
                history = trim_history(req.history or [])
                # A resume's history already ends with this turn's parked
                # tool call; the user message is further back inside it.
                messages = history if req.resume else history + [{"role": "user", "content": req.message}]
                result = await respond(
                    messages, screenshots_b64=req.screenshots, ax_state=req.ax_state,
                    on_chunk=on_chunk, on_status=on_status,
                    research_confirm=req.research_confirm, resume=req.resume,
                )
                await queue.put(("result", result))
            except Exception as e:
                # The client only gets "Type: message"; log the traceback here.
                log.exception("[turn_error] pipeline raised %s", type(e).__name__)
                await queue.put(("error", e))

        turn_started = time.monotonic()
        task = asyncio.create_task(run())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "chunk":
                    yield sse({"type": "chunk", "text": payload})
                elif kind == "status":
                    yield sse({"type": "status", "text": payload})
                elif kind == "error":
                    yield sse({"type": "error", "message": f"{type(payload).__name__}: {payload}"})
                    return
                else:  # "result"
                    result = payload
                    tokens_in = sum(u["input_tokens"] for u in result.usage)
                    tokens_out = sum(u["output_tokens"] for u in result.usage)
                    event_id = await db.insert_event(
                        user_id=user["id"], tokens_in=tokens_in, tokens_out=tokens_out,
                        model=MODEL, source_tier=result.confidence_tier,
                        latency_ms=round((time.monotonic() - turn_started) * 1000),
                        prompt=req.message, response=result.text,
                    )
                    if result.pending_research_query is not None:
                        # Parked for approval. Spend so far is logged above; ratings
                        # land on the resume's event.
                        yield sse({
                            "type": "research_prompt",
                            "query": result.pending_research_query,
                            "history": result.messages,
                        })
                        return
                    yield sse({
                        "type": "done",
                        "event_id": str(event_id),
                        "remaining": remaining,
                        # Confidence badge disabled (README: Done payload).
                        "source_tier": "",
                        "sources": result.sources,
                        "intent": "",
                        "locate_type": "",
                        "element": "",
                        "search_term": "",
                        "walkthrough_steps": result.walkthrough_steps or [],
                        # Older clients ignore it.
                        "auto_run": result.auto_run,
                        "history": result.messages,
                    })
                    return
        finally:
            if not task.done():
                # Client disconnected mid-turn (Esc cancel, or a crash): abort the model call.
                task.cancel()
                log.warning("[turn_cancelled] client disconnected mid-turn; pipeline task cancelled")

    return StreamingResponse(stream(), media_type="text/event-stream")
