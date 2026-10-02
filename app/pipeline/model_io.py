"""The one place the pipeline calls the Anthropic API, plus converting its response
blocks back into request-shaped dicts. Tests patch `model_io._client.messages`."""
import logging
import time
from typing import Awaitable, Callable

from anthropic import AsyncAnthropic

from app.core.config import CENTRAL_ANTHROPIC_KEY, MAX_TOKENS, MODEL

OnChunk = Callable[[str], Awaitable[None]]
OnStatus = Callable[[str], Awaitable[None]]

log = logging.getLogger(__name__)

_client = AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY)


async def call_model(
    system: list[dict],
    tools_param: list[dict] | None,
    msgs: list[dict],
    on_chunk: OnChunk | None,
    tool_choice: dict | None = None,
    model: str = MODEL,
):
    """One messages.create call, or the streamed equivalent when on_chunk is given
    (only the writer streams). Logs per-call timing and cache accounting as
    [model_call] so cache breakpoints can be checked from prod logs."""
    kwargs = dict(model=model, max_tokens=MAX_TOKENS, system=system, messages=msgs)
    if tools_param is not None:
        kwargs["tools"] = tools_param
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    t0 = time.monotonic()
    if on_chunk is None:
        resp = await _client.messages.create(**kwargs)
    else:
        async with _client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                await on_chunk(text)
            resp = await stream.get_final_message()
    # getattr guards: mocked tests return bare SimpleNamespace usage.
    u = getattr(resp, "usage", None)
    log.warning(
        "[model_call] %s %.1fs in=%s cache_read=%s cache_write=%s out=%s forced=%s",
        model, time.monotonic() - t0,
        getattr(u, "input_tokens", "?"),
        getattr(u, "cache_read_input_tokens", 0) or 0,
        getattr(u, "cache_creation_input_tokens", 0) or 0,
        getattr(u, "output_tokens", "?"),
        tool_choice is not None,
    )
    return resp


def serialize_content(content: list) -> list[dict]:
    """SDK response blocks -> plain request-shape dicts. Response objects carry
    fields the request schema rejects, and Result.messages is replayed as the next
    turn's history. Thinking blocks are dropped: the history validator rejects them
    and this non-interleaved loop doesn't need them (README: History round-trip)."""
    out = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        elif b.type == "thinking":
            continue
        else:
            out.append(b.model_dump())
    return out


def usage_dict(u) -> dict:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }
