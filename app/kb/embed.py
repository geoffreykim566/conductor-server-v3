"""Voyage embeddings, shared by seeding (input_type="document") and lookup_concept
(input_type="query"). Both sides must use VOYAGE_MODEL."""
import asyncio
import time

import httpx

from app.core.config import VOYAGE_API_KEY, VOYAGE_MODEL

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

# Minimum gap between Voyage calls, ahead of lookup.py's retry (README: Rate limits).
_MIN_INTERVAL_S = 3.0
_lock = asyncio.Lock()
_last_call_at = 0.0


async def embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Return one embedding per text from the Voyage API."""
    global _last_call_at
    async with _lock:
        wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call_at)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_at = time.monotonic()

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _VOYAGE_URL,
            headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
            json={"model": VOYAGE_MODEL, "input": texts, "input_type": input_type},
        )
        if resp.status_code == 429:
            print(f"[voyage_429] rate limited: input_type={input_type} n_texts={len(texts)}")
        resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]
