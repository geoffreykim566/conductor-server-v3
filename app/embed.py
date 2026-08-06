"""Voyage AI embedding helper — shared by seeding (index time) and retrieval (query time).

Use input_type='document' when embedding problem/solution text at load time.
Use input_type='query' when embedding the model's own lookup_concept paraphrase.
Both sides must use VOYAGE_MODEL — symmetry is the only hard constraint.
"""
import asyncio
import time

import httpx

from app.config import VOYAGE_API_KEY, VOYAGE_MODEL

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

# Proactive spacing, not just reactive retry -- found live 2026-08-06: two full
# 38-scenario battery runs both died on sustained 429s partway through despite
# _embed_with_retry's backoff (tools.py), because that only reacts *after* a
# call already got rate-limited. This is the one chokepoint every embed() call
# passes through regardless of caller, so enforcing a minimum gap here caps
# sustained throughput before Voyage ever has a reason to 429 in the first
# place. 3s floor is a conservative starting point, not a measured tier limit
# (none published/known here) -- widen if battery runs still hit 429s.
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
