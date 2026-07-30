"""Voyage AI embedding helper — shared by seeding (index time) and retrieval (query time).

Use input_type='document' when embedding problem/solution text at load time.
Use input_type='query' when embedding the model's own lookup_concept paraphrase.
Both sides must use VOYAGE_MODEL — symmetry is the only hard constraint.
"""
import httpx

from app.config import VOYAGE_API_KEY, VOYAGE_MODEL

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


async def embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Return one embedding per text from the Voyage API."""
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
