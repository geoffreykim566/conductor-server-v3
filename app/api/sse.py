"""Server-sent event framing for /v3/chat."""
import json


def _json_default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    raise TypeError(f"not serializable: {type(o)}")


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, default=_json_default)}\n\n"
