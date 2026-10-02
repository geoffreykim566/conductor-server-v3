"""Request guards that run before validation: the body-size cap and 422 logging."""
import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

# Uvicorn reads the whole body before pydantic runs, so the cap lives here. Bounded
# by one turn's screenshots plus a full text history (screenshots never accumulate).
_MAX_BODY_BYTES = 10 * 1024 * 1024


async def log_validation_errors(request: Request, exc: RequestValidationError):
    """Log which validator fired (the client shows every 422 as "message too long"),
    and drop pydantic's input echo, which for screenshots is the whole image."""
    errors = [
        {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
        for e in exc.errors()
    ]
    log.warning("[validation_422] %s %s: %s", request.method, request.url.path, errors)
    return JSONResponse({"detail": errors}, status_code=422)


async def reject_oversized_bodies(request: Request, call_next):
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > _MAX_BODY_BYTES:
                return JSONResponse({"detail": "request_too_large"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "invalid_content_length"}, status_code=400)
    return await call_next(request)
