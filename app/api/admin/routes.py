"""/admin: the internal KPI dashboard page and the two JSON endpoints it calls,
behind basic auth. Ported from v1 (README.md)."""
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.api.admin import queries
from app.core.config import ADMIN_PASSWORD, ADMIN_USER

router = APIRouter()
_security = HTTPBasic()
_HTML = Path(__file__).parent / "static" / "admin.html"


def _require_admin(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    # Fail closed: with no password configured, nobody gets in.
    ok = bool(ADMIN_PASSWORD) and \
         secrets.compare_digest(credentials.username, ADMIN_USER) and \
         secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(_: None = Depends(_require_admin)):
    return HTMLResponse(_HTML.read_text())


@router.get("/admin/stats", include_in_schema=False)
async def stats(_: None = Depends(_require_admin)):
    return await queries.stats()


@router.get("/admin/events", include_in_schema=False)
async def events_log(
    _: None = Depends(_require_admin),
    page: int = 1,
    limit: int = 50,
    user_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    event_type: str | None = None,
):
    return await queries.events_log(page, limit, user_id, from_date, to_date, event_type)
