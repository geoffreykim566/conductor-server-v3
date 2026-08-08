"""HMAC signing for conductor ids — fabricated ids die here, before the DB.

Wire format: `<uuid>.<hex sig>` where sig = HMAC-SHA256(CONDUCTOR_ID_SECRET,
canonical uuid string). Only /v1/register mints tokens; everything else just
verifies, so a valid token is proof the server issued this id. Ported from
server/app/signing.py (v1) — same design, v3's own secret.
"""
import hashlib
import hmac
import uuid

from app.config import CONDUCTOR_ID_SECRET


def _sig(cid: uuid.UUID) -> str:
    return hmac.new(
        CONDUCTOR_ID_SECRET.encode(), str(cid).encode(), hashlib.sha256
    ).hexdigest()


def sign(cid: uuid.UUID) -> str:
    """Token for a server-minted id. Raises if no secret is configured."""
    if not CONDUCTOR_ID_SECRET:
        raise RuntimeError("CONDUCTOR_ID_SECRET is not set")
    return f"{cid}.{_sig(cid)}"


def verify(token: str) -> uuid.UUID | None:
    """UUID behind a valid token, else None. Pure math — never touches the DB.

    Fails closed: with no secret configured, every token is rejected.
    """
    if not CONDUCTOR_ID_SECRET:
        return None
    cid_str, sep, sig = token.partition(".")
    if not sep:
        return None
    try:
        cid = uuid.UUID(cid_str)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sig(cid)):
        return None
    return cid
