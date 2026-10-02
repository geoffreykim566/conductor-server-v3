"""HMAC-signed conductor ids: `<uuid>.<hex sig>`, sig = HMAC-SHA256(CONDUCTOR_ID_SECRET,
uuid). Only /v3/register signs; everything else verifies, so forged ids never reach the DB."""
import hashlib
import hmac
import uuid

from app.core.config import CONDUCTOR_ID_SECRET


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
