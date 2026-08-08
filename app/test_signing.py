"""Unit tests for signing.py -- HMAC round-trip and tamper/no-secret rejection.

No DB needed (pure math, see signing.py's own docstring). Run inside the app
container:
    docker compose exec app python -m app.test_signing
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

from app import signing


def test_round_trip() -> None:
    cid = uuid.uuid4()
    with patch("app.signing.CONDUCTOR_ID_SECRET", "test-secret"):
        token = signing.sign(cid)
        assert signing.verify(token) == cid, "a token this module signed must verify back to the same uuid"
    print("PASS: sign() -> verify() round-trips to the original uuid.")


def test_tampered_signature_rejected() -> None:
    cid = uuid.uuid4()
    with patch("app.signing.CONDUCTOR_ID_SECRET", "test-secret"):
        token = signing.sign(cid)
        cid_str, _, sig = token.partition(".")
        tampered = f"{cid_str}.{'0' * len(sig)}"
        assert signing.verify(tampered) is None, "a flipped signature must not verify"
    print("PASS: a tampered signature is rejected.")


def test_bare_uuid_rejected() -> None:
    with patch("app.signing.CONDUCTOR_ID_SECRET", "test-secret"):
        assert signing.verify(str(uuid.uuid4())) is None, "a bare uuid with no signature suffix must not verify"
    print("PASS: a bare (unsigned) uuid is rejected.")


def test_wrong_secret_rejected() -> None:
    cid = uuid.uuid4()
    with patch("app.signing.CONDUCTOR_ID_SECRET", "secret-a"):
        token = signing.sign(cid)
    with patch("app.signing.CONDUCTOR_ID_SECRET", "secret-b"):
        assert signing.verify(token) is None, "a token signed under one secret must not verify under another"
    print("PASS: a token signed under a rotated-away secret is rejected.")


def test_no_secret_fails_closed() -> None:
    cid = uuid.uuid4()
    with patch("app.signing.CONDUCTOR_ID_SECRET", "test-secret"):
        token = signing.sign(cid)
    with patch("app.signing.CONDUCTOR_ID_SECRET", ""):
        assert signing.verify(token) is None, "with no secret configured, every token must be rejected"
        try:
            signing.sign(uuid.uuid4())
            raise AssertionError("sign() should raise RuntimeError with no secret configured")
        except RuntimeError:
            pass
    print("PASS: no configured secret fails closed for both sign() and verify().")


def main() -> None:
    test_round_trip()
    test_tampered_signature_rejected()
    test_bare_uuid_rejected()
    test_wrong_secret_rejected()
    test_no_secret_fails_closed()


if __name__ == "__main__":
    main()
