"""get_client_ip: IPv6 /64 bucketing and X-Forwarded-For parsing."""
from __future__ import annotations

from types import SimpleNamespace

from app.core.ratelimit import get_client_ip


def _request(headers: dict, client_host: str = "203.0.113.5") -> SimpleNamespace:
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=client_host))


def test_ipv4_unchanged() -> None:
    ip = get_client_ip(_request({}, client_host="198.51.100.7"))
    assert ip == "198.51.100.7", f"IPv4 should pass through unchanged, got {ip}"


def test_ipv6_bucketed_to_slash_64() -> None:
    a = get_client_ip(_request({}, client_host="2001:db8:abcd:12::1"))
    b = get_client_ip(_request({}, client_host="2001:db8:abcd:12::ffff"))
    assert a == b, f"two addresses in the same /64 must bucket to the same key, got {a!r} vs {b!r}"
    assert a == "2001:db8:abcd:12::", f"expected the /64 network address, got {a!r}"


def test_ipv6_different_subnet_different_bucket() -> None:
    a = get_client_ip(_request({}, client_host="2001:db8:abcd:12::1"))
    b = get_client_ip(_request({}, client_host="2001:db8:abcd:13::1"))
    assert a != b, "addresses in different /64s must not collide"


def test_x_forwarded_for_uses_rightmost_entry() -> None:
    # Rightmost is our own reverse proxy's append; anything left of it is
    # client-controlled and forgeable (see ratelimit.py's own docstring).
    ip = get_client_ip(_request({"X-Forwarded-For": "1.2.3.4, 203.0.113.9"}))
    assert ip == "203.0.113.9", f"expected the rightmost XFF entry, got {ip!r}"


def test_malformed_ip_left_as_is() -> None:
    ip = get_client_ip(_request({}, client_host="not-an-ip"))
    assert ip == "not-an-ip", f"a malformed/hostname value should pass through, got {ip!r}"
