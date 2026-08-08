"""Unit tests for ratelimit.py's get_client_ip -- IPv6 /64 bucketing and
X-Forwarded-For parsing. Pure functions, no DB/network needed. Run inside the
app container:
    docker compose exec app python -m app.test_ratelimit
"""
from __future__ import annotations

from types import SimpleNamespace

from app.ratelimit import get_client_ip


def _request(headers: dict, client_host: str = "203.0.113.5") -> SimpleNamespace:
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=client_host))


def test_ipv4_unchanged() -> None:
    ip = get_client_ip(_request({}, client_host="198.51.100.7"))
    assert ip == "198.51.100.7", f"IPv4 should pass through unchanged, got {ip}"
    print("PASS: IPv4 address passes through unchanged.")


def test_ipv6_bucketed_to_slash_64() -> None:
    a = get_client_ip(_request({}, client_host="2001:db8:abcd:12::1"))
    b = get_client_ip(_request({}, client_host="2001:db8:abcd:12::ffff"))
    assert a == b, f"two addresses in the same /64 must bucket to the same key, got {a!r} vs {b!r}"
    assert a == "2001:db8:abcd:12::", f"expected the /64 network address, got {a!r}"
    print("PASS: two IPv6 addresses in the same /64 bucket identically.")


def test_ipv6_different_subnet_different_bucket() -> None:
    a = get_client_ip(_request({}, client_host="2001:db8:abcd:12::1"))
    b = get_client_ip(_request({}, client_host="2001:db8:abcd:13::1"))
    assert a != b, "addresses in different /64s must not collide"
    print("PASS: different /64 subnets bucket differently.")


def test_x_forwarded_for_uses_rightmost_entry() -> None:
    # Rightmost is our own reverse proxy's append; anything left of it is
    # client-controlled and forgeable (see ratelimit.py's own docstring).
    ip = get_client_ip(_request({"X-Forwarded-For": "1.2.3.4, 203.0.113.9"}))
    assert ip == "203.0.113.9", f"expected the rightmost XFF entry, got {ip!r}"
    print("PASS: X-Forwarded-For uses the rightmost (proxy-appended) entry.")


def test_malformed_ip_left_as_is() -> None:
    ip = get_client_ip(_request({}, client_host="not-an-ip"))
    assert ip == "not-an-ip", f"a malformed/hostname value should pass through, got {ip!r}"
    print("PASS: a malformed IP string is left as-is rather than raising.")


def main() -> None:
    test_ipv4_unchanged()
    test_ipv6_bucketed_to_slash_64()
    test_ipv6_different_subnet_different_bucket()
    test_x_forwarded_for_uses_rightmost_entry()
    test_malformed_ip_left_as_is()


if __name__ == "__main__":
    main()
