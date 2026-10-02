"""Sniffing a pushed screenshot's media type from its base64 prefix (pipeline.context)."""
from __future__ import annotations

import base64

from app.pipeline.context import media_type_for_b64


def test_screenshot_media_type_sniffed() -> None:
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
    jpg = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 8).decode()
    assert media_type_for_b64(png) == "image/png"
    assert media_type_for_b64(jpg) == "image/jpeg"
    assert media_type_for_b64("zzzz") == "image/png", "unknown prefix falls back to png (v0.3.0 behavior)"
