"""ChatRequest: the /v3/chat body. The client is not a trust boundary: history is
replayed straight into the model call, so only the exact shapes the pipeline produces
are allowed. Why each limit is what it is: README.md (Request limits)."""
import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

MAX_HISTORY_LEN = 300  # sanity ceiling; history.trim_history does the real trim
MAX_MESSAGE_CHARS = 4_000
MAX_TEXT_CHARS = 20_000  # generous vs real tool-result JSON sizes
MAX_TOOL_INPUT_CHARS = 4_000
MAX_BLOCKS_PER_MSG = 10
_ALLOWED_ROLES = {"user", "assistant"}
_ALLOWED_MSG_KEYS = {"role", "content"}
_ALLOWED_TEXT_BLOCK_KEYS = {"type", "text"}
_ALLOWED_TOOL_USE_KEYS = {"type", "id", "name", "input"}
_ALLOWED_TOOL_RESULT_KEYS = {"type", "tool_use_id", "content"}

# Per screenshot (base64, ~1.5MB raw) and per turn (the client also caps at 4 windows).
MAX_SCREENSHOT_CHARS = 2 * 1024 * 1024
MAX_SCREENSHOTS = 4
# The client's AX dump caps itself near 12k chars.
MAX_AX_STATE_CHARS = 32 * 1024


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=MAX_MESSAGE_CHARS)
    # Exactly what a prior "done" event sent as "history". None starts a new conversation.
    history: list[dict] | None = None
    # This turn's window captures. Never part of history.
    screenshots: list[str] | None = Field(None, max_length=MAX_SCREENSHOTS)
    # This turn's Accessibility text dump. Never part of history.
    ax_state: str | None = Field(None, max_length=MAX_AX_STATE_CHARS)
    # v0.3.2+: ask before web research runs (the stream ends in "research_prompt").
    research_confirm: bool = False
    # Second half of a parked research turn: `history` is the parked transcript and
    # `message` is ignored. Not charged a second free message.
    resume: Literal["allow_research", "deny_research"] | None = None

    @field_validator("screenshots")
    @classmethod
    def _validate_screenshots(cls, shots: list[str] | None) -> list[str] | None:
        if shots is None:
            return shots
        kept: list[str] = []
        for s in shots:
            if not isinstance(s, str) or not s:
                raise ValueError("invalid screenshot")
            if len(s) > MAX_SCREENSHOT_CHARS:
                # Drop, don't reject: screenshots are best-effort context (README).
                log.warning(
                    "[screenshot_dropped] %d chars > cap %d", len(s), MAX_SCREENSHOT_CHARS
                )
                continue
            kept.append(s)
        return kept or None

    @field_validator("history")
    @classmethod
    def _validate_history(cls, msgs: list[dict] | None) -> list[dict] | None:
        if msgs is None:
            return msgs
        if len(msgs) > MAX_HISTORY_LEN:
            raise ValueError(f"too many history messages (max {MAX_HISTORY_LEN})")
        for msg in msgs:
            if not isinstance(msg, dict) or set(msg) - _ALLOWED_MSG_KEYS:
                raise ValueError("unexpected keys in message")
            role = msg.get("role")
            if role not in _ALLOWED_ROLES:
                raise ValueError("invalid role")
            content = msg.get("content")
            if isinstance(content, str):
                if role != "user":
                    raise ValueError("only user turns may have plain-string content")
                if len(content) > MAX_TEXT_CHARS:
                    raise ValueError("message text too long")
                continue
            if not isinstance(content, list):
                raise ValueError("invalid content")
            if len(content) > MAX_BLOCKS_PER_MSG:
                raise ValueError(f"too many content blocks (max {MAX_BLOCKS_PER_MSG})")
            for block in content:
                if not isinstance(block, dict):
                    raise ValueError("invalid content block")
                btype = block.get("type")
                if role == "assistant":
                    if btype == "text":
                        if set(block) - _ALLOWED_TEXT_BLOCK_KEYS:
                            raise ValueError("unexpected keys in text block")
                        if not isinstance(block.get("text", ""), str):
                            raise ValueError("invalid text block")
                        if len(block.get("text", "")) > MAX_TEXT_CHARS:
                            raise ValueError("message text too long")
                    elif btype == "tool_use":
                        if set(block) - _ALLOWED_TOOL_USE_KEYS:
                            raise ValueError("unexpected keys in tool_use block")
                        if not isinstance(block.get("id"), str) or not isinstance(block.get("name"), str):
                            raise ValueError("invalid tool_use block")
                        if len(json.dumps(block.get("input", {}))) > MAX_TOOL_INPUT_CHARS:
                            raise ValueError("tool_use input too large")
                    else:
                        raise ValueError(f"unsupported assistant block type: {btype!r}")
                else:  # user
                    if btype != "tool_result":
                        raise ValueError(f"unsupported user block type: {btype!r}")
                    if set(block) - _ALLOWED_TOOL_RESULT_KEYS:
                        raise ValueError("unexpected keys in tool_result block")
                    if not isinstance(block.get("tool_use_id"), str):
                        raise ValueError("invalid tool_result block")
                    # Always a string: a block list could smuggle images/cache_control past the checks above.
                    if not isinstance(block.get("content", ""), str):
                        raise ValueError("invalid tool_result content")
                    if len(block.get("content", "")) > MAX_TEXT_CHARS:
                        raise ValueError("tool_result content too long")
        return msgs
