"""Tool name -> executor, and the status line shown while a tool runs. Tests swap
entries with patch.dict(executors.EXECUTORS, ...) and patch `executors.research`."""
from app import research, tools


async def _ack_clarifying_question() -> dict:
    """The call itself is the signal; it still needs a real tool_result round-trip."""
    return {"acknowledged": True}


async def _now(result: dict) -> dict:
    """Wraps a synchronous executor's result so every entry can be awaited."""
    return result


EXECUTORS = {
    "open_plugin": lambda inp, fixture: _now(tools.queue_open_plugin(inp)),
    "set_param": lambda inp, fixture: _now(tools.queue_set_param(inp)),
    "lookup_concept": lambda inp, fixture: tools.lookup_concept(inp["problem"]),
    "open_setting": lambda inp, fixture: _now(tools.queue_open_setting(inp, fixture)),
    "ask_clarifying_question": lambda inp, fixture: _ack_clarifying_question(),
    "web_research": lambda inp, fixture: research.web_research(inp["query"]),
}

# Only the two real waits get a status; action tools are instant and would just flash.
_TOOL_STATUS_PRIORITY = ["web_research", "lookup_concept"]


def status_for_tools(tool_uses: list) -> str | None:
    names = {tu.name for tu in tool_uses}
    for name in _TOOL_STATUS_PRIORITY:
        if name not in names:
            continue
        if name == "web_research":
            return "Searching the web for a verified answer…"
        if name == "lookup_concept":
            return "Searching internal knowledge base…"
    return None
