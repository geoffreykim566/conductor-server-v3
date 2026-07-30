"""v3 pipeline: query + context -> tool-calling loop -> response.

respond() takes the full running conversation (already ending with the newest
user turn) and returns the full conversation after this turn completes, so the
caller (the test harness) just appends the next user turn for the next call.
History is the only state — nothing is pinned server-side between turns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic

from app import tools
from app.config import CENTRAL_ANTHROPIC_KEY, MAX_TOKENS, MODEL
from app.prompt import SYSTEM_PROMPT

MAX_ITERATIONS = 6

_client = AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY)

_EXECUTORS = {
    "lookup_concept": lambda inp, fixture: tools.lookup_concept(inp["problem"]),
    "get_walkthrough": lambda inp, fixture: tools.get_walkthrough(inp["solution"]),
    "read_ax_state": lambda inp, fixture: tools.read_ax_state(inp["query"], fixture=fixture),
}


@dataclass
class Result:
    text: str
    walkthrough_steps: list | None
    trace: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)


async def respond(messages: list[dict], ax_fixture: dict | None = None) -> Result:
    msgs = list(messages)
    trace: list[dict] = []
    walkthrough_steps: list | None = None

    for _ in range(MAX_ITERATIONS):
        resp = await _client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools.TOOLS,
            messages=msgs,
        )
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text")
            msgs.append({"role": "assistant", "content": resp.content})
            return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs)

        msgs.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for tu in tool_uses:
            executor = _EXECUTORS.get(tu.name)
            result = {"error": f"unknown tool {tu.name!r}"} if executor is None else await executor(tu.input, ax_fixture)
            trace.append({"tool": tu.name, "input": tu.input, "output": result})
            if tu.name == "get_walkthrough" and result.get("attached"):
                walkthrough_steps = result.get("steps")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })
        msgs.append({"role": "user", "content": tool_results})

    return Result(
        text="[max tool-call iterations reached without a final answer]",
        walkthrough_steps=walkthrough_steps,
        trace=trace,
        messages=msgs,
    )
