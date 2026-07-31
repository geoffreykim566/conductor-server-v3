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

# Hard, code-enforced cap on lookup_concept calls within one turn — a prompt
# instruction to "stop retrying" is a request, not a guarantee. Found live
# 2026-07-30: a genuinely absent topic ("wheres the monitor button") kept
# returning a *different* wrong match on every retried phrasing, so the model
# never hit a clean, repeated "nothing here" signal and just kept trying new
# wording until it hit the overall iteration ceiling with no answer at all.
# This intervenes explicitly before that happens, distinct from the overall
# MAX_ITERATIONS safety net below (which covers any tool, not just this one).
LOOKUP_ATTEMPT_LIMIT = 4

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
    lookup_attempts = 0

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
            if tu.name == "lookup_concept":
                lookup_attempts += 1
            if tu.name == "lookup_concept" and lookup_attempts > LOOKUP_ATTEMPT_LIMIT:
                result = {
                    "error": (
                        f"Too many lookup attempts ({lookup_attempts}) without a clear answer. "
                        "Stop searching now — answer from general Logic Pro knowledge if you're "
                        "genuinely confident, or tell the user you don't have a verified answer "
                        "for this. Do not call lookup_concept again this turn."
                    )
                }
            else:
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

    # Safety net: never return a truly empty response, regardless of why the
    # loop didn't converge on its own. Force one final tools-off call so the
    # model must synthesize whatever it already learned into a real answer.
    final = await _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT + "\n\nAnswer now with your best available information — no more tool calls.",
        messages=msgs,
    )
    text = "".join(b.text for b in final.content if b.type == "text")
    msgs.append({"role": "assistant", "content": final.content})
    return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs)
