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
    "get_walkthrough": lambda inp, fixture: tools.get_walkthrough(inp["solution"], ax_fixture=fixture),
}


@dataclass
class Result:
    text: str
    walkthrough_steps: list | None
    trace: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)


def _usage_dict(u) -> dict:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


async def respond(messages: list[dict], ax_fixture: dict | None = None) -> Result:
    msgs = list(messages)
    trace: list[dict] = []
    usage: list[dict] = []
    walkthrough_steps: list | None = None
    lookup_attempts = 0

    # Pushed, not pulled: live state (when known for this turn) is handed to the
    # model automatically rather than waiting on it to decide to call a tool for
    # it -- a decision point it's already been observed skipping under real
    # conditions (found live 2026-08-04, the monitor-button hallucination).
    system_text = SYSTEM_PROMPT
    if ax_fixture:
        state_lines = "\n".join(f"- {k}: {v}" for k, v in ax_fixture.items())
        system_text += (
            "\n\n## Live state for this turn\n\n"
            "Read directly from the running Logic Pro project via the Accessibility "
            "API -- ground truth, not something the user said or you inferred. This "
            "outranks stated claims, seed_weight, and anything read from a screenshot "
            "when they conflict.\n\n"
            f"{state_lines}"
        )
    # Cached as its own block: identical across every iteration of this turn's
    # loop (system+tools resent unchanged on each one -- measured 2026-08-05:
    # 66 calls, 207,869 uncached input tokens across a 24-scenario battery).
    # The safety-net call below appends its own instruction as a second,
    # uncached block so it still hits this same cache entry.
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

    for _ in range(MAX_ITERATIONS):
        resp = await _client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools.TOOLS,
            messages=msgs,
        )
        usage.append(_usage_dict(resp.usage))
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text")
            msgs.append({"role": "assistant", "content": resp.content})
            return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs, usage=usage)

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
            elif tu.name == "get_walkthrough" and walkthrough_steps is not None:
                # Commit-to-one: only the first successful attach in a turn
                # reaches the user (found live 2026-08-05 -- multiple attaches
                # in one turn meant the *last* one silently won, contradicting
                # whichever candidate the response text actually led with).
                result = {
                    "attached": False,
                    "reason": (
                        "a walkthrough was already attached this turn -- only one attaches per "
                        "turn. Mention a fallback candidate in prose instead, and only attach it "
                        "if the user comes back and says the first one didn't work."
                    ),
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
        system=system + [{
            "type": "text",
            "text": "\n\nAnswer now with your best available information — no more tool calls.",
        }],
        messages=msgs,
    )
    usage.append(_usage_dict(final.usage))
    text = "".join(b.text for b in final.content if b.type == "text")
    msgs.append({"role": "assistant", "content": final.content})
    return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs, usage=usage)
