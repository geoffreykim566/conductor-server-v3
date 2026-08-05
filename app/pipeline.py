"""v3 pipeline: query + context -> tool-calling loop -> response.

respond() takes the full running conversation (already ending with the newest
user turn) and returns the full conversation after this turn completes, so the
caller (the test harness) just appends the next user turn for the next call.
History is the only state — nothing is pinned server-side between turns.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic

from app import tools
from app.config import CENTRAL_ANTHROPIC_KEY, MAX_TOKENS, MODEL
from app.prompt import SYSTEM_PROMPT

MAX_ITERATIONS = 6

# A/B switch, env-controlled rather than a respond() parameter -- lets the same
# battery runner be invoked twice (THINKING_ENABLED=true / =false) to isolate
# whether thinking mode itself changes outcomes, versus just being on for every
# run and never knowing if a pass was the fix or the reasoning budget (found
# live 2026-08-05, results were confounded across 3 bundled changes at once).
THINKING_ENABLED = os.environ.get("THINKING_ENABLED", "true").strip().lower() != "false"

# Hard, code-enforced cap on lookup_concept calls within one turn — a prompt
# instruction to "stop retrying" is a request, not a guarantee. Found live
# 2026-07-30: a genuinely absent topic ("wheres the monitor button") kept
# returning a *different* wrong match on every retried phrasing, so the model
# never hit a clean, repeated "nothing here" signal and just kept trying new
# wording until it hit the overall iteration ceiling with no answer at all.
# This intervenes explicitly before that happens, distinct from the overall
# MAX_ITERATIONS safety net below (which covers any tool, not just this one).
LOOKUP_ATTEMPT_LIMIT = 4

# Extended thinking -- a real reasoning budget, not just exposing something
# that was already happening silently. Added 2026-08-05 specifically to
# investigate two live findings the prompt/trace alone couldn't explain: why
# no_sound_direct sometimes leads with a lower-weight, unevidenced candidate,
# and why ax_override_critical_test occasionally attaches a walkthrough for a
# value already known from pushed AX state. max_tokens is widened by the
# thinking budget so real output isn't starved by the reasoning budget.
THINKING_BUDGET_TOKENS = 2048

_client = AsyncAnthropic(api_key=CENTRAL_ANTHROPIC_KEY)


def _model_kwargs() -> dict:
    if THINKING_ENABLED:
        return {
            "max_tokens": MAX_TOKENS + THINKING_BUDGET_TOKENS,
            "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        }
    return {"max_tokens": MAX_TOKENS}

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
    thinking: list[str] = field(default_factory=list)


async def respond(messages: list[dict], ax_fixture: dict | None = None) -> Result:
    msgs = list(messages)
    trace: list[dict] = []
    thinking: list[str] = []
    walkthrough_steps: list | None = None
    lookup_attempts = 0

    # Pushed, not pulled: live state (when known for this turn) is handed to the
    # model automatically rather than waiting on it to decide to call a tool for
    # it -- a decision point it's already been observed skipping under real
    # conditions (found live 2026-08-04, the monitor-button hallucination).
    system = SYSTEM_PROMPT
    if ax_fixture:
        state_lines = "\n".join(f"- {k}: {v}" for k, v in ax_fixture.items())
        system += (
            "\n\n## Live state for this turn\n\n"
            "Read directly from the running Logic Pro project via the Accessibility "
            "API -- ground truth, not something the user said or you inferred. This "
            "outranks stated claims, seed_weight, and anything read from a screenshot "
            "when they conflict.\n\n"
            f"{state_lines}"
        )

    for _ in range(MAX_ITERATIONS):
        resp = await _client.messages.create(
            model=MODEL,
            system=system,
            tools=tools.TOOLS,
            messages=msgs,
            **_model_kwargs(),
        )
        thinking.extend(b.thinking for b in resp.content if b.type == "thinking")
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text")
            msgs.append({"role": "assistant", "content": resp.content})
            return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs, thinking=thinking)

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
        system=system + "\n\nAnswer now with your best available information — no more tool calls.",
        messages=msgs,
        **_model_kwargs(),
    )
    thinking.extend(b.thinking for b in final.content if b.type == "thinking")
    text = "".join(b.text for b in final.content if b.type == "text")
    msgs.append({"role": "assistant", "content": final.content})
    return Result(text=text, walkthrough_steps=walkthrough_steps, trace=trace, messages=msgs, thinking=thinking)
