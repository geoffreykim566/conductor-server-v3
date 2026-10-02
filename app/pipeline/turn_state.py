"""Everything a turn accumulates while its tool loop runs, and rebuilding it from a
parked transcript on resume (nothing is stored server-side between requests)."""
import json
from dataclasses import dataclass, field

from app import tools
from app.pipeline import settings
from app.pipeline.cards import attach_key, endorse_key, mark_replaced, replace_span


def _lookup_is_unproductive(output: dict, buckets_seen: set) -> bool:
    """See settings.COUNT_ALL_LOOKUPS. A capped/errored call isn't a lookup result at all."""
    if "error" in output:
        return False
    if output.get("match") == "none":
        return True
    if str(output.get("match_confidence", "")).startswith("weak"):
        return True
    return output.get("problem") in buckets_seen


@dataclass
class TurnState:
    trace: list[dict] = field(default_factory=list)
    # Every iteration's text, not just the last: text written alongside a tool
    # call is part of the answer.
    text_parts: list[str] = field(default_factory=list)
    lookups_counted: int = 0  # toward settings.LOOKUP_ATTEMPT_LIMIT
    buckets_seen: set = field(default_factory=set)
    card_steps: list = field(default_factory=list)  # every attach this turn, in call order
    attach_keys: set = field(default_factory=set)
    queued_by_lookup: dict[str, tuple[int, int]] = field(default_factory=dict)  # endorse key -> its span
    route_spans: dict[str, tuple[int, int]] = field(default_factory=dict)  # model-called open_setting -> its span
    pick_forced: bool = False  # the one pick-or-ask re-call per turn

    @property
    def steps(self) -> list | None:
        return list(self.card_steps) or None

    def append_steps(self, tool: str, inp: dict, steps: list | None) -> tuple[int, int]:
        start = len(self.card_steps)
        self.card_steps.extend(steps or [])
        self.attach_keys.add(attach_key(tool, inp))
        return start, len(self.card_steps)

    def replace_steps(self, start: int, end: int, new: list) -> None:
        replace_span(self.card_steps, start, end, new, self.queued_by_lookup, self.route_spans)

    def count_lookup(self, output: dict) -> None:
        if settings.COUNT_ALL_LOOKUPS or _lookup_is_unproductive(output, self.buckets_seen):
            self.lookups_counted += 1
        self.buckets_seen.add(output.get("problem"))


def restore_turn_state(tail: list[dict]) -> TurnState:
    """Replay the completed tool-loop iterations between the user message and the
    pending assistant message, so an attach made before a research prompt survives."""
    state = TurnState()
    calls: dict[str, dict] = {}
    for m in tail:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if m.get("role") == "assistant":
                if b.get("type") == "text" and b.get("text"):
                    state.text_parts.append(b["text"])
                elif b.get("type") == "tool_use":
                    calls[b.get("id")] = {"tool": b.get("name"), "input": b.get("input") or {}}
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in calls:
                call = calls[b["tool_use_id"]]
                try:
                    output = json.loads(b.get("content") or "{}")
                except (TypeError, ValueError):
                    output = {"raw": b.get("content")}
                if not isinstance(output, dict):
                    output = {"raw": output}
                state.trace.append({"tool": call["tool"], "input": call["input"], "output": output})
                if call["tool"] == "lookup_concept":
                    state.count_lookup(output)
                    # Actions a lookup queued itself are recorded only in its own result.
                    for auto in output.get("on_card") or []:
                        state.trace.append({"tool": auto["tool"], "input": auto["input"],
                                            "output": auto["output"], "auto_from": output.get("problem")})
                        if not auto["output"].get("attached"):
                            continue
                        span = state.append_steps(auto["tool"], auto["input"], auto["output"].get("steps"))
                        state.queued_by_lookup[endorse_key(auto["tool"], auto["input"])] = span
                if call["tool"] in tools.ACTION_TOOLS and output.get("attached"):
                    key = endorse_key(call["tool"], call["input"])
                    new = output.get("steps") or []
                    if output.get("already_queued") or output.get("replaces_earlier"):
                        # Replaced an earlier span live; replay the same replacement.
                        span = state.queued_by_lookup.pop(key, None) or state.route_spans.get(key)
                        if span:
                            state.replace_steps(*span, new)
                            if call["tool"] == "open_setting":
                                state.route_spans[key] = (span[0], span[0] + len(new))
                        if output.get("replaces_earlier"):
                            mark_replaced(state.trace[:-1], call["input"].get("name"))
                        state.attach_keys.add(attach_key(call["tool"], call["input"]))
                        continue
                    span = state.append_steps(call["tool"], call["input"], new)
                    if call["tool"] == "open_setting":
                        state.route_spans[key] = span
    return state
