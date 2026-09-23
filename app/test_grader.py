"""Unit tests for run_graded_battery's action assertions (2026-09-21).

The case that motivated them: on 2026-09-18 args_set_param_threshold_not_loaded
graded PASS on a turn whose reply said "The threshold is set to -18 dB" for a
set_param attach commit-to-one had refused -- walkthrough_destination was its
only assertion. That exact trace must now fail. Also checks that a v040-shaped
trace (template rows through get_walkthrough) and a typed-tool trace grade the
same, since the two baselines are compared against each other.

No DB, no model: hand-built traces only. Run inside the app container:
    docker compose exec app python -m app.test_grader
"""
from __future__ import annotations

from app.run_graded_battery import _actual_outcome, _grade

THRESHOLD_EXPECT = {
    "action_calls": [
        {"tool": "open_plugin", "args": {"plugin": "compressor"}},
        {"tool": "set_param", "args": {"plugin": "compressor", "param": "threshold", "value": "-18"}},
    ],
    "no_refused_actions": True,
    "no_completion_claims": True,
}


def _failed(verdicts: list[dict]) -> set[str]:
    return {v["field"] for v in verdicts if not v["pass"]}


def test_0918_false_pass_now_fails() -> None:
    # Verbatim shape of test_runs/2026-09-18/args_battery.log:127-146.
    trace = [
        {"tool": "lookup_concept", "input": {"problem": "open plugin"},
         "output": {"match": "single", "problem": "plugin manager"}},
        {"tool": "lookup_concept", "input": {"problem": "add compressor to track"},
         "output": {"match": "single", "problem": "multipressor"}},
        {"tool": "get_walkthrough", "input": {"solution": "open plugin", "args": {"plugin": "Compressor"}},
         "output": {"attached": True, "destination": "open plugin", "steps": [{"ax_open_plugin": "Compressor"}]}},
        {"tool": "get_walkthrough",
         "input": {"solution": "set plugin parameter",
                   "args": {"plugin": "Compressor", "param": "Threshold", "value": -18}},
         "output": {"attached": False, "reason": "a walkthrough was already attached this turn"}},
    ]
    text = "Compressor is loaded on Audio 1. The threshold is set to -18 dB."
    failed = _failed(_grade(THRESHOLD_EXPECT, _actual_outcome(trace, text)))
    assert failed == {"action_calls", "no_refused_actions", "no_completion_claims"}, failed
    print("PASS: the 09-18 false-PASS trace fails on action_calls, no_refused_actions and no_completion_claims.")


def test_typed_tools_pass() -> None:
    trace = [
        {"tool": "open_plugin", "input": {"plugin": "Compressor", "track": None, "new_instance": False},
         "output": {"attached": True, "steps": [{"ax_open_plugin": "Compressor"}]}},
        {"tool": "set_param", "input": {"plugin": "Compressor", "param": "Threshold", "value": -18, "track": None},
         "output": {"attached": True, "steps": []}},
    ]
    text = "This adds Compressor to the selected track. This sets Compressor Threshold to -18."
    failed = _failed(_grade(THRESHOLD_EXPECT, _actual_outcome(trace, text)))
    assert not failed, failed
    print("PASS: the same request through typed tools passes, template sentence included.")


def test_legacy_and_typed_grade_alike() -> None:
    expect = {"action_calls": [{"tool": "open_plugin", "args": {"plugin": "channel eq"},
                                "args_absent": ["new_instance"]}]}
    legacy = [{"tool": "get_walkthrough", "input": {"solution": "open plugin", "args": {"plugin": "Channel EQ", "new": True}},
               "output": {"attached": True, "destination": "open plugin"}}]
    typed = [{"tool": "open_plugin", "input": {"plugin": "Channel EQ", "track": None, "new_instance": True},
              "output": {"attached": True}}]
    for trace in (legacy, typed):
        assert _failed(_grade(expect, _actual_outcome(trace))) == {"action_calls"}, trace
    typed[0]["input"]["new_instance"] = False
    legacy[0]["input"]["args"].pop("new")
    for trace in (legacy, typed):
        assert not _failed(_grade(expect, _actual_outcome(trace))), trace
    print("PASS: v040 template attaches and typed-tool calls grade identically, incl. args_absent.")


def test_order_matters() -> None:
    trace = [
        {"tool": "set_param", "input": {"plugin": "Compressor", "param": "Threshold", "value": -18}, "output": {"attached": True}},
        {"tool": "open_plugin", "input": {"plugin": "Compressor"}, "output": {"attached": True}},
    ]
    assert "action_calls" in _failed(_grade(THRESHOLD_EXPECT, _actual_outcome(trace)))
    print("PASS: set before open fails the ordered action_calls assertion.")


def test_true_negative() -> None:
    expect = {"no_action_calls": True, "action_calls": None}
    question = [{"tool": "lookup_concept", "input": {"problem": "buffer size"}, "output": {"match": "single"}},
                {"tool": "get_walkthrough", "input": {"solution": "buffer size"}, "output": {"attached": True, "destination": "buffer size"}}]
    assert not _failed(_grade(expect, _actual_outcome(question)))
    refused_only = [{"tool": "open_plugin", "input": {"plugin": "x"}, "output": {"attached": False, "reason": "r"}}]
    assert _failed(_grade(expect, _actual_outcome(refused_only))) == {"no_action_calls", "action_calls"}
    print("PASS: no_action_calls passes a walkthrough-only turn and fails even a refused action call.")


def test_completion_claims() -> None:
    expect = {"no_completion_claims": True}
    bad = ["The threshold is set to -18 dB.", "Channel EQ is now loaded on Audio 1.",
           "I've added a compressor.", "Track 3 has been muted."]
    ok = ["This adds Channel EQ to Audio 1.", "Press Run and it will set the threshold to -18.",
          "Set the threshold to -18 in the compressor window."]
    for t in bad:
        assert _failed(_grade(expect, _actual_outcome([], t))), t
    for t in ok:
        assert not _failed(_grade(expect, _actual_outcome([], t))), t
    print("PASS: completion-claim check flags past-tense claims, not future/imperative phrasing.")


def test_setting_calls() -> None:
    def call(inp, ok=True):
        return {"tool": "open_setting", "input": inp, "output": {"attached": ok}}
    trace = [call({"name": "buffer size", "value": "1024"}, ok=False), call({"name": "buffer size", "value": "larger"}),
             {"tool": "open_plugin", "input": {"plugin": "Compressor"}, "output": {"attached": True}}]
    larger = {"tool": "open_setting", "args": {"name": "buffer size", "value": "larger"}}
    assert not _failed(_grade({"setting_calls": [larger]}, _actual_outcome(trace)))
    assert _failed(_grade({"forbidden_setting_calls": [larger]}, _actual_outcome(trace))) == {"forbidden_setting_calls"}
    # a refused call doesn't count as made; open_setting stays out of action_calls
    assert not _failed(_grade({"forbidden_setting_calls": [{"tool": "open_setting", "args": {"value": "1024"}}],
                               "action_calls": [{"tool": "open_plugin", "args": {"plugin": "compressor"}}]},
                              _actual_outcome(trace)))
    pane = {"tool": "open_setting", "args": {"name": "buffer size"}, "args_absent": ["value"]}
    assert _failed(_grade({"setting_calls": [pane]}, _actual_outcome(trace[1:2]))) == {"setting_calls"}
    assert not _failed(_grade({"setting_calls": [pane]}, _actual_outcome([call({"name": "buffer size"})])))
    print("PASS: setting_calls / forbidden_setting_calls grade open_setting name + value; refusals don't count.")


def main() -> None:
    test_0918_false_pass_now_fails()
    test_typed_tools_pass()
    test_legacy_and_typed_grade_alike()
    test_order_matters()
    test_true_negative()
    test_completion_claims()
    test_setting_calls()


if __name__ == "__main__":
    main()
