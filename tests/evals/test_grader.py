"""The battery grader's action and setting assertions, on hand-built traces."""
from __future__ import annotations

from evals.battery.grader import actual_outcome, grade

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


def test_refused_set_param_with_completion_claim_fails() -> None:
    """A refused set_param plus a reply claiming it's done fails all three assertions
    (the 09-18 false PASS, in typed-tool shape)."""
    trace = [
        {"tool": "open_plugin", "input": {"plugin": "Compressor"},
         "output": {"attached": True, "steps": [{"ax_open_plugin": "Compressor"}]}},
        {"tool": "set_param", "input": {"plugin": "Compressor", "param": "Threshold", "value": -18},
         "output": {"attached": False, "reason": "a walkthrough was already attached this turn"}},
    ]
    text = "Compressor is loaded on Audio 1. The threshold is set to -18 dB."
    failed = _failed(grade(THRESHOLD_EXPECT, actual_outcome(trace, text)))
    assert failed == {"action_calls", "no_refused_actions", "no_completion_claims"}, failed


def test_typed_tools_pass() -> None:
    """The same request through typed tools passes, template sentence included."""
    trace = [
        {"tool": "open_plugin", "input": {"plugin": "Compressor", "track": None, "new_instance": False},
         "output": {"attached": True, "steps": [{"ax_open_plugin": "Compressor"}]}},
        {"tool": "set_param", "input": {"plugin": "Compressor", "param": "Threshold", "value": -18, "track": None},
         "output": {"attached": True, "steps": []}},
    ]
    text = "This adds Compressor to the selected track. This sets Compressor Threshold to -18."
    failed = _failed(grade(THRESHOLD_EXPECT, actual_outcome(trace, text)))
    assert not failed, failed


def test_args_absent() -> None:
    """args_absent fails on a truthy arg and treats a falsy one as absent."""
    expect = {"action_calls": [{"tool": "open_plugin", "args": {"plugin": "channel eq"},
                                "args_absent": ["new_instance"]}]}
    typed = [{"tool": "open_plugin", "input": {"plugin": "Channel EQ", "track": None, "new_instance": True},
              "output": {"attached": True}}]
    assert _failed(grade(expect, actual_outcome(typed))) == {"action_calls"}, typed
    typed[0]["input"]["new_instance"] = False
    assert not _failed(grade(expect, actual_outcome(typed))), typed


def test_order_matters() -> None:
    """Set before open fails the ordered action_calls assertion."""
    trace = [
        {"tool": "set_param", "input": {"plugin": "Compressor", "param": "Threshold", "value": -18}, "output": {"attached": True}},
        {"tool": "open_plugin", "input": {"plugin": "Compressor"}, "output": {"attached": True}},
    ]
    assert "action_calls" in _failed(grade(THRESHOLD_EXPECT, actual_outcome(trace)))


def test_true_negative() -> None:
    """no_action_calls passes a route-only turn and fails even a refused action call."""
    expect = {"no_action_calls": True, "action_calls": None}
    question = [{"tool": "cite_kb", "input": {"entries": ["buffer size"]}, "output": {"match": "single"}},
                {"tool": "open_setting", "input": {"name": "buffer size"}, "output": {"attached": True, "destination": "buffer size"}}]
    assert not _failed(grade(expect, actual_outcome(question)))
    refused_only = [{"tool": "open_plugin", "input": {"plugin": "x"}, "output": {"attached": False, "reason": "r"}}]
    assert _failed(grade(expect, actual_outcome(refused_only))) == {"no_action_calls", "action_calls"}


def test_completion_claims() -> None:
    """The completion-claim check flags past-tense claims, not future/imperative phrasing."""
    expect = {"no_completion_claims": True}
    bad = ["The threshold is set to -18 dB.", "Channel EQ is now loaded on Audio 1.",
           "I've added a compressor.", "Track 3 has been muted."]
    ok = ["This adds Channel EQ to Audio 1.", "Press Run and it will set the threshold to -18.",
          "Set the threshold to -18 in the compressor window."]
    for t in bad:
        assert _failed(grade(expect, actual_outcome([], t))), t
    for t in ok:
        assert not _failed(grade(expect, actual_outcome([], t))), t


def test_setting_calls() -> None:
    """setting_calls / forbidden_setting_calls grade open_setting name + value; refusals don't count."""
    def call(inp, ok=True):
        return {"tool": "open_setting", "input": inp, "output": {"attached": ok}}
    trace = [call({"name": "buffer size", "value": "1024"}, ok=False), call({"name": "buffer size", "value": "larger"}),
             {"tool": "open_plugin", "input": {"plugin": "Compressor"}, "output": {"attached": True}}]
    larger = {"tool": "open_setting", "args": {"name": "buffer size", "value": "larger"}}
    assert not _failed(grade({"setting_calls": [larger]}, actual_outcome(trace)))
    assert _failed(grade({"forbidden_setting_calls": [larger]}, actual_outcome(trace))) == {"forbidden_setting_calls"}
    # a refused call doesn't count as made; open_setting stays out of action_calls
    assert not _failed(grade({"forbidden_setting_calls": [{"tool": "open_setting", "args": {"value": "1024"}}],
                              "action_calls": [{"tool": "open_plugin", "args": {"plugin": "compressor"}}]},
                             actual_outcome(trace)))
    pane = {"tool": "open_setting", "args": {"name": "buffer size"}, "args_absent": ["value"]}
    assert _failed(grade({"setting_calls": [pane]}, actual_outcome(trace[1:2]))) == {"setting_calls"}
    assert not _failed(grade({"setting_calls": [pane]}, actual_outcome([call({"name": "buffer size"})])))
