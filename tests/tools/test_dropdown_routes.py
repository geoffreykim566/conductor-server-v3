"""open_setting on dropdown routes: picking a checked value or stopping at the pane."""
from __future__ import annotations

from conftest import queue_setting as _setting


def test_no_value_opens_pane_only() -> None:
    """No value: the pane only; the dropdown is never clicked open (an open menu
    swallowed the next step's keystrokes). The menu verifies on the dropped row."""
    out = _setting("buffer size")
    assert out["steps"] == [{"menu_path": ["Logic Pro", "Settings", "Audio"], "expect": ["I/O Buffer Size"]}], out
    assert "pane only" in out["note"], out


def test_named_value_ends_on_choose_step() -> None:
    """A value the user named: the route, then a choose step carrying the route for revert."""
    out = _setting("buffer size", "256 samples", said="set my buffer to 256")
    assert out["chooses"] == "256" and out["steps"][-1]["choose"] == "256", out
    assert out["steps"][-1]["reopen"] == out["steps"][:-1] == [
        {"menu_path": ["Logic Pro", "Settings", "Audio"]}, {"click_value_of": "I/O Buffer Size"}], out


def test_buffer_size_values() -> None:
    """A number the user didn't name is refused; a direction is the model's to pick."""
    out = _setting("buffer size", "1024", said="my audio keeps crackling")
    assert out["attached"] is False and "larger or smaller" in out["reason"], out
    assert _setting("buffer size", "larger", said="my audio keeps crackling")["chooses"] == "larger"
    assert _setting("buffer size", "Smaller")["chooses"] == "smaller"
    out = _setting("buffer size", "2048", said="set buffer to 2048")
    assert out["attached"] is False and "isn't one of" in out["reason"], out


def test_sample_rate_values() -> None:
    """Sample rate: named in any common spelling, never chosen for the user, no direction."""
    assert _setting("sample rate", "48000", said="change the sample rate to 48k")["chooses"] == "48 kHz"
    assert _setting("sample rate", "44.1", said="set it to 44100")["chooses"] == "44.1 kHz"
    assert _setting("sample rate", "48 kHz", said="my file sounds slowed down")["attached"] is False
    assert _setting("sample rate", "larger", said="set sample rate higher")["attached"] is False


def test_options_the_model_may_pick() -> None:
    """Threads: Automatic only when the user names it (the KB says max on Apple
    Silicon and the route can't pick a number). Flex time: model may pick Monophonic."""
    assert _setting("processing threads", "automatic", said="system overload on my M3")["attached"] is False
    assert _setting("processing threads", "automatic", said="set processing threads to automatic")["chooses"] == "Automatic"
    assert _setting("flex time", "mono", said="the vocal timing is off")["chooses"] == "Monophonic"
    assert _setting("flex time", "Speed (FX)", said="fix the timing")["attached"] is False
    assert _setting("flex time", "Speed (FX)", said="set flex to speed")["chooses"] == "Speed (FX)"


def test_flex_pitch_fixed_option() -> None:
    """Flex pitch = the same popup with a fixed option: it ends on a choose step
    (a plain click lost the popup to the client's open-menu cleanup) carrying the
    route for Revert, and it takes no value."""
    out = _setting("flex pitch")
    assert out["chooses"] == "Flex Pitch" and out["steps"][-1]["choose"] == "Flex Pitch", out
    assert out["steps"][-1]["reopen"] == out["steps"][:-1] and len(out["steps"]) == 3, out
    assert not any(s.get("click_text") == "Flex Pitch" for s in out["steps"]), out
    assert _setting("flex pitch", "Monophonic")["attached"] is False


def test_region_smart_tempo() -> None:
    """Pane only keeps the dropped row as the Region click's `expect`; what the
    control displays ("Bars"/"Beats") names the option."""
    assert _setting("follow tempo")["steps"] == [
        {"shortcut": "I"}, {"click_text": "Region", "expect": ["Smart Tempo"]}]
    assert _setting("follow tempo", "off", said="stop the region stretching")["chooses"] == "Off"
    assert _setting("follow tempo", "On", said="fix the stretching")["attached"] is False
    out = _setting("follow tempo", "On + Align Bars", said="set smart tempo to on + align bars")
    assert out["steps"][-1]["shows"] == {"On + Align Bars": "Bars", "On + Align Bars and Beats": "Beats"}, out
    assert "shows" not in _setting("buffer size", "256", said="set my buffer to 256")["steps"][-1]
    assert out["chooses"] == "On + Align Bars" and out["steps"][-1]["reopen"] == [
        {"shortcut": "I"}, {"click_text": "Region"}, {"click_value_of": "Smart Tempo"}], out
    assert _setting("follow tempo", "beats", said="set smart tempo to beats")["chooses"] == "On + Align Bars and Beats"
    assert _setting("follow tempo", "On + Align Bars", said="set it to bars")["chooses"] == "On + Align Bars"


def test_offered_option_counts_as_named() -> None:
    """An option the previous reply offered is the user's pick once they answer;
    without the offer it's still refused."""
    offer = "which one do you want: on, on + align bars, or on + align bars and beats?"
    assert _setting("follow tempo", "On + Align Bars and Beats", said="the last one", offered=offer)["chooses"] == \
        "On + Align Bars and Beats"
    assert _setting("buffer size", "1024", said="yes", offered="want me to set it to 1024?")["chooses"] == "1024"
    assert _setting("follow tempo", "On + Align Bars and Beats", said="the last one")["attached"] is False
    assert _setting("sample rate", "48 kHz", said="still slow", offered="")["attached"] is False


def test_smart_tempo_import_default() -> None:
    """Options are named without their punctuation, and "smart tempo" names Logic's "ST"."""
    assert _setting("smart tempo import default", "Flex On, Smart Tempo On",
                    said="set imported files to flex on smart tempo on")["chooses"] == "Flex On, Smart Tempo On"
    assert _setting("smart tempo import default", "Flex On, ST On + Align Bars",
                    said="imported files: flex on, smart tempo on + align bars")["chooses"] == "Flex On, ST On + Align Bars"
    assert _setting("smart tempo import default", "Flex On, Smart Tempo On",
                    said="turn smart tempo on for imports")["attached"] is False
    assert _setting("smart tempo import default", "Flex Off", said="stop my imports stretching")["chooses"] == "Flex Off"


def test_ambiguous_value_lists_options() -> None:
    """A value that fits several options is refused with those options, so the
    reply asks which one instead of naming one that doesn't exist."""
    out = _setting("smart tempo import default", "Flex On", said="set imported files to flex on")
    assert out["attached"] is False and "could be any of" in out["reason"], out
    assert all(o in out["reason"] for o in ("Flex On, Smart Tempo Off", "Flex On, Smart Tempo On",
                                            "Flex On, ST On + Align Bars", "Flex On, ST On + Align Bars and Beats")), out
    assert "Flex Off" not in out["reason"], out


def test_pane_only_note_asks_which_option() -> None:
    """The pane-only note asks which option, naming them -- unless there are fewer
    than two to name (threads); the user's answer then counts as naming it."""
    assert "ask the user which one" in _setting("smart tempo import default")["note"]
    assert "Flex On, Smart Tempo On" in _setting("smart tempo import default")["note"]
    assert _setting("processing threads")["note"].endswith("the user picks there")
    ask = "which one do you want: flex off, flex on, smart tempo off, flex on, smart tempo on, ...?"
    assert _setting("smart tempo import default", "Flex On, Smart Tempo On", said="smart tempo on",
                    offered=ask)["chooses"] == "Flex On, Smart Tempo On"


def test_plain_route_takes_no_value() -> None:
    out = _setting("mixer", "on")
    assert out["attached"] is False and "no value" in out["reason"], out
    assert "note" not in _setting("mixer") and _setting("mixer")["steps"] == [{"shortcut": "X"}]
