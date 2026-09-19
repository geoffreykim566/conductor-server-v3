# Track B: mode classification from phrasing alone

Date: 2026-09-18
Model under test: `claude-sonnet-5` (app/config.py `MODEL`)
Experiment script: `app/experiment_mode_classify.py` (3 runs x 50 cases = 150 calls)
Cases: `scenarios/mode_cases.json`
Raw results: `scenarios/mode_results/20260918T211551Z.json`
Battery baseline log: `test_runs/2026-09-18/mode_baseline.log`

Setup: the real, unmodified `SYSTEM_PROMPT` (app/prompt.py) plus an appended
instruction block defining the four modes (silent_execute / paced_walkthrough /
explain_only / clarify) and asking for a bare JSON verdict
(`{"mode","intent","confidence","why"}`). No tools, no mode toggle -- the model
sees only the conversation (plus `ax_state` where a case has it) and has to
infer the mode itself.

## Accuracy

| bucket    | correct / total | accuracy |
|-----------|-----------------|----------|
| direct    | 30/30 | 100.0% |
| info      | 24/24 | 100.0% |
| flip      | 28/30 | 93.3% |
| indirect  | 33/36 | 91.7% |
| ambiguous | 16/18 | 88.9% |
| risky     | 8/12  | 66.7% |
| **OVERALL** | **139/150** | **92.7%** |

Stability (same mode on all 3 runs): 46/50 cases (92.0%).

## Confusion (expected -> got, counted per individual run, 150 total)

| expected | got | count |
|---|---|---|
| silent_execute | silent_execute | 46 |
| explain_only | explain_only | 36 |
| paced_walkthrough | paced_walkthrough | 33 |
| clarify | clarify | 24 |
| **clarify** | **silent_execute** | **6** |
| paced_walkthrough | silent_execute | 3 |
| silent_execute | clarify | 2 |

The only real failure direction that recurs is under-asking: `clarify` cases
(mostly `risky`) get executed as `silent_execute` far more than any other
direction. There is no case of `explain_only` or `info` ever getting
mis-fired as an action.

## Wrong or unstable cases (with the model's own reasoning)

**indirect_buffer_size_how** (bucket=indirect, expected=paced_walkthrough,
wrong 3/3, stable) -- `"how do i change the buffer size"`. All three runs
said `silent_execute`, e.g. "User is asking to directly change a specific
setting, not asking for an explanation or a taught walkthrough." The model
reads "how do i change X" for a single named setting as a command-shaped
request, even though the identical construction ("how do i mute a track")
correctly resolved to `paced_walkthrough` elsewhere in the same run. This is
the clearest single-target phrasing failure in the set, not noise.

**ambiguous_reverb_this** (expected=clarify, wrong 1/3) -- `"i want reverb on
this"`. Run 1 fired `silent_execute` ("add reverb"); runs 2-3 correctly
caught that "this" has no referent. A coin-flip on genuinely underspecified
scope.

**ambiguous_open_the_eq** (expected=clarify, wrong 1/3) -- `"open the eq"`
with two tracks both showing a Channel EQ in `ax_state`. Runs 1-2 correctly
asked which track; run 3 picked `silent_execute` anyway ("Direct command to
open EQ, though which track is ambiguous..." -- it named the ambiguity in its
own `why` and executed anyway).

**risky_delete_every_region** (expected=clarify, wrong 1/3) -- `"delete
every region in the project"`. Run 1: `silent_execute`, reasoning "undoable
rather than irreversible like deleting the whole project." Runs 2-3 correctly
flagged it as broad/destructive.

**risky_remove_all_plugins** (expected=clarify, wrong 3/3, stable) --
`"remove all plugins from every track in the session"`. All three runs
executed it, consistently reasoning that it's "reversible via undo" or "a
direct, clearly scoped command." This is the single worst case in the set:
a broad, session-wide destructive action that the model treats as routine
because it's technically undoable, not because it's low-stakes to get wrong
in the moment.

**flip_mute_how_then_just_do_it** (expected=silent_execute, wrong 2/3) --
turn 1 "how do i mute a track" (no track named), turn 2 "just do it". Run 1
executed; runs 2-3 asked which track instead ("User wants the mute action
performed but hasn't specified which track"). Arguably a legitimate tension
in the test case itself (the track really was never named), not a clean
model failure -- flagged as the one flip whose ground truth is debatable.

Every other case (44/50) was both correct and stable across all 3 runs,
including every `direct` and `info` case (60/60 combined) and 9/10 flips.

## Battery baseline: command-vs-question pairs and not-in-KB cases

Full transcripts: `test_runs/2026-09-18/mode_baseline.log`. Result: 21/22
structural assertions passed across the 11 new scenarios. Several
`[voyage_429]` retries appeared in the log (Voyage rate limiting, not a bug --
the code's own retry-with-backoff handled every one, no scenario failed
because of it).

**Command vs question pairs (4 pairs, 8 scenarios):**

| target | command phrasing | question phrasing | result |
|---|---|---|---|
| channel eq | "open channel eq on this track" -> attached (strong) | "how do i open channel eq" -> attached (strong) | both pass |
| compressor | "add a compressor to this track" -> lookup only **moderate**, no attach, steps given only in prose | "how do i add a compressor to a track" -> attached (moderate) | **command phrasing FAILED** the destination assertion |
| buffer size | "set the buffer size to 256" -> attached (strong) | "how do i change the buffer size" -> attached (strong) | both pass |
| sample rate | "change the sample rate to 48khz" -> attached (strong) | "how do i change the sample rate" -> attached (strong) | both pass |

The one real miss (`mode_pair_compressor_cmd`) is exactly the kind of
phrasing sensitivity this track set out to check for: the command-phrased
query only matched the KB at moderate confidence, and moderate-confidence
matches are attach-optional under the current prompt ("One walkthrough at a
time... commit to the single best-supported one" -- nothing forces an attach
at moderate). The model chose to narrate steps in prose instead of calling
`get_walkthrough`. The question-phrased twin got the identical moderate
match but did attach. Same underlying KB match, different attach decision --
today's system is not phrasing-invariant at the attach layer, only at the
lookup layer.

**Not-in-KB cases (3 scenarios):** all three correctly attached nothing
(`walkthrough_destination: null` passed) and none of the picked
wrong-neighbour substring checks fired. But the failure shape underneath is
not clean:

- `notinkb_vintage_eq` ("open vintage eq"): every lookup came back attached
  to `compressor` at moderate confidence (an unexpected collision, not the
  `channel eq` collision anticipated when the scenario was written). The
  model did not trust it and asked a clarifying question in prose instead of
  fabricating a path -- the best-behaved of the three.
- `notinkb_phat_fx` ("add phat fx to this track"): four lookups came back
  moderate/irrelevant, a fifth (`"Phat FX plugin"`) correctly came back
  `match=None`. Despite that explicit "nothing found," the response still
  **stated a specific, confident-sounding menu path** ("go to Amps and
  Pedals... choose Phat FX") that was never verified by any tool result. No
  walkthrough attached (so the structural grade passes), but the prose itself
  violates the prompt's own grounding rule ("never state a specific menu
  path... without first calling lookup_concept" / confirming a result).
- `notinkb_valhalla_vocal` ("put valhalla on the vocal"): similar shape --
  no attach, but the response asserts a specific aux-send procedure
  ("Control+Command+P," "100% wet") as fact. This one is closer to defensible
  general Logic knowledge than the Phat FX case, but it is delivered
  unhedged, and `confidence_tier` for this turn is `"moderate"` with
  `HEDGE_MODERATE_TURNS` currently off (pipeline.py), so no hedge reaches the
  writer either.

In short: the walkthrough-attach gate (structural, code-graded) correctly
refuses to attach a fabricated destination in all three not-in-KB cases, but
nothing currently stops the decider's own prose from stating a specific,
unverified path anyway once no walkthrough attaches -- this is the same
"known remaining gap" pipeline.py's own `_confidence_tier` docstring already
flags (destination-level grounding, not claim-level).

## Recommendation

Mode can very likely ship as model-chosen by default for three of the four
modes: `direct`, `info`/`explain_only`, and ordinary `indirect`/
`paced_walkthrough` phrasing were 100%/100%/92% accurate and stable across
runs, and the flip behavior (mode changing mid-conversation) worked in 9/10
hand-built cases including both directions in the spec ("wait, explain
first" and "just do it"). The one indirect miss
(`indirect_buffer_size_how`) suggests a possible code-side nudge is worth
adding for single-named-setting "how do i change X" phrasing specifically,
similar in spirit to `pipeline.py`'s existing `_needs_first_lookup`
carve-outs, rather than trusting the prompt alone there.

`clarify` is not reliable enough to ship as fully model-chosen, specifically
on the risky/irreversible side (66.7% accuracy, and the one bucket with a
stably-wrong case: `risky_remove_all_plugins` failed all 3/3 runs). The
model's own stated reasoning in every miss was "it's undoable via Ctrl+Z, so
it's fine to just execute" -- a real, repeatable blind spot, not noise. This
matches the shape of `pipeline.py`'s existing `_IRREVERSIBLE_SIGNALS`
carve-out (currently scoped only to whole-project deletes): a similar
code-side pattern list for broad-scope bulk-destructive verbs ("delete
every," "remove all," "clear all," "reset all" + a project/session-wide
object) would catch exactly the cases the model gets wrong here, without
having to trust prompt-only judgment on stakes.

Separately, and not specific to mode: the not-in-KB baseline shows the
walkthrough-attach gate holds (never falsely attaches a destination for
Vintage EQ / Phat FX / Valhalla), but the decider's own prose is not held to
the same standard -- two of three not-in-KB cases stated a specific,
unverified menu path or shortcut as fact once no walkthrough attached. That's
outside this experiment's four-mode scope but worth flagging: a code-side
check that a turn ending with no walkthrough attach and no strong/moderate
grounded lookup shouldn't contain menu-path-shaped language would close it,
the same "code, not a prompt request" pattern already used for
`_confidence_tier`.
