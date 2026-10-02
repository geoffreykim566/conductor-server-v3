"""Pipeline tunables. Always read them as `settings.X` at call time, never
`from ...settings import X`: tests and evals/latency_variants.py patch them on this
module. Why each value is what it is: README.md (Tunables)."""

MAX_ITERATIONS = 6

# Hard cap on lookup_concept calls per turn.
LOOKUP_ATTEMPT_LIMIT = 4
# True: every lookup counts toward the cap. False: only unproductive ones (kept for A/B).
COUNT_ALL_LOOKUPS = True

# First decider call on a turn that needs grounding: True forces lookup_concept,
# False forces "some tool" (so a direct command can go straight to an action).
FORCE_FIRST_LOOKUP = False

# Steps one card may hold.
MAX_ATTACHES_PER_TURN = 6

# A moderate single-solution lookup goes to pick-or-ask rather than auto-attaching.
PICK_ON_MODERATE_SINGLE = True

# Tell the writer to hedge on "moderate" turns. Off: the tier over-fires on observations.
HEDGE_MODERATE_TURNS = False
