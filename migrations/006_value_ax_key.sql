-- value_ax_key: names which pushed ax_fixture key already reflects this
-- solution's own live value this turn (unlike toggle_ax_key, there's no
-- fixed "target" -- the key's mere presence means the value is already
-- known, not something the user needs walked through checking). Fixes
-- multiturn_evidence_arrives_later_turn (v3-log.md 2026-08-18): when a
-- solution's real fix has a step the pushed AX state already answers,
-- get_walkthrough shouldn't attach a walkthrough for re-checking it.
alter table solutions
    add column if not exists value_ax_key text;
