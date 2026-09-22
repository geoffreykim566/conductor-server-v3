-- action (2026-09-21): the call a solution maps to when a lookup lands on it
-- -- {"open_setting": "<route name in seed/routes.json>"}, {"open_plugin":
-- {"plugin": ...}}, or a list of such calls. Replaces path / extends_to:
-- navigation paths now live once, in seed/routes.json, reached through the
-- open_setting action tool (never embedded or matched against). path,
-- extends_to, toggle_ax_key and value_ax_key are left in place but no longer
-- written; the toggle gate moved onto the route.
alter table solutions
    add column if not exists action jsonb;
