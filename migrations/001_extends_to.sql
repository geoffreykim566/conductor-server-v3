-- extends_to: a solution with no path of its own can point at exactly one
-- other solution to inherit its path instead of ever duplicating that path
-- onto more than one row (e.g. a diagnosis borrowing a destination's real
-- navigable path). Deliberately not chainable -- enforced at load time in
-- embed_and_load.py, not here -- a path is a concrete, terminal thing, no
-- reason for more than one hop.
alter table solutions
    add column if not exists extends_to uuid references solutions(id);

alter table solutions
    add constraint solutions_path_xor_extends_to
    check (path is null or extends_to is null);
