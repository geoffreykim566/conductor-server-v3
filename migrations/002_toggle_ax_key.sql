-- toggle_ax_key: names which pushed ax_fixture key reflects whether this
-- solution's target state is already true. get_walkthrough refuses to attach
-- when it is -- attaching would just toggle an already-correct state away.
-- v1 of this: assumes the target/desired value is always boolean true (every
-- current toggle-type destination is "make X visible/on"); revisit if a
-- toggle with a different target value shows up.
alter table solutions
    add column if not exists toggle_ax_key text;
