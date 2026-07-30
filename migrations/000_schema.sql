-- v3 schema: problem/solution buckets as real relational tables, not a generic
-- typed-entry-with-jsonb-body table wearing a new "type" value. See server-v3/plan.md.

create extension if not exists vector;
create extension if not exists pgcrypto;

-- The atomic answer unit: a diagnosis, a fix, a destination. Replaces the generic
-- kb_entries table's navigation/technical_problem/quality/concept types generically.
create table if not exists solutions (
    id          uuid primary key default gen_random_uuid(),
    name        text not null unique,
    embedding   vector(1024) not null,
    content     jsonb not null,   -- symptom/cause/fix/do_not text, whatever this solution needs to say
    path        jsonb,            -- executable steps, if any; null = no walkthrough for this solution
    tier        text not null default 'established'
);

-- A named symptom with its own aliases/embedding. Only exists when there's real
-- ambiguity worth grouping solutions under — a standalone destination never needs
-- a row here, it's just a solution searched directly (the "single" match path).
create table if not exists problems (
    id          uuid primary key default gen_random_uuid(),
    name        text not null unique,
    aliases     text[] not null default '{}',
    embedding   vector(1024) not null,
    note        text
);

-- The ranked bucket itself. A solution can belong to more than one problem
-- (e.g. "sample rate mismatch" is a candidate for both "song sounds slowed" and
-- "crackling during playback") — deliberately many-to-many, not a single owner.
create table if not exists problem_solutions (
    problem_id      uuid not null references problems(id) on delete cascade,
    solution_id     uuid not null references solutions(id) on delete cascade,
    seed_weight     int,
    distinguisher   text,
    primary key (problem_id, solution_id)
);

create index if not exists solutions_embedding_idx on solutions using hnsw (embedding vector_cosine_ops);
create index if not exists problems_embedding_idx on problems using hnsw (embedding vector_cosine_ops);
