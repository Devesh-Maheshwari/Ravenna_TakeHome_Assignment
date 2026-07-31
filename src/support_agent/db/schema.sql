-- Schema for the agentic customer support assistant.
--
-- Applied idempotently by scripts/init_db.py. There is no migration tool here
-- on purpose — see docs/DESIGN_DECISIONS.md ("Alembic").
--
-- Not defined here: LangGraph's checkpoint tables. AsyncPostgresSaver.setup()
-- owns those, and it runs in the app lifespan. Two stores is deliberate:
-- the checkpointer holds the agent's working state, the `messages` table below
-- is the durable API-facing transcript.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- ---------------------------------------------------------------------------
-- Reference data (seeded from the problem statement)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS customers (
    customer_id          TEXT PRIMARY KEY,               -- 'cust-001'
    name                 TEXT        NOT NULL,
    email                TEXT        NOT NULL UNIQUE,
    plan                 TEXT        NOT NULL CHECK (plan IN ('free', 'pro', 'enterprise')),
    status               TEXT        NOT NULL CHECK (status IN ('active', 'past_due', 'suspended', 'cancelled', 'trialing')),
    seats_used           INTEGER     NOT NULL,
    seats_limit          INTEGER,                        -- NULL means unlimited (enterprise)
    storage_used_gb      NUMERIC(10, 2) NOT NULL,
    signed_up_on         DATE        NOT NULL,
    billing_cycle        TEXT        CHECK (billing_cycle IN ('monthly', 'annual')),
    monthly_rate_usd     NUMERIC(10, 2) NOT NULL,
    integrations_enabled TEXT[]      NOT NULL DEFAULT '{}',
    workspace_name       TEXT        NOT NULL,
    trial_ends_at        DATE
);

CREATE INDEX IF NOT EXISTS customers_email_lower_idx ON customers (lower(email));
CREATE INDEX IF NOT EXISTS customers_name_lower_idx  ON customers (lower(name));


-- Builds the keyword-search document for one article.
--
-- This exists because a generated column requires an IMMUTABLE expression, and
-- `array_to_string` is marked STABLE. That marking is conservative rather than
-- accurate: its signature is `anyarray`, and some element types have
-- non-immutable output functions. For `text[]` it is genuinely immutable, so
-- wrapping it in a function we mark IMMUTABLE is sound rather than a workaround.
--
-- `'english'::regconfig` is cast explicitly to select the IMMUTABLE two-argument
-- `to_tsvector` overload; the single-argument form resolves the config from a
-- session GUC and is only STABLE.
CREATE OR REPLACE FUNCTION kb_search_document(
    question text,
    response text,
    tags     text[]
) RETURNS tsvector
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT setweight(to_tsvector('english'::regconfig, coalesce(question, '')), 'A') ||
           setweight(to_tsvector('english'::regconfig, coalesce(array_to_string(tags, ' '), '')), 'B') ||
           setweight(to_tsvector('english'::regconfig, coalesce(response, '')), 'C');
$$;


-- The 25 resolved tickets that make up the knowledge base.
CREATE TABLE IF NOT EXISTS kb_articles (
    ticket_id              TEXT PRIMARY KEY,             -- 'TK-001'
    customer_question      TEXT   NOT NULL,
    support_agent_response TEXT   NOT NULL,
    tags                   TEXT[] NOT NULL DEFAULT '{}',
    resolution_type        TEXT   NOT NULL CHECK (resolution_type IN ('self-service', 'informational', 'escalated')),
    category               TEXT   NOT NULL,

    -- Keyword half of hybrid search. Generated so it can never drift from the
    -- source text. Weights: the question matters most (A), then tags (B), then
    -- the response body (C).
    search_vector tsvector GENERATED ALWAYS AS (
        kb_search_document(customer_question, support_agent_response, tags)
    ) STORED,

    -- Semantic half. Populated by scripts/seed_db.py; NULL until then, and the
    -- repository degrades to keyword-only when it is NULL.
    embedding vector(1536)
);

CREATE INDEX IF NOT EXISTS kb_articles_search_idx   ON kb_articles USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS kb_articles_tags_idx     ON kb_articles USING GIN (tags);
CREATE INDEX IF NOT EXISTS kb_articles_category_idx ON kb_articles (category);
CREATE INDEX IF NOT EXISTS kb_articles_question_trgm_idx
    ON kb_articles USING GIN (lower(customer_question) gin_trgm_ops);

-- HNSW rather than IVFFlat: IVFFlat needs representative data to train its
-- lists, and 25 rows is not that. At this size the planner will seq-scan
-- regardless; the index exists so the query does not have to change at 10k rows.
CREATE INDEX IF NOT EXISTS kb_articles_embedding_idx
    ON kb_articles USING hnsw (embedding vector_cosine_ops);


-- ---------------------------------------------------------------------------
-- Conversation state
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status           TEXT        NOT NULL DEFAULT 'active'
                     CHECK (status IN ('active', 'expired', 'escalated', 'closed')),
    -- Set once the agent identifies who it is talking to; NULL before that.
    customer_id      TEXT        REFERENCES customers (customer_id) ON DELETE SET NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Absolute cap, written at creation. The idle cap is derived from
    -- last_activity_at at read time, so changing the idle TTL needs no backfill.
    expires_at       TIMESTAMPTZ NOT NULL,
    metadata         JSONB       NOT NULL DEFAULT '{}'::jsonb
);

-- Drives the expiry sweeper: find live sessions that have gone quiet.
CREATE INDEX IF NOT EXISTS sessions_sweep_idx
    ON sessions (last_activity_at) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS sessions_customer_idx ON sessions (customer_id);


-- Compact per-session memory used to build customer context across conversations.
-- Full transcripts remain in `messages`; only summaries and structured open work
-- are carried into a new session.
CREATE TABLE IF NOT EXISTS customer_session_memories (
    session_id       UUID PRIMARY KEY REFERENCES sessions (id) ON DELETE CASCADE,
    customer_id      TEXT        NOT NULL REFERENCES customers (customer_id) ON DELETE CASCADE,
    summary          TEXT        NOT NULL DEFAULT '',
    unresolved_tasks JSONB       NOT NULL DEFAULT '[]'::jsonb,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS customer_session_memories_customer_idx
    ON customer_session_memories (customer_id, updated_at DESC);


-- The API-facing transcript. `role` uses the vocabulary the spec's response
-- examples use ('customer'/'agent'), not LangGraph's ('human'/'ai');
-- api/schemas.py owns the translation.
CREATE TABLE IF NOT EXISTS messages (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID        NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    turn_index    INTEGER     NOT NULL,
    role          TEXT        NOT NULL CHECK (role IN ('customer', 'agent', 'system')),
    content       TEXT        NOT NULL,
    -- Denormalised onto the message so GET /sessions/{id} needs no join.
    tools_used    TEXT[]      NOT NULL DEFAULT '{}',
    trace_id      UUID,
    latency_ms    INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id, id);


-- ---------------------------------------------------------------------------
-- Tickets created by the agent
-- ---------------------------------------------------------------------------

-- Starts at 1042 so the first ticket is TK-1042, matching the worked example
-- in the problem statement. The seeded knowledge base uses TK-001..TK-025 in a
-- different table, so there is no collision.
CREATE SEQUENCE IF NOT EXISTS ticket_number_seq START WITH 1042;

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   TEXT PRIMARY KEY DEFAULT ('TK-' || nextval('ticket_number_seq')),
    session_id  UUID        REFERENCES sessions (id) ON DELETE SET NULL,
    customer_id TEXT        REFERENCES customers (customer_id) ON DELETE SET NULL,
    subject     TEXT        NOT NULL,
    description TEXT        NOT NULL,
    category    TEXT        NOT NULL,
    priority    TEXT        NOT NULL DEFAULT 'normal'
                CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    status      TEXT        NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'in_progress', 'waiting_on_customer', 'resolved', 'closed')),
    -- Set when the ticket came from escalate_to_human rather than create_ticket.
    escalation_reason TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS tickets_session_idx  ON tickets (session_id);
CREATE INDEX IF NOT EXISTS tickets_customer_idx ON tickets (customer_id);


-- ---------------------------------------------------------------------------
-- Observability: one row per agent turn
-- ---------------------------------------------------------------------------

-- This is the "how would an engineer debug a bad response" answer. `steps` holds
-- the ordered node visits and tool calls for the turn; GET /sessions/{id}/trace
-- replays it. JSONB rather than a child table because it is written once and
-- read whole — there is no query that wants a single step in isolation.
CREATE TABLE IF NOT EXISTS agent_traces (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID        NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    trace_id      UUID        NOT NULL,
    turn_index    INTEGER     NOT NULL,
    steps         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    iterations    INTEGER     NOT NULL DEFAULT 0,
    outcome       TEXT        NOT NULL
                  CHECK (outcome IN ('resolved', 'escalated', 'refused', 'error')),
    latency_ms    INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_traces_session_idx ON agent_traces (session_id, turn_index);
CREATE INDEX IF NOT EXISTS agent_traces_trace_idx   ON agent_traces (trace_id);
