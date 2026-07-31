# Architecture

This document describes the code that serves the live HTTP endpoint. Design
trade-offs are recorded separately in
[`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md).

## Request path

```text
POST /sessions/{id}/messages
    │
    ├─ api/routes.py                 validate and map the response
    ├─ services/sessions.py          reject closed, escalated, or expired sessions
    ├─ services/conversation.py
    │      ├─ persist the customer message before agent work
    │      ├─ screen safety and scope
    │      ├─ apply deterministic support/action policies
    │      ├─ retrieve or run the bounded model/tool loop when needed
    │      └─ persist reply, memory, trace, usage, and latency
    └─ api/schemas.py                own the public wire format
```

The route is deliberately thin. `handle_message` can be called from tests or
scripts without duplicating turn logic.

## Live agent loop

```text
customer message
      │
      ▼
input guardrail ── blocked ──► refusal
      │ allowed
      ▼
deterministic policy router
      │
      ├─ context answer / clarification / confirmed action
      ├─ account, ticket, or knowledge-base tool
      └─ bounded model ⇄ audited tools
                         │
                         └─ iteration limit or repeated failure ► offer handoff
      │
      ▼
persist response + ordered trace + refresh customer memory
```

Deterministic policies own operations where consistency matters: prompt and
scope rejection, write confirmation, explicit escalation, ticket status,
customer identity, high-confidence retrieval, topic stacks, and pending approval
stacks. Natural language that needs judgment falls through to a model bound to
the same tool registry. Each model/tool round trip consumes one iteration; the
configured maximum prevents an unbounded loop.

The repository also contains a compiled LangGraph implementation under
`src/support_agent/agent/graph.py`. It is directly tested as a reference and can
support an alternate worker, but the FastAPI message endpoint uses the custom
orchestrator above. The assignment permits either approach; documenting this
boundary avoids implying that the live endpoint invokes code it does not.

## Conversation and customer state

State has three different scopes:

| Scope | Examples | Storage |
|---|---|---|
| Turn | tool calls, outcome, token usage, latency | `agent_traces` and response |
| Conversation | transcript, current topic, pending-topic LIFO stack, approval LIFO stack | `messages`, `sessions.metadata` |
| Identified customer | latest-prior summary, unresolved work, source-scoped open tickets | `customer_session_memories` |

The full transcript belongs only to its session. Starting a fresh identified
customer conversation does not copy old messages into the new transcript. It
claims a compact summary of the latest prior session and structured unresolved
work, which gives the agent continuity without presenting days of old chat as a
new conversation. Guest sessions are browser-scoped and do not receive
cross-session customer memory.

When topics are interrupted, `pending_topics` is stored next-first as a LIFO
stack. Completing or handing off the current topic pops the next topic. Pending
write approvals use the same next-first rule in `pending_actions`; only a clear
affirmative performs the top action, and a rejection removes only that action.

## Session lifecycle

Every session has a UUID and two expiry limits:

- Idle TTL: calculated from `last_activity_at` on access.
- Absolute TTL: materialized as `expires_at` when the session is created.

Lazy validation prevents a stale session from answering even between sweeper
runs. A background sweeper changes stale active rows to `expired`. Expiry and
explicit closure are status transitions, not deletion, so historical transcripts
remain retrievable. Posting to a closed, expired, or escalated session returns
HTTP 409 and requires a new session.

For the demo UI, starting a new conversation for an identified customer closes
older active conversations for the same application source in the same database
operation. Other customers and non-UI sources remain isolated.

## Tool design

The assignment requires three tools; the live registry exposes five:

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Ground a how-to, policy, or troubleshooting answer |
| `lookup_customer` | Read plan, status, seats, storage, and integrations |
| `create_ticket` | Create confirmed routine human follow-up |
| `check_ticket_status` | Read an existing ticket status |
| `escalate_to_human` | Stop the bot session and hand control to a person |

All tools return `ToolResult` with distinct `success`, `not_found`, `ambiguous`,
and `error` statuses. An ambiguous customer lookup therefore asks for a stronger
identifier rather than guessing. Tool exceptions become error results so one
integration failure cannot crash the entire HTTP turn.

`POST /messages` exposes tools at two levels:

- `tools_used`: stable distinct names for compatibility and transcript display.
- `tool_calls`: ordered compact objects with `tool`, a redacted/summarised
  `query`, and `result_summary`, matching the assignment's example shape.

The full `reasoning.steps` array also records duration, status, safe structured
result fields, and errors. It exposes decisions and actions, not private model
chain-of-thought.

## Knowledge-base retrieval

The default local path uses PostgreSQL full-text and trigram matching over the
seeded corpus, with lexical-coverage checks that reject confident-looking but
unrelated matches. Optional embeddings can be seeded for semantic retrieval;
support data is not sent to an embedding provider by default.

High-confidence matches answer from the stored support resolution. `TK-005`, the
assignment's known CSV-export issue, deterministically offers a ticket and stores
the confirmation. On approval, the agent creates the ticket and resumes the
interrupted upgrade topic. Unknown codes and weak matches ask for diagnostic
details or fall through to the model instead of presenting the nearest article
as fact.

## Data model

The commented source of truth is
[`src/support_agent/db/schema.sql`](../src/support_agent/db/schema.sql).

| Table | Holds |
|---|---|
| `customers` | Seeded demo accounts |
| `kb_articles` | Resolved support articles and search columns |
| `sessions` | Identity, customer binding, lifecycle, and conversation state |
| `messages` | Durable API-shaped transcripts |
| `tickets` | Created tickets and escalations |
| `agent_traces` | Ordered per-turn reasoning/observability records |
| `customer_session_memories` | Customer/source-scoped summaries and unresolved work |

## Observability

Every completed turn returns a `trace_id`. The same identifier joins the API
response, persisted trace, and structured logs. Prometheus metrics count turn
outcomes and tool calls and observe latency and agent iterations. This lets an
engineer distinguish a retrieval mistake, bad model decision, failed tool,
guardrail refusal, or persistence problem without relying on prose alone.
