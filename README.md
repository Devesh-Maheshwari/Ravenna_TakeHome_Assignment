# Agentic Customer Support Assistant

A multi-turn customer support agent exposed as an HTTP API. It holds a conversation
across turns, decides when to look something up versus answer from context, chains
tool calls, escalates when it is out of its depth, and reports its own reasoning so
a bad response can be debugged after the fact.

Built on **LangChain-compatible tool calling**, **FastAPI** (API), and
**PostgreSQL** (sessions, transcripts, pending work, tickets, traces, and local
full-text/trigram retrieval).

> **Status: runnable and tested.** Multi-turn chat, customer-bound sessions,
> conversation switching/resume, pending-topic recovery, five tools, escalation,
> ambiguity handling, clarification, injection/scope guardrails, traces, and
> operational endpoints work. The suite currently has 48 tests, including 42
> database-backed capability tests mapped to the assignment requirements.

## Quickstart

```bash
open -a "Docker Desktop"          # prerequisite: Docker Desktop and uv
make start
```

From a fresh clone, `make start` creates `.venv` and a safe local `.env` when
missing, synchronizes locked dependencies, starts PostgreSQL, safely applies and
seeds the database, launches FastAPI and Streamlit, and prints a readable guide
to the chat UI, Swagger/ReDoc API documentation, health endpoint, and metrics
endpoint. Add `OPENAI_API_KEY` to `.env` for provider-backed answers; without it,
the documented local fallback remains available. Press Ctrl+C to stop the API
and UI processes it started.

The individual `make up`, `make init`, `make seed`, `make run`, and `make ui`
commands remain available when you want to operate each component separately.

`make test` runs the suite. Tests that need a live LLM are marked `llm` and skipped
by default; everything else runs against a scripted fake model and costs nothing.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/sessions` | Create a conversation session |
| `POST` | `/sessions/{id}/messages` | Send a customer message, get the agent's reply |
| `GET`  | `/sessions` | Find recent browser- or customer-scoped conversations |
| `GET`  | `/sessions/{id}` | Full conversation history and metadata |
| `GET`  | `/sessions/{id}/trace` | Per-turn reasoning trace — the debugging surface |
| `GET`  | `/demo/customers` | Names and ids for the seeded demo-customer picker |
| `GET`  | `/metrics` | Prometheus exposition format |
| `GET`  | `/health` | Liveness plus database reachability |

`POST /sessions/{id}/messages` returns both stable tool names in `tools_used` and
assignment-shaped call details in `tool_calls` (`tool`, redacted/summarised
`query`, and `result_summary`). It also includes the ordered reasoning trace,
escalation status, and a `trace_id` that appears in every related log line.
Interactive docs are available at `/docs` once the server is running.

## Customer memory and conversations

The Streamlit console has two levels of navigation:

1. Select a seeded demo customer (or Guest).
2. Resume the latest active conversation, start a fresh one, or explicitly open
   an older transcript from Conversation history.

A fresh conversation has an empty visible transcript. The agent receives only a
compact summary of the latest prior conversation plus structured unresolved
topics, pending confirmations, and open tickets. Starting fresh closes older
active UI conversations for that customer but preserves their full transcripts
as read-only history. Expired sessions behave the same way: they are never
reopened, while their compact context and unresolved work carry into the next
session. Guest conversations remain browser-scoped and have no durable
customer-account memory.

Conversation history is shown only for identified demo customers. Guest remains
anonymous and browser-scoped, so it has no customer-history picker. UI-created
sessions are also source-isolated from automated test sessions, keeping pytest
traffic out of the demo history and out of customer memory.

Session expiration uses the first limit reached: 30 minutes without a customer
message (idle limit) or 24 hours total (absolute limit). Therefore a conversation
can correctly show as expired on the same calendar day. History labels show the
first customer request, status, local creation time, and a short reference.

## How it works

```
entry → guardrail → llm ⇄ tools → finalize → end
             ↓                        ↑
          (refuse) ──────────────────┘
      any node → escalation → finalize
```

The live path is a custom bounded orchestrator: deterministic routing handles
safety-sensitive actions and common support intents, while a LangChain-compatible
model can select the same audited tools when natural language needs model
judgment. The model/tool cycle is capped by an iteration budget. A separately
tested LangGraph reference implementation remains available for alternate workers;
it is not the HTTP request path.

Five tools, where the spec asks for three:

| Tool | Used when |
|---|---|
| `search_knowledge_base` | The question resembles something support has answered before |
| `lookup_customer` | The answer depends on this customer's plan, status, or usage |
| `create_ticket` | The issue needs human follow-up and the customer confirms |
| `check_ticket_status` | The customer asks about a ticket they already have |
| `escalate_to_human` | The agent is out of its depth, or the customer asks for a person |

## Where things live

| Looking for | Go to |
|---|---|
| The live agent loop | `src/support_agent/services/conversation.py` |
| Tool definitions | `src/support_agent/tools/` |
| Prompts and policies | `src/support_agent/agent/prompts.py` |
| Session lifecycle and expiry | `src/support_agent/services/` |
| Guardrails and prompt-injection defence | `src/support_agent/security/guardrails.py` |
| Endpoints and response shapes | `src/support_agent/api/` |
| Schema | `src/support_agent/db/schema.sql` |
| Why we did not use X | `docs/DESIGN_DECISIONS.md` |
| What is tested and what is not | `docs/TESTING.md` |

## Documentation

- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — **verified** commands for running and checking each layer, plus a findings log
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — agent loop, tool design, session management
- [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) — the four open-ended design challenges, and what we deliberately left out
- [`docs/TESTING.md`](docs/TESTING.md) — coverage, and an honest account of the limitations

> **Implementation status.** The database-backed chat path, actions, switching,
> escalation, local retrieval, and Streamlit UI are covered by executable
> acceptance tests. Remote corpus embeddings are opt-in; the default retrieval
> path keeps support articles local.

## Demo walkthrough

`python scripts/demo_conversation.py` replays the worked example from the problem
statement end to end: a vague opening, a plan question that triggers
`lookup_customer`, a mid-conversation topic shift to a bug that triggers
`search_knowledge_base`, ticket creation on confirmation, and the agent returning to
the parked upgrade topic afterwards.

Run the Streamlit UI alongside it to watch tool calls appear in the sidebar as they
happen — that is the clearest way to show the agent loop on video.
