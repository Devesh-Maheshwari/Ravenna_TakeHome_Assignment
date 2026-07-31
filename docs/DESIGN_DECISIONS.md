# Design Decisions

Two halves: how the four open-ended design challenges were addressed, and what
was deliberately left out. The second half matters as much as the first — an
absence with no reasoning attached reads as an oversight.

Context for every decision below: this was built to a **3–4 hour budget**. The
goal was a small, complete, legible system rather than a large partial one.

---

## Part 1 — The four open-ended design challenges

The problem statement asks for at least two. All four are addressed.

### Guardrails and boundaries

Two layers, and they are not equally strong.

**The structural layer** (`agent/prompts.py`) is the one that matters. The system
prompt is the only text with authority. Customer messages, knowledge-base article
bodies, and tool output are all delimited and labelled as untrusted *data* before
they enter the model's context. This holds against phrasings nobody anticipated,
because it does not depend on recognising anything.

**The heuristic layer** (`security/guardrails.py`) screens input before the model
sees it, for instruction-override attempts, system-prompt extraction, role
reassignment, and off-topic requests. By construction it only catches what it has
seen before.

Being honest about that ordering is the point. A regex list sold as a solution is
worse than one sold as defence in depth, because it invites trusting it. The
realistic claim: heuristics catch obvious attempts, raise the effort required for
the rest, and make attempts visible in `support_agent_guardrail_blocks_total`.

Screening runs **before** the model call, not after. Screening a generated
response means having already paid for the tokens, and risks a partial answer
leaking before the check completes.

Both checks are tuned to **under-block**. "Ignore what I said before, my actual
problem is..." is a normal support message. So is a customer pasting a stack
trace. Refusing a real customer is a worse product failure than answering a
borderline question, and heuristic screens drift toward over-blocking unless
something holds them accountable — which is why roughly half of
`tests/test_capabilities.py` also asserts that the guardrail does *not* fire.

The residual risk we did not close: **indirect injection through the knowledge
base**. Our corpus is fixed and curated, so it is theoretical here. A real
deployment indexes customer-written tickets, at which point the knowledge base
becomes an untrusted input channel and the structural framing is doing all the
work.

### Ambiguity resolution

Handled at the tool boundary rather than only in the prompt, because prompt
instructions about uncertainty are advisory and data structures are not.

`ToolResult` (`tools/base.py`) makes `AMBIGUOUS` a first-class status alongside
`SUCCESS`, `NOT_FOUND`, and `ERROR`. A tool that returns `None` for "no match"
and `None` for "seven matches" forces the model to guess, and it will guess
fluently. A name search matching several accounts returns the candidates and lets
the agent ask which one — returning the first row would be a cross-customer data
leak wearing a helpful face.

`NOT_FOUND` and `ERROR` are also kept distinct. "No such customer" is information
to relay; "the database is unreachable" is a reason to escalate. Collapsing them
produces an agent that apologises for the customer not existing.

Two prompt-level rules complete it: ask one focused question rather than a list,
preferring the question that narrows the space most; and never take a write
action on an ambiguous request — ticket creation is offered and confirmed, never
inferred.

### Handoff and escalation

Recognition comes from complementary model and deterministic policies, because
each covers the other's failure mode.

**Model judgment** — the bounded model/tool loop can select
`escalate_to_human` for natural-language cases that do not match a fixed route.
It is good at nuance but unreliable at admitting defeat.

**Deterministic policies** (`services/conversation.py`) — explicit requests for
a person hand off immediately; refunds, account recovery, and other
authority-requiring requests ask for confirmation; repeated provider failures
offer escalation; and iteration-budget exhaustion ends safely instead of
looping. Weak or missing retrieval asks for details or falls through to model
judgment rather than treating the nearest article as truth.

A model that never gives up is stopped by the budget. A fixed rule alone misses
nuance. Together they produce a bounded path with an explicit human handoff.

`escalate_to_human` is a separate tool from `create_ticket` rather than a flag on
it, which forces the model to make the distinction explicitly and makes
escalation rate directly measurable instead of inferred from ticket text.

A handoff flips the session to `escalated`, opens a ticket carrying the reason
and a conversation summary so the human does not start cold, and tells the
customer plainly what happens next.

### Observability and debugging

Three layers, all self-hosted, each answering a question the others cannot.

| Layer | Where | Answers |
|---|---|---|
| **Traces** — one row per turn in `agent_traces`, replayable at `GET /sessions/{id}/trace` | `agent/trace.py` | *Why was **this** response wrong?* Every node visit and tool call, with arguments, summarised result, and duration. |
| **Metrics** — Prometheus at `GET /metrics` | `observability/metrics.py` | *Is something wrong **right now**?* Tool error rate, escalation rate, iteration distribution, token spend. |
| **Logs** — structlog JSON | `observability/logging.py` | The connective tissue. |

The `trace_id` is the same value in the API response, in every related log line,
and in the persisted trace row. Without one identifier spanning all three, three
observability layers are three disconnected islands.

The metrics are deliberately **domain** metrics, not generic HTTP counts. Request
rate and 5xx rate say nothing about an agent that returns `200` with a
confidently wrong answer. Escalation rate, tool error rate, and iteration count
do. That difference is the whole distinction between monitoring a web service and
monitoring an agent.

What a trace records and what it does not: tool *arguments* are captured, because
"why did it search for that?" is the most common debugging question. Tool
*results* are summarised, because three knowledge-base articles inlined verbatim
would bury the answer. Model reasoning is represented by the decisions made —
which tool, which arguments, which order — not by chain-of-thought text, which is
both what we can honestly expose and what is actually useful.

Writing a trace is best-effort. A failure to record observability must never fail
a turn the customer would otherwise have received.

---

## Part 2 — Deliberately not built

Each of these is a real tool with real value. None earns its cost at this scope.

### Alembic (schema migrations)

**What it buys you.** Versioned, reversible schema evolution. It lets a team
change a live schema without downtime or data loss, gives a readable history of
why the schema looks the way it does, and makes rollback a command instead of an
incident. Non-negotiable the moment there is production data you cannot drop.

**Why not here.** The schema is authored once and never evolves within the
assignment. There is no production data to preserve and no second developer to
coordinate with. Alembic would contribute exactly one revision file plus an async
`env.py` that is notoriously fiddly to configure — cost with no corresponding
benefit. `db/schema.sql` applied idempotently by `scripts/init_db.py` produces
the identical database in a form a reviewer reads top to bottom in thirty seconds.

**What would change our mind.** The first deploy to an environment whose data we
cannot drop.

### SQLAlchemy ORM

**What it buys you.** Typed models, relationship loading, a unit of work and
identity map, database portability, injection safety by construction, and the
elimination of a lot of hand-written CRUD. On a schema with deep relationships
and many contributors it pays for itself quickly.

**Why not here.** Six tables, no polymorphism, no deep relationship graph. The
one query that actually matters — hybrid keyword-plus-vector retrieval with rank
fusion — is hand-written SQL under any ORM, so the ORM would abstract the easy
half and step aside for the hard half. Pydantic already provides typed structures
at the API boundary, so the typing benefit is largely duplicated. Async
SQLAlchemy also brings greenlet and lazy-loading failure modes that cost
debugging time a four-hour build does not have. Parameterised psycopg queries
retain the injection safety without the layer.

**What would change our mind.** Relationships appearing in the schema, or several
developers writing queries and needing guardrails.

### asyncpg

**What it buys you.** The fastest Postgres driver for Python by a clear margin,
and the right default for a high-throughput data-plane service.

**Why not here.** The optional LangGraph reference worker's
`AsyncPostgresSaver` is built on psycopg3. Choosing asyncpg would require a
second driver and pool if that worker is enabled. More importantly, the live
path's bottleneck is the LLM call, measured in hundreds of milliseconds to
seconds; driver overhead is not the useful optimization target. One psycopg3
stack is simpler for both the live repositories and the reference worker.

**What would change our mind.** A read path hot enough that driver overhead shows
up in a profile.

### Datadog, Laminar, or any hosted APM

**What it buys you.** Hosted dashboards, alerting, long retention, cross-service
correlation, and on-call integration. Genuinely valuable once a service is live
and someone can be paged for it.

**Why not here.** Two reasons, and the second is the real one.

First, a reviewer cannot run our Datadog account, so anything built there is
invisible to them and demonstrates nothing.

Second: these products are **destinations, not instrumentation**. The engineering
work is deciding which signals are worth emitting — and that is exactly what the
three layers in Part 1 do. Shipping those signals to a vendor is a configuration
change, not a design change. Our metrics are already in Prometheus exposition
format and our logs are already structured JSON carrying a trace id, so pointing
Grafana, Datadog, or an OpenTelemetry collector at this service is a scrape
config with no application change.

**The one exception we do wire up** is LangSmith, behind two environment
variables and off by default. It is LLM-native, LangChain instruments it with
zero code from us, and it makes prompt and tool-call inspection visible during
the demo.

**What would change our mind.** A real deployment with an on-call rotation.

### A Dockerfile for the API

**What it buys you.** A reproducible deploy artifact and dev/prod parity.

**Why not here.** The reviewer's flow is clone, `make up`, `make run`.
Containerising the API adds a build step and a rebuild-on-every-edit loop during
development, in exchange for parity with a production environment that does not
exist. Postgres *is* containerised, because it is the dependency that is
genuinely painful to install by hand.

### Retention and purging of expired sessions

Expiry is a **status transition, not a delete**. Transcripts survive for support
review and for the evaluation suite. Actually purging old rows is a retention
policy — it depends on a legal answer about how long customer conversations may
be kept, which is not an engineering decision to make unilaterally in a take-home.

---

## Part 3 — The deliberate luxury

Hybrid knowledge-base search (keyword + embeddings, fused by reciprocal rank) is
the one thing here that exceeds the requirement.

It earns its place because the dataset rewards it. `"403 forbidden error"` is a
keyword match — the literal token appears in TK-014. `"I can't get into my
account"` is a semantic one — it shares no content words with the article that
answers it. Both shapes appear in the problem statement's own examples, and
running only one half of the hybrid loses a class of question entirely.

RRF is used rather than a weighted score blend because `tsvector` ranks and
cosine distances are not on comparable scales; fusing by *rank* sidesteps having
to normalise them against each other.

It is also the marked **cut line**. If the clock had run out, keyword-only still
satisfies the specification and nothing else would have had to change — the
repository degrades to keyword search on its own when `embedding` is NULL.
