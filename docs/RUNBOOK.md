# Runbook — running and verifying the system

Every command here has been executed against this project, not written from
memory. Where something does not work yet, that is stated rather than implied.

## Current state

The project has a runnable, database-backed support workflow:

| Layer | Runnable today | Why |
|---|---|---|
| Postgres container | **Yes** | `docker-compose.yml` is real |
| Schema | **Yes** | `schema.sql` is real, verified end to end below |
| Seed data | **Yes** | `make seed` locally upserts 15 customers and 25 articles |
| Python toolchain | **Yes** | Install, imports, `ruff`, and test collection all pass |
| FastAPI runtime | **Yes** | `make run`, `/docs`, `/health`, and `/metrics` work |
| Conversation API | **Yes** | Sessions, messages, history, KB answers, and traces work |
| Streamlit UI | **Yes** | `make ui` renders the chat shell and API status |
| Test suite | **Yes** | 42 capability tests and 48 total tests pass; capability tests use real Postgres |

From a fresh clone, `make start` creates local configuration and the virtualenv,
then starts and seeds Postgres plus the API and UI. `make run` and `make ui`
remain available separately; `make test` covers the eleven requested capabilities.

---

## Stage 1 — Database (verified working)

```bash
open -a "Docker Desktop"          # daemon must be up; check with: docker info
docker compose up -d
docker compose ps                 # expect: Up (healthy)
```

Expected — healthy in about 4 seconds:

```
NAME               IMAGE                    STATUS                   PORTS
ravenna_support_agent_db   pgvector/pgvector:pg17   Up 3 seconds (healthy)   0.0.0.0:5434->5432/tcp
```

If it is not healthy, watch the logs: `docker compose logs -f db`.

### Apply the schema

Use the idempotent target:

```bash
make init
```

Verify seven application tables:

```bash
docker compose exec -T db psql -U support -d support_agent -c "\dt"
```

Expected: `agent_traces`, `customer_session_memories`, `customers`,
`kb_articles`, `messages`, `sessions`, and `tickets`.

Re-running the same command must be a clean no-op — it prints `already exists,
skipping` notices and exits 0. That idempotency is what replaces a migration
tool, so it is worth actually checking rather than assuming.

### Open a psql shell

```bash
docker compose exec db psql -U support -d support_agent
```

Or from the host, if you have `psql` installed:

```bash
psql postgresql://support:support@localhost:5434/support_agent
```

---

## Stage 2 — Seed data (verified working)

Load or refresh the provided data:

```bash
make seed
```

Confirm the counts:

```bash
docker compose exec -T db psql -U support -d support_agent -c \
  "SELECT (SELECT count(*) FROM customers) AS customers,
          (SELECT count(*) FROM kb_articles) AS kb_articles;"
```

Expected: `15` customers, `25` articles. Loading this data is also what proves
the schema's `CHECK` constraints match the real values for `plan`, `status`, and
`resolution_type`.

Note the field mapping: the JSON keeps the problem statement's `created_at`,
which becomes `signed_up_on` in our schema so it does not collide with the
row-creation timestamps used elsewhere.

---

## Stage 3 — Retrieval (verified working)

This checks the generated `search_vector` column and the keyword half of hybrid
search.

```bash
docker compose exec -T db psql -U support -d support_agent -c "
SELECT ticket_id, round(ts_rank(search_vector, q)::numeric, 4) AS rank
FROM kb_articles, websearch_to_tsquery('english', '403 forbidden error') q
WHERE search_vector @@ q ORDER BY ts_rank(search_vector, q) DESC LIMIT 3;"
```

Expected: `TK-014`, rank `1.0000`.

```bash
docker compose exec -T db psql -U support -d support_agent -c "
SELECT ticket_id, round(ts_rank(search_vector, q)::numeric, 4) AS rank
FROM kb_articles, websearch_to_tsquery('english', 'how do I reset my password') q
WHERE search_vector @@ q ORDER BY ts_rank(search_vector, q) DESC LIMIT 3;"
```

Expected: `TK-001`, rank `0.9997`.

### The query that proves why hybrid search is needed

```bash
docker compose exec -T db psql -U support -d support_agent -c "
SELECT count(*) AS keyword_hits
FROM kb_articles, websearch_to_tsquery('english', 'I cannot get into my account') q
WHERE search_vector @@ q;"
```

Expected: **`0`**.

This is a real customer phrasing of an account-access problem, and keyword search
returns nothing at all for it — the article that answers it shares no content
words with the question. It is the empirical justification for the embeddings
half of the hybrid, and the reason semantic retrieval is not decoration here.
Once `scripts/seed_db.py` populates the `embedding` column, this same query
should return the account-access articles.

### Ticket numbering

```bash
docker compose exec -T db psql -U support -d support_agent -c "
INSERT INTO tickets (subject, description, category)
VALUES ('smoke test','verifying sequence','billing') RETURNING ticket_id;"
```

Expected: `TK-1042`, matching the worked example in the problem statement.

Clean up after testing, or the first real ticket will not be `TK-1042`:

```bash
docker compose exec -T db psql -U support -d support_agent -c "
DELETE FROM tickets WHERE subject = 'smoke test';
ALTER SEQUENCE ticket_number_seq RESTART WITH 1042;"
```

---

## Stage 4 — Python environment (verified working)

```bash
make install                   # create .env/.venv if absent, then uv sync
```

**You do not need to activate the virtualenv.** Every `make` target invokes
`.venv/bin/<tool>` by path. That is deliberate: on a machine with anaconda
installed, a bare `uvicorn` or `pytest` resolves to anaconda's copy, which cannot
see this project — see the findings log.

Verify the toolchain — all of these pass today:

```bash
make verify                    # compileall + ruff + test collection + row counts
make lint                      # "All checks passed!"
```

Or individually, calling the venv explicitly:

```bash
.venv/bin/python -m compileall -q src tests scripts ui
.venv/bin/ruff check .
.venv/bin/pytest --collect-only -q | tail -1      # "48 tests collected"

.venv/bin/python -c "
import support_agent.main, support_agent.agent.graph, support_agent.tools.registry
import support_agent.api.routes, support_agent.db.pool, support_agent.observability.metrics
print('all packages import cleanly')"
```

If you prefer an activated shell, `source .venv/bin/activate` works too — it just
is not required.

`make test` runs the real database-backed suite; collection-only is useful for a
quick discovery check.

Note the interpreter: `uv` picked CPython 3.13.5 here, which satisfies the
`>=3.12` floor in `pyproject.toml`. The `target-version = "py312"` setting is a
ruff lint target, not a runtime constraint, so this is expected.

---

## Stage 5 — Backend runtime and UI

The API runtime, database-backed conversation path, actions, escalation, session
switching, and UI work today.

```bash
make start   # one command: bootstrap + Postgres + schema + seed + API + UI

# Or run individual components after installation:
make run     # works: uvicorn on http://localhost:8000
make ui      # works: Streamlit on http://localhost:8501
make test    # 48 deterministic tests (42 mapped to the 11 capabilities)
```

Smoke sequence once the backend runs:

```bash
curl -s localhost:8000/health

SID=$(curl -s -XPOST localhost:8000/sessions -H 'content-type: application/json' \
      -d '{}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
echo "session: $SID"

curl -s -XPOST "localhost:8000/sessions/$SID/messages" \
  -H 'content-type: application/json' \
  -d '{"message":"Hi, I need help with my Raven account"}' | python3 -m json.tool

curl -s "localhost:8000/sessions/$SID" | python3 -m json.tool
curl -s "localhost:8000/sessions/$SID/trace" | python3 -m json.tool
curl -s localhost:8000/metrics | head -30
```

What to check at each step:

- `/health` reports the database as reachable.
- The first message returns a clarifying question with `tools_used` **empty** —
  the opener is too vague to act on, so calling a tool here is the defect.
- `reasoning.steps` is populated and ordered, and `trace_id` in the response
  matches an entry from the `/trace` endpoint.
- `GET /sessions/{id}` uses roles `customer` and `agent`, not `human` and `ai`.
- `/metrics` includes `support_agent_turns_total`, not only HTTP counters.

---

## Resetting

```bash
docker compose down -v      # also drops the data volume
docker compose up -d
# then re-apply Stage 1 and Stage 2
```

`make reset` does this in one step once the scripts are implemented.

---

## Findings log

Issues found by actually running the above, kept so they are not rediscovered.

### `schema.sql` — generated column rejected as non-immutable (fixed)

First application of the schema failed with:

```
ERROR:  generation expression is not immutable
```

Cause: the `search_vector` generated column called `array_to_string(tags, ' ')`,
and `array_to_string` is marked `STABLE` in PostgreSQL, not `IMMUTABLE`.
Generated columns require an immutable expression. Confirmed by querying
`pg_proc.provolatile` rather than guessing.

The marking is conservative rather than accurate — the signature is `anyarray`,
and some element types have non-immutable output functions. For `text[]` it is
genuinely immutable.

Fix: extracted the expression into a `kb_search_document()` SQL function marked
`IMMUTABLE`, and cast the text-search config explicitly to `'english'::regconfig`
so the `IMMUTABLE` two-argument `to_tsvector` overload is selected instead of the
`STABLE` single-argument one. Verified by applying the schema three times and
running the retrieval queries in Stage 3.

### `main.py` — import-time side effect made the package unimportable (fixed)

`import support_agent.main` failed with `NotImplementedError`, because the module
ended with `app = create_app()` and that line runs on import.

The immediate symptom was cosmetic — a stubbed factory cannot build an app — but
the underlying shape was worth changing regardless. A module-level `app` means
every import constructs the entire application, including opening a connection
pool, which makes the module awkward to import from a test that wants to build an
app with overridden dependencies.

Fix: removed the module-level `app` and switched to uvicorn's factory mode, so
the Makefile now runs `uvicorn support_agent.main:create_app --factory`. The
factory and runtime lifespan are implemented; startup opens the database pool
without blocking, while `/health` reports whether Postgres is reachable. If you
invoke uvicorn by hand, `--factory` is required — pointing it at
`support_agent.main:app` will fail with an attribute error.

### Makefile — bare commands resolved to anaconda, not the venv (fixed)

`make run` failed with:

```
ModuleNotFoundError: No module named 'support_agent'
```

The traceback pointed at `/Users/Patron/anaconda3/lib/python3.13/site-packages/uvicorn/`
— **anaconda's** uvicorn, not the venv's. `which uvicorn python pytest streamlit`
with no venv active resolved every one of them to `~/anaconda3/bin`. The package
was installed correctly in `.venv`; the wrong interpreter was looking for it.

This is a nasty failure because the error names the *project* and points at an
import, so it reads like a packaging or `src`-layout problem when it is purely
PATH resolution.

Fix: the Makefile calls `.venv/bin/<tool>` by explicit path rather than relying
on an activated shell. Individual targets have a `check-venv` prerequisite with
an actionable error. `make install` bootstraps configuration and dependencies,
while `make start` invokes that bootstrap automatically so a fresh clone needs
only one project command after Docker Desktop is running.

**Related gotcha:** `make run` uses `--reload`, and uvicorn's reloader can keep
the parent process alive when an app factory raises. To diagnose a future
factory startup failure directly, run it without reload:

```bash
.venv/bin/uvicorn support_agent.main:create_app --factory --port 8000
```
