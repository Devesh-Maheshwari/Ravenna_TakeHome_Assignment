# Testing

## Running

```bash
make up
make init
make test
```

The suite uses the configured local PostgreSQL instance and does not require a
live model call. Model-dependent routing is exercised with deterministic fake
models, while the API, repositories, tools, session lifecycle, and retrieval
queries are real.

## Capability coverage

`tests/test_capabilities.py` contains 42 executable tests mapped to the eleven
requested capabilities with `@pytest.mark.capability(number)`.

| Capability | Covered behavior |
|---|---|
| Multi-turn | Model receives prior turns; topic shifts persist and resume; the assignment's worked flow is deterministic |
| Sessions | UUID isolation, persistence, validation, close, idle/absolute TTL, real-row sweep and expired-session 409 |
| Switching | Browser-scoped listing, customer binding, transcript restoration |
| Escalation | Explicit human request, authority confirmation, repeated failure |
| Actions | Account lookup, KB search, create/check ticket, human handoff, ordered two-tool chaining |
| Reasoning | Ordered trace steps, structured tool-call summaries, stored trace correlation |
| Retrieval | Exact, alternate-phrasing, and typo queries using local search |
| Ambiguity | Multiple customer matches return candidates rather than guessing |
| Clarification | Vague inputs receive exactly one focused question |
| Injection | Override, extraction, reassignment, and obfuscated patterns |
| Scope | Poetry, math, weather, and programming requests are refused |

## Test design

- Real PostgreSQL is used because full-text, trigram, arrays, JSONB, and ticket
  sequencing should not be simulated by mocks.
- Real LLM calls are excluded from `make test` because they are nondeterministic
  and billable. Fake responses test history and multi-step tool calling.
- Knowledge-base retrieval is local by default. Corpus embeddings are opt-in so
  support articles are not uploaded without explicit approval.
- Assertions target behavior and tool paths rather than exact model prose.

## Remaining limits

- The heuristic injection and scope checks are defense in depth, not a complete
  security assessment.
- The corpus contains only 25 articles; relevance thresholds must be retuned at
  production scale.
- There is no load, rate-limit, or sustained concurrency test yet.
- The Streamlit UI has lightweight state-value regressions plus live browser
  acceptance coverage; most business behavior is tested at the API boundary.
- A direct provider smoke call should be run after changing model credentials:

```bash
.venv/bin/python -c \
  "from support_agent.agent.llm import build_chat_model; from support_agent.config import get_settings; print(build_chat_model(get_settings()).invoke('Reply with only OK').content)"
```
