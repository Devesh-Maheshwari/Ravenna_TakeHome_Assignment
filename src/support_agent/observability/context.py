"""Ambient request context.

`contextvars`, not middleware state or threadlocals, because the work is async and
a turn hops between the route handler, orchestrator, model, and several tool
coroutines. A `ContextVar` follows the task across all of them, so nothing
has to thread a `trace_id` parameter through six call signatures purely so the
logger can see it.

The `trace_id` set here is the same one written to `agent_traces.trace_id` and
returned in the API response. One identifier joins the log line, the persisted
trace, and what the customer's client received.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID, uuid4

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


@contextmanager
def turn_context(*, trace_id: UUID, session_id: UUID) -> Iterator[None]:
    """Bind trace and session ids for the duration of one turn, then restore."""
    trace_token = trace_id_var.set(str(trace_id))
    session_token = session_id_var.set(str(session_id))
    try:
        yield
    finally:
        session_id_var.reset(session_token)
        trace_id_var.reset(trace_token)


def current_context() -> dict[str, str]:
    """The bound ids, for the structlog processor to merge into every event."""
    values = {
        "trace_id": trace_id_var.get(),
        "session_id": session_id_var.get(),
        "request_id": request_id_var.get(),
    }
    return {key: value for key, value in values.items() if value is not None}


def new_trace_id() -> UUID:
    """Mint a trace id for a turn."""
    return uuid4()
