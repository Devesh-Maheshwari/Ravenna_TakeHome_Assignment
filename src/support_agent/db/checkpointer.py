"""Checkpointer setup for the optional LangGraph reference worker.

The live FastAPI endpoint uses `services/conversation.py` and reconstructs model
context from the durable transcript. An alternate worker built from
`agent/graph.py` can instead persist framework working state here, keyed by
`thread_id = session_id`.

This is intentionally *not* the same store as the `messages` table:

  - The checkpointer holds LangGraph's internal state — message objects, pending
    tool calls, channel versions. Its shape is LangGraph's to change.
  - `messages` is the durable, API-shaped transcript with tool metadata and
    timings, which `GET /sessions/{id}` returns.

Coupling the API contract to a framework's internal serialisation format would
be the cheaper choice today and the wrong one on the first LangGraph upgrade.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool


async def build_checkpointer(pool: AsyncConnectionPool) -> AsyncPostgresSaver:
    """Construct the saver over the shared pool and run its one-time migrations.

    `setup()` creates the checkpoint tables if they are absent and is safe to call
    on every boot.
    """
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver
