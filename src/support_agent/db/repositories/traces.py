"""Persistence for per-turn reasoning traces.

This is the storage half of the "how would an engineer debug a bad response"
answer. One row per agent turn, holding the ordered node visits and tool calls
that produced it, replayable through `GET /sessions/{id}/trace` long after the
request is gone.

`steps` is JSONB rather than a child table because a trace is written once and
read whole — no query wants a single step in isolation, so a join buys nothing
and costs a table.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True, slots=True)
class TraceRecord:
    id: int
    session_id: UUID
    trace_id: UUID  # also stamped on every log line for this turn
    turn_index: int
    steps: list[dict[str, Any]]
    iterations: int  # llm<->tools round trips; a runaway loop is visible here
    outcome: str  # resolved | escalated | refused | error
    latency_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime


async def save(
    pool: AsyncConnectionPool,
    *,
    session_id: UUID,
    trace_id: UUID,
    turn_index: int,
    steps: list[dict[str, Any]],
    iterations: int,
    outcome: str,
    latency_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> TraceRecord:
    """Persist one turn's trace.

    Failure to write a trace must never fail the customer's turn — the caller
    logs and continues. Observability is not worth a 500.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO agent_traces (
                session_id, trace_id, turn_index, steps, iterations, outcome,
                latency_ms, input_tokens, output_tokens
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, session_id, trace_id, turn_index, steps, iterations,
                      outcome, latency_ms, input_tokens, output_tokens, created_at
            """,
            (
                session_id,
                trace_id,
                turn_index,
                Jsonb(steps),
                iterations,
                outcome,
                latency_ms,
                input_tokens,
                output_tokens,
            ),
        )
        row = await cursor.fetchone()
    return TraceRecord(**row)


async def list_for_session(pool: AsyncConnectionPool, session_id: UUID) -> list[TraceRecord]:
    """Every turn's trace for a session, in order."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id, session_id, trace_id, turn_index, steps, iterations,
                   outcome, latency_ms, input_tokens, output_tokens, created_at
            FROM agent_traces
            WHERE session_id = %s
            ORDER BY turn_index, id
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
    return [TraceRecord(**row) for row in rows]


async def get_by_trace_id(pool: AsyncConnectionPool, trace_id: UUID) -> TraceRecord | None:
    """Look up a single turn by the id that appears in the logs."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id, session_id, trace_id, turn_index, steps, iterations,
                   outcome, latency_ms, input_tokens, output_tokens, created_at
            FROM agent_traces
            WHERE trace_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (trace_id,),
        )
        row = await cursor.fetchone()
    return TraceRecord(**row) if row else None
