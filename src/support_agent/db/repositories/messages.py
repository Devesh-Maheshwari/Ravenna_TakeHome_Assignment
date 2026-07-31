"""The API-facing conversation transcript.

`role` uses the vocabulary from the problem statement's response examples —
`customer` and `agent` — not LangGraph's `human`/`ai`. The translation happens in
`api/schemas.py`, so the wire format never leaks framework naming and a LangGraph
change cannot alter the published API.

`tools_used` is denormalised onto the agent's message so that
`GET /sessions/{id}` renders the full history in one query with no join.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    session_id: UUID
    turn_index: int  # customer message and its agent reply share a turn
    role: str  # customer | agent | system
    content: str
    tools_used: list[str]
    trace_id: UUID | None
    latency_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime


async def append(
    pool: AsyncConnectionPool,
    *,
    session_id: UUID,
    turn_index: int,
    role: str,
    content: str,
    tools_used: list[str] | None = None,
    trace_id: UUID | None = None,
    latency_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> Message:
    """Append one message to the transcript."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO messages (
                session_id, turn_index, role, content, tools_used, trace_id,
                latency_ms, input_tokens, output_tokens
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, session_id, turn_index, role, content, tools_used,
                      trace_id, latency_ms, input_tokens, output_tokens, created_at
            """,
            (
                session_id,
                turn_index,
                role,
                content,
                tools_used or [],
                trace_id,
                latency_ms,
                input_tokens,
                output_tokens,
            ),
        )
        row = await cursor.fetchone()
    return Message(**row)


async def list_for_session(pool: AsyncConnectionPool, session_id: UUID) -> list[Message]:
    """Full history in chronological order. Backs `GET /sessions/{id}`."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id, session_id, turn_index, role, content, tools_used,
                   trace_id, latency_ms, input_tokens, output_tokens, created_at
            FROM messages
            WHERE session_id = %s
            ORDER BY id
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
    return [Message(**row) for row in rows]


async def next_turn_index(pool: AsyncConnectionPool, session_id: UUID) -> int:
    """The turn number the incoming message belongs to (0 for an empty session)."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT COALESCE(max(turn_index) + 1, 0) AS turn_index
            FROM messages
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
    return int(row["turn_index"])
