"""Support tickets created by the agent.

Ticket ids come from a Postgres sequence with a `TK-` prefix, starting at 1042 so
the first one issued is `TK-1042` — matching the worked example in the problem
statement. Generating the id in the database rather than in Python means two
concurrent requests cannot mint the same one.

`escalation_reason` distinguishes a ticket the customer asked for
(`create_ticket`) from one opened because the agent gave up (`escalate_to_human`).
That distinction is what makes the escalation rate measurable.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True, slots=True)
class Ticket:
    ticket_id: str  # 'TK-1042'
    session_id: UUID | None
    customer_id: str | None
    subject: str
    description: str
    category: str
    priority: str  # low | normal | high | urgent
    status: str  # open | in_progress | waiting_on_customer | resolved | closed
    escalation_reason: str | None
    created_at: datetime
    updated_at: datetime


async def create(
    pool: AsyncConnectionPool,
    *,
    subject: str,
    description: str,
    category: str,
    priority: str = "normal",
    session_id: UUID | None = None,
    customer_id: str | None = None,
    escalation_reason: str | None = None,
) -> Ticket:
    """Open a ticket. The database assigns the `TK-` identifier."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO tickets (
                session_id, customer_id, subject, description, category,
                priority, escalation_reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                session_id,
                customer_id,
                subject,
                description,
                category,
                priority,
                escalation_reason,
            ),
        )
        row = await cursor.fetchone()
    return Ticket(**row)


async def get(pool: AsyncConnectionPool, ticket_id: str) -> Ticket | None:
    """Fetch by id. Backs `check_ticket_status`."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM tickets WHERE upper(ticket_id) = upper(%s)", (ticket_id,)
        )
        row = await cursor.fetchone()
    return Ticket(**row) if row else None


async def list_for_session(pool: AsyncConnectionPool, session_id: UUID) -> list[Ticket]:
    """Every ticket opened during one conversation."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM tickets WHERE session_id = %s ORDER BY created_at",
            (session_id,),
        )
        rows = await cursor.fetchall()
    return [Ticket(**row) for row in rows]


async def list_for_customer(
    pool: AsyncConnectionPool,
    customer_id: str,
    *,
    source: str | None = None,
    limit: int = 10,
) -> list[Ticket]:
    """Recent tickets for an account, newest first."""
    source_clause = (
        "AND s.metadata ->> 'source' = %s"
        if source
        else "AND s.metadata ->> 'source' IS NULL"
    )
    source_params: tuple = (source,) if source else ()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT t.*
            FROM tickets t
            JOIN sessions s ON s.id = t.session_id
            WHERE t.customer_id = %s
              {source_clause}
            ORDER BY t.created_at DESC
            LIMIT %s
            """,
            (customer_id, *source_params, limit),
        )
        rows = await cursor.fetchall()
    return [Ticket(**row) for row in rows]
