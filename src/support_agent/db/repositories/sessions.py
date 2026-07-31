"""Session records: the conversation's identity, status, and lifetime.

Two independent expiry rules, both enforced here:

  - **Idle TTL** — derived at read time from `last_activity_at`, so changing the
    setting takes effect immediately with no backfill.
  - **Absolute TTL** — written into `expires_at` at creation, so a session that
    is chatty forever still terminates.

`status` is the escalation surface too: `escalate_to_human` flips it to
`escalated`, which is what stops the agent from continuing to answer once a
human has been brought in.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True, slots=True)
class Session:
    id: UUID
    status: str  # active | expired | escalated | closed
    customer_id: str | None  # set once the agent identifies the customer
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    expires_at: datetime
    metadata: dict[str, Any]


async def create(
    pool: AsyncConnectionPool,
    *,
    expires_at: datetime,
    metadata: dict[str, Any] | None = None,
) -> Session:
    """Insert a new active session. The database assigns the UUID."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO sessions (expires_at, metadata)
            VALUES (%s, %s)
            RETURNING id, status, customer_id, created_at, updated_at,
                      last_activity_at, expires_at, metadata
            """,
            (expires_at, Jsonb(metadata or {})),
        )
        row = await cursor.fetchone()
    return Session(**row)


async def get(pool: AsyncConnectionPool, session_id: UUID) -> Session | None:
    """Fetch by id without touching activity timestamps."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id, status, customer_id, created_at, updated_at,
                   last_activity_at, expires_at, metadata
            FROM sessions
            WHERE id = %s
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
    return Session(**row) if row else None


async def touch(pool: AsyncConnectionPool, session_id: UUID) -> None:
    """Bump `last_activity_at`, deferring idle expiry. Called on each message."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET last_activity_at = now(), updated_at = now()
            WHERE id = %s
            """,
            (session_id,),
        )


async def set_status(pool: AsyncConnectionPool, session_id: UUID, status: str) -> None:
    """Transition status — used by escalation, closure, and expiry."""
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE sessions SET status = %s, updated_at = now() WHERE id = %s",
            (status, session_id),
        )


async def attach_customer(pool: AsyncConnectionPool, session_id: UUID, customer_id: str) -> None:
    """Bind an identified customer to the session so later turns can skip lookup."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET customer_id = %s, updated_at = now()
            WHERE id = %s
            """,
            (customer_id, session_id),
        )


async def close_other_active(
    pool: AsyncConnectionPool,
    *,
    customer_id: str,
    except_session_id: UUID,
    source: str | None = None,
) -> int:
    """Supersede older active UI conversations while preserving their history."""
    source_clause = (
        "AND metadata ->> 'source' = %s"
        if source
        else "AND metadata ->> 'source' IS NULL"
    )
    source_params: tuple = (source,) if source else ()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            UPDATE sessions
            SET status = 'closed', updated_at = now()
            WHERE customer_id = %s
              AND id <> %s
              AND status = 'active'
              {source_clause}
            """,
            (customer_id, except_session_id, *source_params),
        )
    return cursor.rowcount


async def expire_stale(
    pool: AsyncConnectionPool,
    *,
    idle_cutoff: datetime,
    now: datetime,
) -> int:
    """Mark active sessions expired past either TTL. Returns the count expired.

    Both cutoffs are passed in rather than computed here so the sweeper and the
    tests agree on what "now" means.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            UPDATE sessions
            SET status = 'expired', updated_at = %s
            WHERE status = 'active'
              AND (last_activity_at < %s OR expires_at <= %s)
            """,
            (now, idle_cutoff, now),
        )
    return cursor.rowcount


async def count_active(pool: AsyncConnectionPool) -> int:
    """Backs the `support_agent_active_sessions` gauge."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT count(*) AS count FROM sessions WHERE status = 'active'"
        )
        row = await cursor.fetchone()
    return int(row["count"])


async def merge_metadata(
    pool: AsyncConnectionPool,
    session_id: UUID,
    updates: dict[str, Any],
) -> Session:
    """Merge application state such as pending work into a session's JSON metadata."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            UPDATE sessions
            SET metadata = metadata || %s, updated_at = now()
            WHERE id = %s
            RETURNING id, status, customer_id, created_at, updated_at,
                      last_activity_at, expires_at, metadata
            """,
            (Jsonb(updates), session_id),
        )
        row = await cursor.fetchone()
    return Session(**row)


async def list_recent(
    pool: AsyncConnectionPool,
    *,
    client_id: str | None = None,
    customer_id: str | None = None,
    status: str | None = None,
    source: str | None = None,
    limit: int = 50,
) -> list[Session]:
    """Recent sessions for browser or customer-scoped conversation switching."""
    conditions: list[str] = []
    values: list = []
    if client_id:
        conditions.append("metadata ->> 'client_id' = %s")
        values.append(client_id)
    if customer_id:
        conditions.append("customer_id = %s")
        values.append(customer_id)
    if status:
        conditions.append("status = %s")
        values.append(status)
    if source:
        conditions.append("metadata ->> 'source' = %s")
        values.append(source)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    values.append(limit)
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT id, status, customer_id, created_at, updated_at,
                   last_activity_at, expires_at, metadata
            FROM sessions
            {where}
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            tuple(values),
        )
        rows = await cursor.fetchall()
    return [Session(**row) for row in rows]
