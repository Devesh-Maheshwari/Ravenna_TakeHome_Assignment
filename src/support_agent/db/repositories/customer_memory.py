"""Durable, compact customer context across otherwise isolated conversations."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True, slots=True)
class CustomerContext:
    summary: str
    unresolved_tasks: list[dict[str, Any]]
    summary_source_session_id: str | None = None


async def claim_for_new_session(
    pool: AsyncConnectionPool,
    *,
    customer_id: str,
    session_id: UUID,
    source: str | None = None,
    summary_limit: int = 1,
) -> CustomerContext:
    """Move open work to a new session and return compact prior summaries."""
    source_clause = (
        "s.metadata ->> 'source' = %s"
        if source
        else "s.metadata ->> 'source' IS NULL"
    )
    source_params: tuple = (source,) if source else ()
    async with pool.connection() as conn, conn.transaction():
        cursor = await conn.execute(
            f"""
            SELECT memory.session_id, memory.summary, memory.unresolved_tasks
            FROM customer_session_memories memory
            JOIN sessions s ON s.id = memory.session_id
            WHERE memory.customer_id = %s AND memory.session_id <> %s
              AND {source_clause}
            ORDER BY memory.updated_at DESC
            FOR UPDATE
            """,
            (customer_id, session_id, *source_params),
        )
        rows = await cursor.fetchall()
        summary_rows = [row for row in rows[:summary_limit] if row["summary"]]
        summaries = [row["summary"] for row in reversed(summary_rows)]
        summary_source_session_id = (
            str(summary_rows[0]["session_id"]) if summary_rows else None
        )
        tasks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            for task in row["unresolved_tasks"] or []:
                key = str(task)
                if key not in seen:
                    tasks.append(task)
                    seen.add(key)

        await conn.execute(
            f"""
            UPDATE customer_session_memories
            SET unresolved_tasks = '[]'::jsonb, updated_at = now()
            WHERE customer_id = %s AND session_id <> %s
              AND session_id IN (
                  SELECT id FROM sessions s WHERE {source_clause}
              )
              AND unresolved_tasks <> '[]'::jsonb
            """,
            (customer_id, session_id, *source_params),
        )
        await conn.execute(
            """
            INSERT INTO customer_session_memories (
                session_id, customer_id, summary, unresolved_tasks
            )
            VALUES (%s, %s, '', %s)
            ON CONFLICT (session_id) DO UPDATE SET
                unresolved_tasks = EXCLUDED.unresolved_tasks,
                updated_at = now()
            """,
            (session_id, customer_id, Jsonb(tasks)),
        )
    return CustomerContext(
        summary="\n".join(summaries)[-1000:],
        unresolved_tasks=tasks,
        summary_source_session_id=summary_source_session_id,
    )


async def upsert(
    pool: AsyncConnectionPool,
    *,
    session_id: UUID,
    customer_id: str,
    summary: str,
    unresolved_tasks: list[dict[str, Any]],
) -> None:
    """Replace one session's compact summary and current open-work snapshot."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO customer_session_memories (
                session_id, customer_id, summary, unresolved_tasks
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                summary = EXCLUDED.summary,
                unresolved_tasks = EXCLUDED.unresolved_tasks,
                updated_at = now()
            """,
            (session_id, customer_id, summary, Jsonb(unresolved_tasks)),
        )


async def get_for_session(
    pool: AsyncConnectionPool,
    session_id: UUID,
) -> CustomerContext | None:
    """Inspect one session memory; used by tests and support diagnostics."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT summary, unresolved_tasks
            FROM customer_session_memories
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    return CustomerContext(
        summary=row["summary"],
        unresolved_tasks=list(row["unresolved_tasks"] or []),
        summary_source_session_id=None,
    )
