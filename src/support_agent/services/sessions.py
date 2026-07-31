"""Session lifecycle.

Sits between the API layer and the sessions repository, owning the policy the
repository deliberately does not: what counts as expired, and what an expired
session does when someone posts to it.

Expiry is checked lazily on access as well as swept in the background. The
sweeper alone would leave a window where an expired session still answers; the
lazy check alone would leave dead rows accumulating forever. Both, together,
because they fail in opposite directions.

Posting to an expired session is a 409, not a silent resurrection. The customer
gets a new session and a clear reason, rather than an agent that has mysteriously
forgotten the conversation.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from support_agent.api.errors import SessionExpired, SessionNotFound
from support_agent.config import Settings
from support_agent.db.repositories import customer_memory, customers, sessions
from support_agent.db.repositories.sessions import Session


async def create_session(
    pool: AsyncConnectionPool,
    settings: Settings,
    *,
    metadata: dict | None = None,
) -> Session:
    """Open a session, computing `expires_at` from the absolute TTL."""
    expires_at = datetime.now(UTC) + timedelta(hours=settings.session_absolute_ttl_hours)
    session = await sessions.create(pool, expires_at=expires_at, metadata=metadata)
    supplied = metadata or {}
    customer_id = supplied.get("customer_id")
    email = supplied.get("email")
    customer = None
    if customer_id:
        customer = await customers.get_by_id(pool, customer_id)
    elif email:
        customer = await customers.get_by_email(pool, email)
    if customer:
        await sessions.attach_customer(pool, session.id, customer.customer_id)
        if supplied.get("supersede_active"):
            await sessions.close_other_active(
                pool,
                customer_id=customer.customer_id,
                except_session_id=session.id,
                source=supplied.get("source"),
            )
        context = await customer_memory.claim_for_new_session(
            pool,
            customer_id=customer.customer_id,
            session_id=session.id,
            source=supplied.get("source"),
        )
        memory_updates: dict = {
            "customer_name": customer.name,
            "customer_email": customer.email,
            "prior_summary": context.summary,
            "prior_summary_source_session_id": context.summary_source_session_id,
            "unresolved_tasks": context.unresolved_tasks,
        }
        actions = [
            task["action"]
            for task in context.unresolved_tasks
            if task.get("kind") == "pending_action" and isinstance(task.get("action"), dict)
        ]
        topics = [
            task["topic"]
            for task in context.unresolved_tasks
            if task.get("kind") == "pending_topic" and task.get("topic")
        ]
        if actions:
            memory_updates["pending_action"] = actions[0]
            memory_updates["pending_actions"] = actions
        if topics:
            memory_updates["current_topic"] = topics[0]
            memory_updates["pending_topics"] = list(dict.fromkeys(topics[1:]))
        session = await sessions.merge_metadata(
            pool,
            session.id,
            memory_updates,
        )
    return session


async def get_active_session(
    pool: AsyncConnectionPool,
    settings: Settings,
    session_id: UUID,
) -> Session:
    """Fetch a session and assert it is usable.

    Raises `SessionNotFound` / `SessionExpired` (see `api/errors.py`) rather than
    returning None, so every caller handles both cases instead of forgetting one.
    Applies the lazy expiry check and persists the transition if it has lapsed.
    """
    session = await sessions.get(pool, session_id)
    if session is None:
        raise SessionNotFound("Conversation session not found.")
    if session.status == "expired" or is_expired(session, settings, now=datetime.now(UTC)):
        if session.status == "active":
            await sessions.set_status(pool, session_id, "expired")
        raise SessionExpired("This conversation has expired. Please start a new one.")
    if session.status != "active":
        raise SessionExpired(f"This conversation is {session.status}. Please start a new one.")
    return session


def is_expired(session: Session, settings: Settings, *, now: datetime) -> bool:
    """Either TTL breached: idle since `last_activity_at`, or past `expires_at`."""
    idle_cutoff = now - timedelta(minutes=settings.session_idle_ttl_minutes)
    return session.expires_at <= now or session.last_activity_at <= idle_cutoff


async def record_activity(pool: AsyncConnectionPool, session_id: UUID) -> None:
    """Bump activity, deferring idle expiry. Called once per customer message."""
    await sessions.touch(pool, session_id)


async def close_session(pool: AsyncConnectionPool, session_id: UUID) -> None:
    """Close a session that finished cleanly."""
    await sessions.set_status(pool, session_id, "closed")
