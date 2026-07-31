"""Build compact customer memory without copying old transcripts into new chats."""

from typing import Any
from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from support_agent.db.repositories import customer_memory, messages, sessions, tickets


def _compact(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


async def refresh(pool: AsyncConnectionPool, session_id: UUID) -> None:
    """Snapshot the session summary and structured unresolved work."""
    session = await sessions.get(pool, session_id)
    if session is None or session.customer_id is None:
        return
    history = await messages.list_for_session(pool, session_id)
    customer_messages = [message.content for message in history if message.role == "customer"]
    agent_messages = [message.content for message in history if message.role == "agent"]
    summary_parts: list[str] = []
    if customer_messages:
        summary_parts.append(
            "Customer requests: " + " | ".join(_compact(item) for item in customer_messages[-2:])
        )
    if agent_messages:
        summary_parts.append("Latest support response: " + _compact(agent_messages[-1]))
    summary = " ".join(summary_parts)

    metadata = session.metadata
    tasks: list[dict[str, Any]] = []
    pending_actions = metadata.get("pending_actions")
    if not isinstance(pending_actions, list) or not pending_actions:
        single_action = metadata.get("pending_action")
        pending_actions = [single_action] if isinstance(single_action, dict) else []
    for pending_action in pending_actions:
        if not isinstance(pending_action, dict):
            continue
        tasks.append(
            {
                "kind": "pending_action",
                "source_session_id": str(session_id),
                "action": pending_action,
            }
        )
    topics = []
    if metadata.get("current_topic"):
        topics.append(metadata["current_topic"])
    topics.extend(metadata.get("pending_topics", []))
    for topic in dict.fromkeys(topics):
        tasks.append(
            {
                "kind": "pending_topic",
                "source_session_id": str(session_id),
                "topic": topic,
            }
        )
    for ticket in await tickets.list_for_customer(
        pool,
        session.customer_id,
        source=metadata.get("source"),
    ):
        if ticket.status not in {"resolved", "closed"}:
            tasks.append(
                {
                    "kind": "open_ticket",
                    "ticket_id": ticket.ticket_id,
                    "status": ticket.status,
                    "subject": ticket.subject,
                }
            )

    await customer_memory.upsert(
        pool,
        session_id=session_id,
        customer_id=session.customer_id,
        summary=summary,
        unresolved_tasks=tasks,
    )
    await sessions.merge_metadata(
        pool,
        session_id,
        {"unresolved_tasks": tasks},
    )
