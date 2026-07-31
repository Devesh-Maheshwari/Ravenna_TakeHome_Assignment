"""`escalate_to_human` — the model-initiated half of the handoff mechanism.

Distinct from `create_ticket`. A ticket is routine async follow-up on a
well-understood issue. An escalation is the agent saying it should not be the one
handling this — because the customer asked for a person, because the request
needs authority the agent does not have (refunds, account recovery, billing
disputes), or because it has tried and failed.

Making it a separate tool rather than a flag on `create_ticket` means the model
has to make the distinction explicitly, and it makes escalation rate directly
measurable rather than inferred from ticket text.

The live deterministic counterpart lives in `services/conversation.py`; the
optional LangGraph reference worker has the equivalent policy in
`agent/nodes/escalation.py`.
"""

from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from support_agent.db.repositories import sessions, tickets
from support_agent.tools.base import ToolResult, ToolStatus, tool_errors_to_result


@tool_errors_to_result
async def escalate_to_human(
    reason: str,
    conversation_summary: str,
    *,
    urgency: str = "normal",
    session_id: UUID,
    customer_id: str | None = None,
    pool: AsyncConnectionPool,
) -> ToolResult:
    """Hand this conversation to a human support agent.

    Use this when the customer explicitly asks for a person, the request needs
    authority the assistant does not have, or repeated attempts cannot resolve
    the issue. Prefer `create_ticket` for routine follow-up on a understood issue.

    Flips the session to `escalated`, opens a ticket carrying the reason and the
    summary so the human starts informed, and returns the reference to give the
    customer.
    """
    priority = urgency if urgency in {"low", "normal", "high", "urgent"} else "normal"
    ticket = await tickets.create(
        pool,
        subject="Human support escalation",
        description=conversation_summary,
        category="escalation",
        priority=priority,
        session_id=session_id,
        customer_id=customer_id,
        escalation_reason=reason,
    )
    await sessions.set_status(pool, session_id, "escalated")
    return ToolResult(
        ToolStatus.SUCCESS,
        f"Escalated to a human support agent under ticket {ticket.ticket_id}.",
        data={
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "reason": reason,
        },
    )
