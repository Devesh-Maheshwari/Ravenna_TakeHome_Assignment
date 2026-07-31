"""`create_ticket` and `check_ticket_status`.

Example trigger from the problem statement: "I need to talk to someone about a
billing error."

`create_ticket` is the only tool here that *writes*, which changes how it should
be governed. The prompt requires the agent to offer and get confirmation first —
"I can create a support ticket for you. Would you like me to do that?" — because
a ticket filed on a misread request is visible to the customer and annoying to
undo. The confirmation requirement lives in `agent/prompts.py`; this module just
performs the write once asked.

Ticket ids come from a Postgres sequence, so two concurrent turns cannot mint the
same one. The first is `TK-1042`, matching the problem statement's example.
"""

from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from support_agent.db.repositories import tickets
from support_agent.tools.base import ToolResult, ToolStatus, tool_errors_to_result


@tool_errors_to_result
async def create_ticket(
    subject: str,
    description: str,
    *,
    category: str,
    priority: str = "normal",
    session_id: UUID,
    customer_id: str | None = None,
    pool: AsyncConnectionPool,
) -> ToolResult:
    """Open a support ticket for an issue that needs human follow-up.

    This docstring becomes the tool description the model sees. It must
    convey: use it when the issue cannot be resolved in conversation and the
    customer has agreed to a ticket; do not use it to answer a question the
    knowledge base already covers; always confirm before calling.

    Returns the ticket id so the agent can tell the customer what to reference.
    """
    ticket = await tickets.create(
        pool,
        subject=subject,
        description=description,
        category=category,
        priority=priority,
        session_id=session_id,
        customer_id=customer_id,
    )
    return ToolResult(
        ToolStatus.SUCCESS,
        f"Created support ticket {ticket.ticket_id}.",
        data={
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "priority": ticket.priority,
            "category": ticket.category,
        },
    )


@tool_errors_to_result
async def check_ticket_status(
    ticket_id: str,
    *,
    pool: AsyncConnectionPool,
) -> ToolResult:
    """Look up the current status of an existing ticket.

    This docstring becomes the tool description the model sees. Use it
    when the customer refers to a ticket they already have.

    NOT_FOUND for an unknown id, which the agent should relay plainly rather
    than speculating about.
    """
    ticket = await tickets.get(pool, ticket_id)
    if ticket is None:
        return ToolResult(
            ToolStatus.NOT_FOUND,
            f"No support ticket named {ticket_id} was found.",
        )
    return ToolResult(
        ToolStatus.SUCCESS,
        f"Ticket {ticket.ticket_id} is {ticket.status}.",
        data={
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "priority": ticket.priority,
            "subject": ticket.subject,
            "updated_at": ticket.updated_at.isoformat(),
        },
    )
