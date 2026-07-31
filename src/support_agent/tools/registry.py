"""Tool registration — the one place that says which tools exist.

Every tool needs a live connection pool, which a LangChain tool signature has no
room for. `build_tools` closes over the pool and session context for one agent
run, so the model sees clean argument schemas (`query`, `email`, `subject`)
with no infrastructure leaking into them.

Keeping the list in one function also means the tool surface is auditable at a
glance: five entries, and adding a sixth is a visible change rather than a
decorator discovered somewhere in the tree.
"""

from uuid import UUID

from langchain_core.tools import BaseTool, StructuredTool
from psycopg_pool import AsyncConnectionPool

from support_agent.tools.customer import lookup_customer
from support_agent.tools.escalation import escalate_to_human
from support_agent.tools.knowledge_base import search_knowledge_base
from support_agent.tools.tickets import check_ticket_status, create_ticket

# Tool names, referenced by prompts, metrics labels, and tests. Defined here so a
# rename is a single edit rather than a string hunt.
SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
LOOKUP_CUSTOMER = "lookup_customer"
CREATE_TICKET = "create_ticket"
CHECK_TICKET_STATUS = "check_ticket_status"
ESCALATE_TO_HUMAN = "escalate_to_human"

ALL_TOOL_NAMES: tuple[str, ...] = (
    SEARCH_KNOWLEDGE_BASE,
    LOOKUP_CUSTOMER,
    CREATE_TICKET,
    CHECK_TICKET_STATUS,
    ESCALATE_TO_HUMAN,
)

# Tools that mutate state. The prompt requires confirmation before these, and
# tests assert they are never reached from a guardrail-refused turn.
WRITE_TOOLS: frozenset[str] = frozenset({CREATE_TICKET, ESCALATE_TO_HUMAN})


def build_tools(
    *,
    pool: AsyncConnectionPool,
    session_id: UUID,
    customer_id: str | None = None,
) -> list[BaseTool]:
    """Build the tool list bound to this session's context."""

    async def search(query: str, category: str | None = None, limit: int = 3):
        """Search support articles for how-to, troubleshooting, and policy answers."""
        return await search_knowledge_base(query, category=category, limit=limit, pool=pool)

    async def customer(
        customer_id: str | None = None,
        email: str | None = None,
        name: str | None = None,
    ):
        """Look up account-specific plan, status, usage, and integration details."""
        return await lookup_customer(
            customer_id=customer_id,
            email=email,
            name=name,
            pool=pool,
        )

    async def create(
        subject: str,
        description: str,
        category: str,
        priority: str = "normal",
    ):
        """Create a ticket only after the customer confirms they want one."""
        return await create_ticket(
            subject,
            description,
            category=category,
            priority=priority,
            session_id=session_id,
            customer_id=customer_id,
            pool=pool,
        )

    async def status(ticket_id: str):
        """Check the current status of an existing support ticket."""
        return await check_ticket_status(ticket_id, pool=pool)

    async def escalate(
        reason: str,
        conversation_summary: str,
        urgency: str = "normal",
    ):
        """Escalate when the customer requests a human or the issue needs authority."""
        return await escalate_to_human(
            reason,
            conversation_summary,
            urgency=urgency,
            session_id=session_id,
            customer_id=customer_id,
            pool=pool,
        )

    definitions = (
        (SEARCH_KNOWLEDGE_BASE, search),
        (LOOKUP_CUSTOMER, customer),
        (CREATE_TICKET, create),
        (CHECK_TICKET_STATUS, status),
        (ESCALATE_TO_HUMAN, escalate),
    )
    return [
        StructuredTool.from_function(
            coroutine=coroutine,
            name=name,
            description=(coroutine.__doc__ or name).strip(),
        )
        for name, coroutine in definitions
    ]


def tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """Index for dispatch in the tools node."""
    return {tool.name: tool for tool in tools}
