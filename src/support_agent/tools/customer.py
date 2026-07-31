"""`lookup_customer` — account details.

Example trigger from the problem statement: "What plan am I on?"

Identity handling is the sensitive part. Lookup is by exact id or exact email
where possible; name search is offered because customers do not know their
account id, but a name match that hits several people returns AMBIGUOUS rather
than the first row. Returning "the first Bob Smith" would be a data leak wearing
a helpful face.

Once resolved, the customer id is written to session state, so subsequent turns
answer from context instead of looking the same account up repeatedly.

What comes back is scoped to what a support conversation needs — plan, status,
seats, storage, integrations. Not everything the row contains.
"""

from psycopg_pool import AsyncConnectionPool

from support_agent.db.repositories import customers
from support_agent.tools.base import ToolResult, ToolStatus, tool_errors_to_result


@tool_errors_to_result
async def lookup_customer(
    *,
    customer_id: str | None = None,
    email: str | None = None,
    name: str | None = None,
    pool: AsyncConnectionPool,
) -> ToolResult:
    """Look up a customer's account details.

    This docstring becomes the tool description the model sees. It must
    convey: use it when the answer depends on this specific account (plan,
    subscription status, seat usage, which integrations are enabled); prefer
    email or customer id over name; do not call it again for a fact already
    established earlier in the conversation.

    Exactly one identifier should be supplied. AMBIGUOUS when a name matches
    several accounts, with the candidates listed so the agent can ask.
    """
    supplied = [customer_id is not None, email is not None, name is not None]
    if sum(supplied) != 1:
        return ToolResult(
            ToolStatus.ERROR,
            "Supply exactly one of customer_id, email, or name.",
        )

    if customer_id:
        customer = await customers.get_by_id(pool, customer_id)
        matches = [customer] if customer else []
    elif email:
        customer = await customers.get_by_email(pool, email)
        matches = [customer] if customer else []
    else:
        matches = await customers.find_by_name(pool, name or "")

    if not matches:
        return ToolResult(
            ToolStatus.NOT_FOUND,
            "No customer account matched that identifier.",
        )
    if len(matches) > 1:
        return ToolResult(
            ToolStatus.AMBIGUOUS,
            "Several accounts match that name. Ask the customer which one they mean.",
            candidates=[
                {
                    "customer_id": candidate.customer_id,
                    "name": candidate.name,
                    "email": candidate.email,
                    "workspace_name": candidate.workspace_name,
                }
                for candidate in matches
            ],
        )

    customer = matches[0]
    data = {
        "customer_id": customer.customer_id,
        "name": customer.name,
        "email": customer.email,
        "plan": customer.plan,
        "status": customer.status,
        "seats_used": customer.seats_used,
        "seats_limit": customer.seats_limit,
        "storage_used_gb": str(customer.storage_used_gb),
        "billing_cycle": customer.billing_cycle,
        "integrations_enabled": customer.integrations_enabled,
        "workspace_name": customer.workspace_name,
        "trial_ends_at": str(customer.trial_ends_at) if customer.trial_ends_at else None,
    }
    return ToolResult(
        ToolStatus.SUCCESS,
        f"Found the {customer.workspace_name} account on the {customer.plan} plan.",
        data=data,
    )
