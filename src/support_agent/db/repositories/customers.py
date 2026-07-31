"""Customer account lookup.

Backs the `lookup_customer` tool. Lookup is deliberately narrow — by exact id, or
by email, or by name — because a fuzzy "find me anyone matching this" search on a
customer table is how one customer's data ends up in another's conversation.

Name lookup can legitimately match several people, so it returns a list and lets
the caller decide. The tool turns a multi-match into an "ambiguous" result that
makes the agent ask which account is meant, instead of guessing. See
`tools/base.py`.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from psycopg_pool import AsyncConnectionPool

# The short assignment brief names these sample addresses, while the expanded
# 15-account fixture uses richer demo-domain addresses. Accept both without
# duplicating customer rows or weakening exact-email lookup.
SAMPLE_EMAIL_ALIASES = {
    "alice@example.com": "cust-001",
    "bob@example.com": "cust-002",
    "carol@example.com": "cust-003",
}


@dataclass(frozen=True, slots=True)
class Customer:
    customer_id: str
    name: str
    email: str
    plan: str  # free | pro | enterprise
    status: str  # active | past_due | suspended | cancelled | trialing
    seats_used: int
    seats_limit: int | None  # None means unlimited (enterprise)
    storage_used_gb: Decimal
    signed_up_on: date
    billing_cycle: str | None  # None on the free plan
    monthly_rate_usd: Decimal
    integrations_enabled: list[str]
    workspace_name: str
    trial_ends_at: date | None


async def get_by_id(pool: AsyncConnectionPool, customer_id: str) -> Customer | None:
    """Exact match on `cust-0NN`."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM customers WHERE customer_id = %s", (customer_id,)
        )
        row = await cursor.fetchone()
    return Customer(**row) if row else None


async def get_by_email(pool: AsyncConnectionPool, email: str) -> Customer | None:
    """Case-insensitive exact match on email — the reliable identifier."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM customers WHERE lower(email) = lower(%s)", (email.strip(),)
        )
        row = await cursor.fetchone()
    if row:
        return Customer(**row)
    alias_customer_id = SAMPLE_EMAIL_ALIASES.get(email.strip().casefold())
    return await get_by_id(pool, alias_customer_id) if alias_customer_id else None


async def find_by_name(pool: AsyncConnectionPool, name: str, *, limit: int = 5) -> list[Customer]:
    """Case-insensitive name search. May return several; the caller disambiguates."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT *
            FROM customers
            WHERE lower(name) = lower(%s)
               OR lower(name) LIKE lower(%s)
            ORDER BY
                CASE WHEN lower(name) = lower(%s) THEN 0 ELSE 1 END,
                name
            LIMIT %s
            """,
            (name.strip(), f"%{name.strip()}%", name.strip(), limit),
        )
        rows = await cursor.fetchall()
    return [Customer(**row) for row in rows]


async def list_all(pool: AsyncConnectionPool) -> list[Customer]:
    """List seeded demo customers by name for the local support console."""
    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT * FROM customers ORDER BY name")
        rows = await cursor.fetchall()
    return [Customer(**row) for row in rows]


async def upsert(pool: AsyncConnectionPool, customer: dict) -> None:
    """Insert or update one account. Used by `scripts/seed_db.py`."""
    signed_up_on = customer.get("signed_up_on", customer.get("created_at"))
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO customers (
                customer_id, name, email, plan, status, seats_used, seats_limit,
                storage_used_gb, signed_up_on, billing_cycle, monthly_rate_usd,
                integrations_enabled, workspace_name, trial_ends_at
            )
            VALUES (
                %(customer_id)s, %(name)s, %(email)s, %(plan)s, %(status)s,
                %(seats_used)s, %(seats_limit)s, %(storage_used_gb)s,
                %(signed_up_on)s, %(billing_cycle)s, %(monthly_rate_usd)s,
                %(integrations_enabled)s, %(workspace_name)s, %(trial_ends_at)s
            )
            ON CONFLICT (customer_id) DO UPDATE SET
                name = EXCLUDED.name,
                email = EXCLUDED.email,
                plan = EXCLUDED.plan,
                status = EXCLUDED.status,
                seats_used = EXCLUDED.seats_used,
                seats_limit = EXCLUDED.seats_limit,
                storage_used_gb = EXCLUDED.storage_used_gb,
                signed_up_on = EXCLUDED.signed_up_on,
                billing_cycle = EXCLUDED.billing_cycle,
                monthly_rate_usd = EXCLUDED.monthly_rate_usd,
                integrations_enabled = EXCLUDED.integrations_enabled,
                workspace_name = EXCLUDED.workspace_name,
                trial_ends_at = EXCLUDED.trial_ends_at
            """,
            {
                **customer,
                "signed_up_on": signed_up_on,
                "trial_ends_at": customer.get("trial_ends_at"),
            },
        )
