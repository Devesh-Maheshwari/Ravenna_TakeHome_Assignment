"""Load the provided dataset and index the knowledge base.

Reads `data/seed/*.json` — extracted verbatim from the problem statement — and
upserts 25 knowledge-base articles and 15 customer accounts. Upsert rather than
insert so re-seeding is safe and does not require a database reset.

Remote embedding is opt-in rather than the default because support articles may
contain organization data. The default local retrieval uses PostgreSQL full-text
and trigram indexes and sends no corpus content to an external service.

Degrades honestly: if the embeddings call fails, the articles still load with a
NULL `embedding` and search falls back to keyword-only. A missing semantic index
should not mean an empty knowledge base.

Run with `make seed`.
"""

import asyncio
import json
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from support_agent.agent.llm import build_embeddings
from support_agent.config import get_settings
from support_agent.db.pool import close_pool, open_pool
from support_agent.db.repositories import customers, knowledge_base

ROOT = Path(__file__).resolve().parents[1]


async def seed_customers(pool: AsyncConnectionPool | None = None) -> int:
    """Upsert `data/seed/customers.json`. Returns the row count.

    The JSON keeps the problem statement's field names; `created_at` there maps
    to `signed_up_on` in our schema, which avoids colliding with the row-creation
    timestamps used everywhere else in the database.
    """
    owns_pool = pool is None
    if pool is None:
        pool = await open_pool(get_settings().database_url)
        await pool.wait()
    records = json.loads((ROOT / "data/seed/customers.json").read_text())
    try:
        for record in records:
            await customers.upsert(pool, record)
    finally:
        if owns_pool:
            await close_pool(pool)
    return len(records)


async def seed_knowledge_base(
    *,
    embed: bool = False,
    pool: AsyncConnectionPool | None = None,
) -> int:
    """Upsert `data/seed/knowledge_base_tickets.json`, embedding each article."""
    settings = get_settings()
    owns_pool = pool is None
    if pool is None:
        pool = await open_pool(settings.database_url)
        await pool.wait()
    records = json.loads((ROOT / "data/seed/knowledge_base_tickets.json").read_text())
    vectors: list[list[float] | None] = [None] * len(records)
    if embed:
        documents = [
            "\n".join(
                (
                    record["customer_question"],
                    record["support_agent_response"],
                    " ".join(record.get("tags", [])),
                )
            )
            for record in records
        ]
        try:
            embedded = await build_embeddings(settings).aembed_documents(documents)
            vectors = list(embedded)
        except Exception as exc:
            print(f"Embedding unavailable ({type(exc).__name__}); seeding keyword data only.")
    try:
        for record, vector in zip(records, vectors, strict=True):
            await knowledge_base.upsert_article(pool, record, vector)
    finally:
        if owns_pool:
            await close_pool(pool)
    return len(records)


async def main() -> int:
    """Seed both tables and report what landed."""
    pool = await open_pool(get_settings().database_url)
    await pool.wait()
    try:
        customer_count = await seed_customers(pool)
        article_count = await seed_knowledge_base(pool=pool)
    finally:
        await close_pool(pool)
    print(f"Seeded {customer_count} customers and {article_count} knowledge-base articles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
