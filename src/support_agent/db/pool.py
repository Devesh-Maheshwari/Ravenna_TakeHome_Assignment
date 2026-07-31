"""psycopg3 connection pool lifecycle.

One pool for the whole live process. The optional LangGraph reference worker can
also use this pool through its Postgres checkpointer.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


async def open_pool(database_url: str) -> AsyncConnectionPool:
    """Create and open the pool.

    Configured with `row_factory=dict_row` so repositories get dicts rather than
    positional tuples. `autocommit=True` keeps each small repository operation
    atomic and is also required by the optional checkpointer.

    Opening does not block on Postgres. This lets the API start and report a
    degraded `/health` response while Docker is still coming up, rather than
    crashing the uvicorn worker during its lifespan.
    """
    pool = AsyncConnectionPool(
        conninfo=database_url,
        kwargs={"autocommit": True, "row_factory": dict_row},
        min_size=1,
        max_size=10,
        timeout=2.0,
        open=False,
    )
    await pool.open(wait=False)
    return pool


async def close_pool(pool: AsyncConnectionPool) -> None:
    """Close the pool and wait for in-flight connections to be returned."""
    await pool.close()


@asynccontextmanager
async def connection(pool: AsyncConnectionPool) -> AsyncIterator[AsyncConnection]:
    """Check a connection out of the pool for the duration of the block."""
    async with pool.connection() as conn:
        yield conn


async def ping(pool: AsyncConnectionPool) -> bool:
    """`SELECT 1`. Backs the readiness half of `GET /health`."""
    try:
        async with pool.connection(timeout=1.0) as conn:
            await conn.execute("SELECT 1")
    except Exception:
        return False
    return True
