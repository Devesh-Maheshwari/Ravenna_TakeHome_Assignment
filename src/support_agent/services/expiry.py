"""Background session sweeper.

The lazy expiry check in `services/sessions.py` handles sessions someone touches
again. This handles the ones nobody ever comes back to — the common case, since
an abandoned conversation is abandoned precisely because no further request
arrives to trigger a lazy check.

Runs as an asyncio task on the app lifespan rather than as a cron job or an
external scheduler: it is one query on an indexed column every few minutes, and a
separate process to run it would be more moving parts than the work justifies.

Expiry is a status transition, not a delete. The transcript stays for support
review and for the evaluation suite; only the session's usability ends. Purging
old rows is a retention policy, which is a different decision — noted in
`docs/DESIGN_DECISIONS.md`.
"""

import asyncio
from datetime import UTC, datetime, timedelta

from psycopg_pool import AsyncConnectionPool

from support_agent.config import Settings
from support_agent.db.repositories import sessions
from support_agent.observability.logging import get_logger

logger = get_logger(__name__)


async def sweep_once(pool: AsyncConnectionPool, settings: Settings) -> int:
    """Expire every session past either TTL. Returns how many were expired.

    Separated from the loop so tests can drive one pass deterministically instead
    of waiting on a timer.
    """
    now = datetime.now(UTC)
    idle_cutoff = now - timedelta(minutes=settings.session_idle_ttl_minutes)
    return await sessions.expire_stale(pool, idle_cutoff=idle_cutoff, now=now)


async def run_sweeper(pool: AsyncConnectionPool, settings: Settings) -> None:
    """Sweep forever on the configured interval.

    Must survive its own failures: a transient database error should log and wait
    for the next tick, not kill the task and silently stop all expiry for the
    lifetime of the process. Exits cleanly on `CancelledError` at shutdown.
    """
    while True:
        try:
            await sweep_once(pool, settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("session_sweep_failed", error_type=type(exc).__name__)
        await asyncio.sleep(settings.session_sweep_interval_seconds)


def start_sweeper(pool: AsyncConnectionPool, settings: Settings) -> asyncio.Task:
    """Launch the sweeper. The caller holds the task and cancels it on shutdown."""
    return asyncio.create_task(run_sweeper(pool, settings), name="session-expiry-sweeper")
