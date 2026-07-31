"""FastAPI application factory and process lifespan.

The runnable shell owns process-wide configuration, metrics, logging, the
psycopg connection pool, and the idle-session sweeper.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from prometheus_client import CollectorRegistry

from support_agent.api.errors import register_exception_handlers
from support_agent.api.routes import router
from support_agent.config import get_settings
from support_agent.db.pool import close_pool, open_pool
from support_agent.observability.context import request_id_var
from support_agent.observability.logging import configure_logging
from support_agent.observability.metrics import build_metrics
from support_agent.services.expiry import start_sweeper


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open and close process-wide runtime resources.

    The live turn orchestrator uses the shared pool directly. The separately
    tested LangGraph builder can be used by alternate workers with a checkpoint
    saver bound to the same database.
    """
    app.state.pool = await open_pool(app.state.settings.database_url)
    app.state.sweeper_task = start_sweeper(app.state.pool, app.state.settings)
    try:
        yield
    finally:
        app.state.sweeper_task.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.sweeper_task
        await close_pool(app.state.pool)


def create_app() -> FastAPI:
    """Build the application: logging, middleware, routes, exception handlers."""
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="Agentic Customer Support Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics_registry = CollectorRegistry()
    app.state.metrics = build_metrics(app.state.metrics_registry)

    @app.middleware("http")
    async def request_context(request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            request_id_var.reset(token)

    app.include_router(router)
    register_exception_handlers(app)
    return app


# No module-level `app = create_app()`. Uvicorn is run with `--factory` (see the
# Makefile) so the application is constructed on demand rather than as an
# import-time side effect. That keeps `import support_agent.main` free of side
# effects, which is what lets tests build an app with overridden dependencies
# instead of working around a module-level singleton.
