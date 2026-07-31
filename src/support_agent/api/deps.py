"""Dependency injection for route handlers.

Process-wide resources are built once in the lifespan and stashed on
`app.state`. These providers read them back out. Handlers therefore never touch
module-level globals, which lets tests construct isolated application instances.
"""

from typing import Annotated

from fastapi import Depends, Request
from psycopg_pool import AsyncConnectionPool

from support_agent.config import Settings


def get_pool(request: Request) -> AsyncConnectionPool:
    """The shared psycopg pool."""
    return request.app.state.pool


def get_config(request: Request) -> Settings:
    """Application settings."""
    return request.app.state.settings


PoolDep = Annotated[AsyncConnectionPool, Depends(get_pool)]
SettingsDep = Annotated[Settings, Depends(get_config)]
