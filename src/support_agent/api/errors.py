"""Domain exceptions and their HTTP mapping.

Services raise these; handlers never build an `HTTPException` themselves. That
keeps status-code policy in one table instead of scattered across route bodies,
and lets `services/conversation.py` be driven from a script without importing
FastAPI.

The rule for what reaches a customer: a `SupportAgentError` carries a message
written to be read by one. Anything else becomes a generic 500 with the
`request_id` attached — the detail goes to the logs, where the `trace_id` ties it
back to the full turn. Internal messages leak schema and infrastructure, and
apologising in the language of a stack trace is not support.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from support_agent.api.schemas import ErrorResponse
from support_agent.observability.context import request_id_var
from support_agent.observability.logging import get_logger

logger = get_logger(__name__)


class SupportAgentError(Exception):
    """Base class. `code` is the stable machine-readable value in `ErrorResponse`."""

    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SessionNotFound(SupportAgentError):
    """No session with that id. 404."""

    code = "session_not_found"
    status_code = 404


class SessionExpired(SupportAgentError):
    """Session past its idle or absolute TTL. 409.

    Distinct from 404 because the difference is actionable: the customer should
    open a new session, and saying so is more useful than "not found".
    """

    code = "session_expired"
    status_code = 409


class SessionEscalated(SupportAgentError):
    """Handed to a human; the agent should not keep answering. 409."""

    code = "session_escalated"
    status_code = 409


class AgentExecutionError(SupportAgentError):
    """The agent loop failed in a way no tool-level handler caught. 500."""

    code = "agent_error"
    status_code = 500


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers mapping the above to `ErrorResponse` bodies."""

    @app.exception_handler(SupportAgentError)
    async def support_agent_error_handler(
        _request: Request, exc: SupportAgentError
    ) -> JSONResponse:
        body = ErrorResponse(
            code=exc.code,
            message=exc.message,
            request_id=request_id_var.get(),
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_request_error", error_type=type(exc).__name__)
        body = ErrorResponse(
            code="internal_error",
            message="An unexpected error occurred.",
            request_id=request_id_var.get(),
        )
        return JSONResponse(status_code=500, content=body.model_dump(mode="json"))
