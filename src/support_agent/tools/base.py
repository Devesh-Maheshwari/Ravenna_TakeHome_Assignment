"""Shared tool result envelope and error handling.

Every tool returns the same shape. The reason is the ambiguity design challenge:
a tool that returns `None` for "no match" and `None` for "seven matches" forces
the model to guess, and it will guess confidently. Making `AMBIGUOUS` a distinct,
first-class status is what lets the agent ask "which of these two accounts did
you mean?" instead of picking one and being silently wrong.

`NOT_FOUND` and `ERROR` are likewise distinct. "No such customer" is information
the agent should relay; "the database is unreachable" is a reason to escalate.
Collapsing them would make the agent apologise for the customer not existing.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import wraps
from typing import Any

from support_agent.agent.prompts import format_untrusted
from support_agent.observability.logging import get_logger

logger = get_logger(__name__)


class ToolStatus(StrEnum):
    """Also the `outcome` label on `support_agent_tool_calls_total`."""

    SUCCESS = "success"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"  # several candidates — ask, do not guess
    ERROR = "error"  # the tool itself failed


@dataclass(slots=True)
class ToolResult:
    """What every tool returns.

    `message` is written for the model to read: it says what happened and, on a
    non-success status, what to do about it. `data` carries the structured payload.
    `candidates` is populated only on AMBIGUOUS, holding the options to offer.
    """

    status: ToolStatus
    message: str
    data: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def to_model_string(self) -> str:
        """Render for the model's context.

        Third-party text (knowledge-base bodies, customer-supplied fields) is
        wrapped by `prompts.format_untrusted` so it cannot read as instruction.
        """
        payload = {
            "status": self.status.value,
            "message": self.message,
            "data": self.data,
            "candidates": self.candidates,
        }
        return format_untrusted("tool-result", json.dumps(payload, default=str))


def tool_errors_to_result(func: Callable) -> Callable:
    """Decorator converting an unhandled exception into `ToolResult(ERROR, ...)`.

    A tool that raises must never kill the turn. The model needs to see the
    failure to tell the customer about it or route around it, and the orchestrator needs
    the failure counted so deterministic escalation can fire. Logs the exception
    with the turn's `trace_id`, and returns a message safe to show a customer —
    no stack traces, no connection strings.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> ToolResult:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            logger.exception(
                "tool_execution_failed",
                tool=func.__name__,
                error_type=type(exc).__name__,
            )
            return ToolResult(
                status=ToolStatus.ERROR,
                message="The support tool is temporarily unavailable. Do not guess.",
            )

    return wrapper
