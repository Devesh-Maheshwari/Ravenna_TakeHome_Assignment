"""structlog configuration — the connective tissue between traces and metrics.

Every log line carries the `trace_id` and `session_id` from the ambient context
(see `context.py`), so the path from "this metric spiked" to "here are the log
lines" to "here is the full trace for that turn" is a single identifier the whole
way. Without that, three observability layers are three disconnected islands.

JSON in production for machine parsing, pretty console locally for humans,
switched by `LOG_JSON`. Customer message content is redacted before it is logged
(`security/guardrails.redact`) — a support transcript is full of emails and
account details, and logs have a longer reach and a wider audience than the
database does.
"""

import logging
import sys

import structlog

from support_agent.config import Settings
from support_agent.observability.context import current_context


def _merge_context(_logger, _method_name, event_dict):
    return {**current_context(), **event_dict}


def configure_logging(settings: Settings) -> None:
    """Install processors and set the level. Called once, first thing at startup."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=level, force=True)

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            _merge_context,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """A bound logger that picks up ambient trace context automatically."""
    return structlog.get_logger(name)
