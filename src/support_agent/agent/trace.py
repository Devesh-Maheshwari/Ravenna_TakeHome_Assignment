"""Reasoning-trace collection.

The problem statement requires the agent's tool usage and reasoning to be
inspectable in the API response. This module builds that record: one step per
node visit and per tool call, in order, with arguments, outcome, and duration.

Design note — what a step records and what it does not. Tool *arguments* are
captured because "why did it search for that?" is the most common debugging
question. Tool *results* are summarised rather than stored whole, because a
knowledge-base hit is several hundred words and three of them would make the
response unreadable. The full result is reachable via the tool's own data if
someone needs it.

Model reasoning is represented by the decisions it made — which tool, with which
arguments, in which order — not by chain-of-thought text. That is both what the
API can honestly expose and what is actually useful when debugging.
"""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class StepKind(StrEnum):
    GUARDRAIL = "guardrail"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    ESCALATION = "escalation"
    FINALIZE = "finalize"


class Outcome(StrEnum):
    """How a turn ended. Mirrors the `agent_traces.outcome` CHECK constraint and
    the `support_agent_turns_total` metric label."""

    RESOLVED = "resolved"
    ESCALATED = "escalated"
    REFUSED = "refused"
    ERROR = "error"


@dataclass(slots=True)
class TraceStep:
    kind: StepKind
    name: str  # node name, or tool name for TOOL_CALL
    duration_ms: int
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class TraceCollector:
    """Accumulates steps across one turn.

    Created per turn and threaded through the live orchestrator or reference graph.
    """

    trace_id: str
    steps: list[TraceStep] = field(default_factory=list)
    iterations: int = 0

    def record(
        self,
        kind: StepKind,
        name: str,
        *,
        duration_ms: int,
        detail: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Append a step."""
        self.steps.append(
            TraceStep(
                kind=kind,
                name=name,
                duration_ms=duration_ms,
                detail=detail or {},
                error=error,
            )
        )

    def tools_used(self) -> list[str]:
        """Distinct tool names in call order — the `tools_used` field in the API
        response and on the message row."""
        return list(
            dict.fromkeys(step.name for step in self.steps if step.kind == StepKind.TOOL_CALL)
        )

    def to_json(self) -> list[dict[str, Any]]:
        """Serialise for the `agent_traces.steps` JSONB column."""
        return [{**asdict(step), "kind": step.kind.value} for step in self.steps]


def summarise_tool_result(tool_name: str, result: Any, *, max_chars: int = 280) -> str:
    """Compress a tool result into a line fit for the trace.

    Per-tool: a knowledge-base hit becomes its article ids and scores, a customer
    lookup becomes the id and plan (never the full record — see
    `security/guardrails.py` on redaction), a ticket becomes its id.
    """
    if isinstance(result, dict):
        preferred = {
            key: result[key]
            for key in ("ticket_id", "customer_id", "plan", "status", "score")
            if key in result
        }
        text = str(preferred or result)
    else:
        text = str(result)
    text = " ".join(text.split())
    return text if len(text) <= max_chars else f"{text[: max_chars - 1]}…"
