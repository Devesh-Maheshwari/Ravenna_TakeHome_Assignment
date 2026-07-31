"""Wire format for the API.

This module owns one translation that matters: the problem statement's response
examples use the roles **`customer`** and **`agent`**, while LangGraph uses
`human` and `ai`. Converting here — and nowhere else — means the published
contract is ours, not a framework's, and a LangGraph upgrade that renames its
message types cannot change what our customers' clients receive.

The reasoning trace is part of the response, not an afterthought, because the
problem statement asks for tool usage and reasoning to be inspectable. Tool
arguments are included; raw tool payloads are summarised, because three
knowledge-base articles inlined verbatim would bury the answer.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Role(StrEnum):
    """The API's vocabulary, per the problem statement — not LangGraph's."""

    CUSTOMER = "customer"
    AGENT = "agent"
    SYSTEM = "system"


# --- Requests ---------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Body for `POST /sessions`. Everything optional — a session needs no input."""

    metadata: dict[str, Any] = Field(default_factory=dict)


class SendMessageRequest(BaseModel):
    """Body for `POST /sessions/{id}/messages`."""

    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def message_must_have_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain non-whitespace characters")
        return value.strip()


# --- Responses --------------------------------------------------------------


class CreateSessionResponse(BaseModel):
    session_id: UUID
    created_at: datetime


class TraceStepModel(BaseModel):
    """One step in the turn. Mirrors `agent.trace.TraceStep`."""

    kind: str  # guardrail | model_call | tool_call | escalation | finalize
    name: str
    duration_ms: int
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ReasoningTrace(BaseModel):
    """How the agent arrived at this answer.

    `iterations` is the llm<->tools round-trip count; a high value on a simple
    question is the visible symptom of a prompt or tool-description problem.
    """

    steps: list[TraceStepModel]
    iterations: int
    outcome: str  # resolved | escalated | refused | error


class EscalationInfo(BaseModel):
    escalated: bool
    reason: str | None = None
    ticket_id: str | None = None


class UsageInfo(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ToolCallInfo(BaseModel):
    """Compact, assignment-shaped description of one executed tool call."""

    tool: str
    query: str | None = None
    result_summary: str


class MessageResponse(BaseModel):
    """Response to `POST /sessions/{id}/messages`."""

    session_id: UUID
    message_id: str
    response: str
    # Stable names retained for transcript/UI compatibility.
    tools_used: list[str]
    # Structured call details matching the POST example in the assignment.
    tool_calls: list[ToolCallInfo]
    reasoning: ReasoningTrace
    escalation: EscalationInfo
    # Correlates this reply with its log lines and its persisted trace.
    trace_id: UUID
    latency_ms: int
    usage: UsageInfo


class MessageRecord(BaseModel):
    """One transcript entry in `GET /sessions/{id}`.

    `tools_used` is present on agent turns and empty on customer turns, matching
    the shape in the problem statement's example.
    """

    role: Role
    content: str
    timestamp: datetime
    tools_used: list[str] = Field(default_factory=list)


class SessionDetailResponse(BaseModel):
    """Response to `GET /sessions/{id}`."""

    session_id: UUID
    created_at: datetime
    status: str  # active | expired | escalated | closed
    customer_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[MessageRecord]


class SessionSummaryResponse(BaseModel):
    session_id: UUID
    created_at: datetime
    updated_at: datetime
    status: str
    customer_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    sessions: list[SessionSummaryResponse]


class DemoCustomerResponse(BaseModel):
    """Non-sensitive identity fields exposed by the local demo customer picker."""

    customer_id: str
    name: str


class DemoCustomerListResponse(BaseModel):
    customers: list[DemoCustomerResponse]


class TraceResponse(BaseModel):
    """Response to `GET /sessions/{id}/trace` — every turn, for debugging."""

    session_id: UUID
    turns: list[ReasoningTrace]


class ErrorResponse(BaseModel):
    """Uniform error body. `code` is stable and machine-readable; `message` is not."""

    code: str
    message: str
    request_id: str | None = None
