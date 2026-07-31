"""HTTP endpoints.

One module because there are six handlers and they all concern one resource;
splitting them across a `routes/` package would add a directory and an import
graph without making anything easier to find.

Handlers stay thin on purpose — validate, delegate, map to the response model.
The turn logic lives in `services/conversation.py` so it can be driven from a
test or a script without an HTTP client.
"""

from uuid import UUID

from fastapi import APIRouter, Request, Response

from support_agent.api.deps import PoolDep, SettingsDep
from support_agent.api.errors import SessionNotFound
from support_agent.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    DemoCustomerListResponse,
    DemoCustomerResponse,
    EscalationInfo,
    MessageRecord,
    MessageResponse,
    ReasoningTrace,
    SendMessageRequest,
    SessionDetailResponse,
    SessionListResponse,
    SessionSummaryResponse,
    ToolCallInfo,
    TraceResponse,
    UsageInfo,
)
from support_agent.db.pool import ping
from support_agent.db.repositories import customers, messages, traces
from support_agent.db.repositories import sessions as session_repository
from support_agent.observability.metrics import render
from support_agent.services import conversation, sessions
from support_agent.services.expiry import sweep_once

router = APIRouter()


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    body: CreateSessionRequest,
    pool: PoolDep,
    settings: SettingsDep,
) -> CreateSessionResponse:
    """Open a conversation session.

    Returns immediately; no agent work happens until the first message arrives.
    """
    session = await sessions.create_session(pool, settings, metadata=body.metadata)
    return CreateSessionResponse(session_id=session.id, created_at=session.created_at)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    pool: PoolDep,
    settings: SettingsDep,
    client_id: str | None = None,
    customer_id: str | None = None,
    status: str | None = None,
    source: str | None = None,
    limit: int = 50,
) -> SessionListResponse:
    """List recent conversations scoped to a browser or selected customer."""
    await sweep_once(pool, settings)
    records = await session_repository.list_recent(
        pool,
        client_id=client_id,
        customer_id=customer_id,
        status=status,
        source=source,
        limit=min(max(limit, 1), 100),
    )
    return SessionListResponse(
        sessions=[
            SessionSummaryResponse(
                session_id=record.id,
                created_at=record.created_at,
                updated_at=record.updated_at,
                status=record.status,
                customer_id=record.customer_id,
                metadata=record.metadata,
            )
            for record in records
        ]
    )


@router.get("/demo/customers", response_model=DemoCustomerListResponse)
async def list_demo_customers(pool: PoolDep) -> DemoCustomerListResponse:
    """Names and ids for the local seeded-account picker.

    This deliberately excludes email, billing, usage, and other account data.
    A production customer-facing UI must derive identity from authentication
    rather than expose an account picker.
    """
    records = await customers.list_all(pool)
    return DemoCustomerListResponse(
        customers=[
            DemoCustomerResponse(customer_id=record.customer_id, name=record.name)
            for record in records
        ]
    )


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(
    session_id: UUID,
    body: SendMessageRequest,
    request: Request,
    pool: PoolDep,
    settings: SettingsDep,
) -> MessageResponse:
    """Send a customer message and get the agent's reply.

    The one endpoint that runs the agent. Returns the reply together with the
    tools used and the reasoning trace, satisfying the requirement that tool
    usage and reasoning be inspectable in the API response.

    409 on an expired session — the customer starts a new one rather than
    resuming a conversation the agent no longer remembers.
    """
    result = await conversation.handle_message(
        session_id=session_id,
        message=body.message,
        pool=pool,
        settings=settings,
    )
    request.app.state.metrics["turns_total"].labels(outcome=result.outcome).inc()
    request.app.state.metrics["turn_latency_seconds"].observe(result.latency_ms / 1000)
    request.app.state.metrics["agent_iterations"].observe(result.iterations)
    if result.input_tokens:
        request.app.state.metrics["llm_tokens_total"].labels(direction="input").inc(
            result.input_tokens
        )
    if result.output_tokens:
        request.app.state.metrics["llm_tokens_total"].labels(direction="output").inc(
            result.output_tokens
        )
    for step in result.trace_steps:
        detail = step.get("detail") or {}
        if step.get("kind") == "guardrail":
            for reason in detail.get("flags", []):
                request.app.state.metrics["guardrail_blocks_total"].labels(
                    reason=reason
                ).inc()
        if step.get("kind") == "tool_call":
            tool = str(step.get("name"))
            outcome = str(detail.get("status") or "error")
            request.app.state.metrics["tool_calls_total"].labels(
                tool=tool,
                outcome=outcome,
            ).inc()
            request.app.state.metrics["tool_latency_seconds"].labels(tool=tool).observe(
                max(0, int(step.get("duration_ms") or 0)) / 1000
            )
            if tool == "escalate_to_human" and result.escalated:
                reason = str((detail.get("result") or {}).get("reason") or "unspecified")
                request.app.state.metrics["escalations_total"].labels(reason=reason).inc()
    request.app.state.metrics["active_sessions"].set(
        await session_repository.count_active(pool)
    )
    return MessageResponse(
        session_id=session_id,
        message_id=result.message_id,
        response=result.reply,
        tools_used=result.tools_used,
        tool_calls=[ToolCallInfo(**call) for call in result.tool_calls],
        reasoning=ReasoningTrace(
            steps=result.trace_steps,
            iterations=result.iterations,
            outcome=result.outcome,
        ),
        escalation=EscalationInfo(
            escalated=result.escalated,
            ticket_id=result.ticket_id,
        ),
        trace_id=result.trace_id,
        latency_ms=result.latency_ms,
        usage=UsageInfo(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        ),
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: UUID, pool: PoolDep) -> SessionDetailResponse:
    """Full conversation history and session metadata.

    Reads the durable API-shaped `messages` table. Framework-specific state from
    the optional LangGraph reference worker is never part of this contract.
    """
    session = await session_repository.get(pool, session_id)
    if session is None:
        raise SessionNotFound("Conversation session not found.")
    history = await messages.list_for_session(pool, session_id)
    return SessionDetailResponse(
        session_id=session.id,
        created_at=session.created_at,
        status=session.status,
        customer_id=session.customer_id,
        metadata=session.metadata,
        messages=[
            MessageRecord(
                role=message.role,
                content=message.content,
                timestamp=message.created_at,
                tools_used=message.tools_used,
            )
            for message in history
        ],
    )


@router.post("/sessions/{session_id}/close", response_model=SessionDetailResponse)
async def close_session(session_id: UUID, pool: PoolDep) -> SessionDetailResponse:
    """Explicitly close a conversation while preserving its transcript."""
    session = await session_repository.get(pool, session_id)
    if session is None:
        raise SessionNotFound("Conversation session not found.")
    await sessions.close_session(pool, session_id)
    return await get_session(session_id, pool)


@router.get("/sessions/{session_id}/trace", response_model=TraceResponse)
async def get_trace(session_id: UUID, pool: PoolDep) -> TraceResponse:
    """Every turn's reasoning trace for this session.

    The debugging surface: when a customer reports a bad answer, this is what an
    engineer opens. Beyond the required endpoints — see the observability design
    challenge in docs/DESIGN_DECISIONS.md.
    """
    session = await session_repository.get(pool, session_id)
    if session is None:
        raise SessionNotFound("Conversation session not found.")
    records = await traces.list_for_session(pool, session_id)
    return TraceResponse(
        session_id=session_id,
        turns=[
            ReasoningTrace(
                steps=record.steps,
                iterations=record.iterations,
                outcome=record.outcome,
            )
            for record in records
        ],
    )


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Prometheus exposition format."""
    request.app.state.metrics["active_sessions"].set(
        await session_repository.count_active(request.app.state.pool)
    )
    payload, content_type = render(request.app.state.metrics_registry)
    return Response(content=payload, media_type=content_type)


@router.get("/health")
async def health(request: Request) -> dict:
    """Liveness plus database reachability."""
    database_reachable = await ping(request.app.state.pool)
    return {
        "status": "ok" if database_reachable else "degraded",
        "database": {"reachable": database_reachable},
    }
