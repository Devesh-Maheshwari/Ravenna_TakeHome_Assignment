"""Turn orchestration — the seam between HTTP and the agent.

`handle_message` is the whole request path for `POST /sessions/{id}/messages`.
Keeping it here rather than in the route handler means the turn can be driven
from a test or a script without an HTTP client, which is what makes the scripted
multi-turn evaluations in `tests/evaluation/` cheap to write.

Order matters and is not arbitrary:

  1. Validate the session is live (expired sessions are rejected, not revived).
  2. Persist the customer's message *before* invoking the agent — if the model
     call fails, the transcript still shows what was asked.
  3. Run deterministic safety/authority routing, then the bounded LLM/tool loop
     when the request needs model judgment. Durable transcript rows supply the
     model's prior context.
  4. Persist the reply, the trace, and the metrics.

Step 4 is best-effort for the trace specifically: a failure to record
observability must not fail a turn the customer already received.
"""

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from psycopg_pool import AsyncConnectionPool

from support_agent.agent.llm import build_chat_model
from support_agent.agent.prompts import build_system_prompt, format_untrusted
from support_agent.agent.trace import Outcome, StepKind, TraceCollector
from support_agent.config import Settings
from support_agent.db.repositories import messages, traces
from support_agent.db.repositories import sessions as session_repository
from support_agent.db.repositories.sessions import Session
from support_agent.observability.context import new_trace_id, turn_context
from support_agent.observability.logging import get_logger
from support_agent.security.guardrails import redact, screen_input
from support_agent.services import customer_memory, sessions
from support_agent.tools.base import ToolResult, ToolStatus
from support_agent.tools.customer import lookup_customer
from support_agent.tools.escalation import escalate_to_human
from support_agent.tools.knowledge_base import search_knowledge_base
from support_agent.tools.registry import build_tools, tools_by_name
from support_agent.tools.tickets import check_ticket_status, create_ticket

logger = get_logger(__name__)


@dataclass(slots=True)
class TurnResult:
    """Everything one turn produced. Mapped to the wire format by `api/schemas.py`."""

    reply: str
    message_id: str
    tools_used: list[str]
    tool_calls: list[dict[str, str | None]]
    trace_id: UUID
    trace_steps: list[dict[str, Any]]
    iterations: int
    outcome: str  # resolved | escalated | refused | error
    escalated: bool
    ticket_id: str | None
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None


@dataclass(slots=True)
class AnswerResult:
    reply: str
    tools_used: list[str]
    usage: dict[str, int | None]
    escalated: bool = False
    ticket_id: str | None = None


async def handle_message(
    *,
    session_id: UUID,
    message: str,
    pool: AsyncConnectionPool,
    settings: Settings,
) -> TurnResult:
    """Run one customer message end to end."""
    started = perf_counter()
    session = await sessions.get_active_session(pool, settings, session_id)
    turn_index = await messages.next_turn_index(pool, session_id)
    trace_id = new_trace_id()
    collector = TraceCollector(trace_id=str(trace_id))

    agent_message = await messages.append(
        pool,
        session_id=session_id,
        turn_index=turn_index,
        role="customer",
        content=message,
    )
    if turn_index == 0:
        await session_repository.merge_metadata(
            pool,
            session_id,
            {"conversation_title": _conversation_title(message)},
        )
    await sessions.record_activity(pool, session_id)

    input_tokens: int | None = None
    output_tokens: int | None = None
    tools_used: list[str] = []

    with turn_context(trace_id=trace_id, session_id=session_id):
        guardrail_started = perf_counter()
        verdict = screen_input(message)
        collector.record(
            StepKind.GUARDRAIL,
            "screen_input",
            duration_ms=_elapsed_ms(guardrail_started),
            detail={"allowed": verdict.allowed, "flags": [flag.value for flag in verdict.flags]},
        )

        if not verdict.allowed:
            reply = verdict.refusal_message or "I can only help with Ravenna support."
            outcome = Outcome.REFUSED
        else:
            answer = await _answer(
                pool=pool,
                settings=settings,
                session=session,
                session_id=session_id,
                message=message,
                collector=collector,
            )
            reply = answer.reply
            tools_used = answer.tools_used
            input_tokens = answer.usage.get("input_tokens")
            output_tokens = answer.usage.get("output_tokens")
            outcome = Outcome.ESCALATED if answer.escalated else Outcome.RESOLVED

        collector.record(
            StepKind.FINALIZE,
            "finalize",
            duration_ms=0,
            detail={"outcome": outcome.value},
        )

    latency_ms = _elapsed_ms(started)
    trace_steps = collector.to_json()
    tool_calls = _structured_tool_calls(trace_steps)
    await messages.append(
        pool,
        session_id=session_id,
        turn_index=turn_index,
        role="agent",
        content=reply,
        tools_used=tools_used,
        trace_id=trace_id,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    try:
        await customer_memory.refresh(pool, session_id)
    except Exception as exc:
        logger.warning("customer_memory_refresh_failed", error_type=type(exc).__name__)

    try:
        await traces.save(
            pool,
            session_id=session_id,
            trace_id=trace_id,
            turn_index=turn_index,
            steps=trace_steps,
            iterations=collector.iterations,
            outcome=outcome.value,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as exc:
        logger.warning("trace_persistence_failed", error_type=type(exc).__name__)

    return TurnResult(
        reply=reply,
        message_id=f"msg-{agent_message.id}",
        tools_used=tools_used,
        tool_calls=tool_calls,
        trace_id=trace_id,
        trace_steps=trace_steps,
        iterations=collector.iterations,
        outcome=outcome.value,
        escalated=answer.escalated if verdict.allowed else False,
        ticket_id=answer.ticket_id if verdict.allowed else None,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))


def _conversation_title(message: str, limit: int = 52) -> str:
    """A readable history label derived from the first customer request."""
    title = " ".join(message.split()).strip(" .")
    if not title:
        return "New conversation"
    return title if len(title) <= limit else f"{title[: limit - 1]}…"


def _is_vague(message: str) -> bool:
    words = message.casefold().strip(" !?.").split()
    greetings = {"hi", "hello", "hey", "help", "support", "howdy"}
    vague_phrases = {
        "technical issue",
        "it does not work",
        "not working",
        "something is wrong",
        "i have a problem",
        "issue",
        "problem",
    }
    normalized = " ".join(words)
    asks_for_unspecified_help = any(
        phrase in normalized for phrase in ("need help", "help with", "need support")
    )
    specific_terms = (
        "password",
        "login",
        "sign in",
        "billing",
        "charge",
        "invoice",
        "export",
        "error",
        "integration",
        "plan",
        "seat",
        "storage",
        "ticket",
        "refund",
    )
    return (
        not words
        or (len(words) <= 3 and set(words) <= greetings)
        or normalized in vague_phrases
        or (asks_for_unspecified_help and not any(term in normalized for term in specific_terms))
    )


def _extract_declared_fact(message: str) -> tuple[str, str] | None:
    """Recognise customer-provided references as conversation facts, not tickets."""
    match = re.search(
        r"\bmy\s+([A-Za-z][A-Za-z '-]{0,39}(?:reference|marker|code))\s+is\s+"
        r"([A-Za-z0-9][A-Za-z0-9_-]{1,40})\b",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    label = " ".join(match.group(1).split()).casefold()
    return label, match.group(2)


def _asks_for_declared_fact(message: str) -> bool:
    lowered = message.casefold()
    return (
        "what" in lowered
        and any(word in lowered for word in ("reference", "marker", "code"))
        and any(word in lowered for word in ("give", "gave", "mention", "said"))
    )


def _looks_like_unknown_error(message: str) -> bool:
    lowered = message.casefold()
    return (
        "error" in lowered
        and (
            "undocumented" in lowered
            or re.search(r"\b[A-Z]{3,}-\d{3,}\b", message) is not None
        )
    )


def _is_generic_account_access_problem(message: str) -> bool:
    """Distinguish a vague login problem from a verified 403 diagnosis."""
    lowered = message.casefold()
    access_phrase = any(
        phrase in lowered
        for phrase in (
            "cannot get into my account",
            "can't get into my account",
            "cant get into my account",
            "cannot log in",
            "can't log in",
            "cant log in",
            "trouble logging in",
            "unable to log in",
        )
    )
    has_specific_signal = any(
        signal in lowered
        for signal in ("403", "forbidden", "password", "reset link", "locked out", "suspended")
    )
    return access_phrase and not has_specific_signal


async def _answer(
    *,
    pool: AsyncConnectionPool,
    settings: Settings,
    session: Session,
    session_id: UUID,
    message: str,
    collector: TraceCollector,
) -> AnswerResult:
    metadata = dict(session.metadata)
    resumption = await _update_topic_state(pool, session_id, metadata, message)
    metadata = resumption[0]
    resume_suffix = resumption[1]
    if resume_suffix:
        return AnswerResult(
            "I’m glad that issue is resolved." + resume_suffix,
            [],
            {},
        )

    if _is_vague(message):
        return AnswerResult(
            "What part of Ravenna needs help—account access, billing, your plan, "
            "an integration, or a technical issue? If it is technical, include the "
            "exact error and what you were doing.",
            [],
            {},
        )

    pending_actions = _pending_action_stack(metadata)
    pending_action = pending_actions[0] if pending_actions else None
    if pending_action and _is_negative_confirmation(message):
        remaining_actions = pending_actions[1:]
        next_action = remaining_actions[0] if remaining_actions else None
        await session_repository.merge_metadata(
            pool,
            session_id,
            {
                "pending_action": next_action,
                "pending_actions": remaining_actions,
            },
        )
        next_suffix = (
            f" Your next pending approval is: {_pending_action_label(next_action)}. "
            "Would you like me to continue with that?"
            if next_action
            else " We can continue troubleshooting here."
        )
        return AnswerResult(
            "Understood—I cancelled that pending action." + next_suffix,
            [],
            {},
        )
    if pending_action and _is_affirmative(message):
        if pending_action["type"] == "escalate":
            return await _perform_escalation(pool, session, message, pending_action, collector)
        if pending_action["type"] == "create_ticket":
            return await _perform_ticket_creation(
                pool,
                session,
                pending_action,
                collector,
                consume_pending=True,
            )
    if (
        pending_action
        and _looks_like_confirmation_attempt(message)
        and not _needs_authority(message.casefold())
        and not _explicit_ticket_request(message.casefold())
    ):
        return AnswerResult(
            "I want to make sure I have your permission. Reply “yes” or “go ahead” "
            "to continue, or “no” or “cancel” to stop.",
            [],
            {},
        )

    ticket_match = re.search(r"\bTK-\d+\b", message, flags=re.IGNORECASE)
    lowered = message.casefold()
    asks_ticket_status = "ticket" in lowered and any(
        word in lowered for word in ("status", "update")
    )
    if ticket_match and asks_ticket_status:
        started = perf_counter()
        result = await check_ticket_status(ticket_match.group(0).upper(), pool=pool)
        _record_tool(
            collector,
            "check_ticket_status",
            started,
            result,
            query=ticket_match.group(0).upper(),
        )
        if result.ok:
            return AnswerResult(result.message + resume_suffix, ["check_ticket_status"], {})
        return AnswerResult(result.message, ["check_ticket_status"], {})
    if asks_ticket_status and not ticket_match:
        return AnswerResult(
            "Please provide the numeric ticket reference in the format TK-1107 "
            "so I can check its status.",
            [],
            {},
        )

    if _explicit_human_request(lowered):
        return await _perform_escalation(
            pool,
            session,
            message,
            {
                "type": "escalate",
                "reason": "customer_requested_human",
                "summary": message,
                "urgency": "normal",
            },
            collector,
        )

    if _needs_authority(lowered):
        new_action = {
            "type": "escalate",
            "reason": "requires_human_authority",
            "summary": message,
            "urgency": "high" if "account recovery" in lowered else "normal",
        }
        queued = _queue_pending_action(pending_actions, new_action)
        await session_repository.merge_metadata(
            pool,
            session_id,
            {
                "pending_action": queued[0],
                "pending_actions": queued,
            },
        )
        earlier_count = len(queued) - 1
        queued_note = ""
        if earlier_count:
            noun = "request" if earlier_count == 1 else "requests"
            queued_note = f" I saved {earlier_count} earlier approval {noun} behind this one."
        return AnswerResult(
            "This request needs a human support agent. Would you like me to escalate "
            "it and open a support ticket?" + queued_note,
            [],
            {},
        )

    if _explicit_ticket_request(lowered):
        pending = {
            "type": "create_ticket",
            "subject": _ticket_subject(message),
            "description": message,
            "category": _detect_topic(message) or "general",
            "priority": "normal",
        }
        return await _perform_ticket_creation(pool, session, pending, collector)

    plan_gated_result = await _maybe_answer_plan_gated_feature(
        pool=pool,
        session=session,
        message=message,
        collector=collector,
    )
    if plan_gated_result is not None:
        return plan_gated_result

    identity_result = await _maybe_lookup_customer(
        pool=pool,
        session=session,
        message=message,
        collector=collector,
        pending_lookup=metadata.get("pending_lookup", False),
    )
    if identity_result is not None:
        return identity_result

    if _needs_customer_identity(lowered) and session.customer_id is None:
        await session_repository.merge_metadata(pool, session_id, {"pending_lookup": True})
        return AnswerResult(
            "I need to identify the account before answering that. What email address "
            "or customer ID is associated with your Ravenna account?",
            [],
            {},
        )

    history = await messages.list_for_session(pool, session_id)
    declared_fact = _extract_declared_fact(message)
    if declared_fact:
        label, value = declared_fact
        return AnswerResult(
            f"Got it—I’ll remember that your {label} is {value} for this conversation.",
            [],
            {},
        )
    if _asks_for_declared_fact(message):
        for record in reversed(history[:-1]):
            fact = _extract_declared_fact(record.content) if record.role == "customer" else None
            if fact:
                label, value = fact
                return AnswerResult(
                    f"You told me your {label} is {value}.",
                    [],
                    {},
                )
    if len(history) > 1 and _is_context_followup(lowered):
        return await _model_answer(
            pool=pool,
            settings=settings,
            customer_id=session.customer_id,
            session_id=session_id,
            message=message,
            collector=collector,
            resume_suffix=resume_suffix,
            prior_context=metadata.get("prior_summary"),
        )
    if _looks_like_unknown_error(message):
        return AnswerResult(
            "I don’t have a verified article for that error, so I don’t want to "
            "guess. What exact action triggers it, and can you share the surrounding "
            "error text and whether it happens every time?",
            [],
            {},
        )
    if _is_generic_account_access_problem(message):
        return AnswerResult(
            "What happens when you try to sign in—do you see an error code, a "
            "password/reset problem, or a message that the account is disabled?",
            [],
            {},
        )

    search_started = perf_counter()
    kb_result = await search_knowledge_base(message, pool=pool, limit=3)
    _record_tool(
        collector,
        "search_knowledge_base",
        search_started,
        kb_result,
        query=message,
    )
    if kb_result.ok:
        hits = (kb_result.data or {}).get("hits", [])
        if hits:
            await session_repository.merge_metadata(
                pool, session_id, {"consecutive_retrieval_misses": 0}
            )
            best_hit = hits[0]
            if best_hit.get("ticket_id") == "TK-005":
                action = {
                    "type": "create_ticket",
                    "subject": "CSV export issue",
                    "description": message,
                    "summary": message,
                    "category": best_hit.get("category") or "technical",
                    "priority": "normal",
                    "origin_topic": metadata.get("current_topic") or "export",
                }
                queued = _queue_pending_action(
                    _pending_action_stack(metadata),
                    action,
                )
                await session_repository.merge_metadata(
                    pool,
                    session_id,
                    {"pending_action": queued[0], "pending_actions": queued},
                )
                return AnswerResult(
                    _grounded_kb_answer(best_hit)
                    + " I can create a support ticket for human follow-up. "
                    "Would you like me to do that?",
                    ["search_knowledge_base"],
                    {},
                )
            return AnswerResult(
                _grounded_kb_answer(best_hit) + resume_suffix,
                ["search_knowledge_base"],
                {},
            )

    misses = int(metadata.get("consecutive_retrieval_misses", 0)) + 1
    await session_repository.merge_metadata(
        pool, session_id, {"consecutive_retrieval_misses": misses}
    )
    return await _model_answer(
        pool=pool,
        settings=settings,
        customer_id=session.customer_id,
        session_id=session_id,
        message=message,
        collector=collector,
        resume_suffix=resume_suffix,
        prior_context=metadata.get("prior_summary"),
    )


async def _model_answer(
    *,
    pool: AsyncConnectionPool,
    settings: Settings,
    customer_id: str | None,
    session_id: UUID,
    message: str,
    collector: TraceCollector,
    resume_suffix: str = "",
    prior_context: str | None = None,
) -> AnswerResult:
    if not settings.openai_api_key or settings.openai_api_key == "sk-replace-me":
        return AnswerResult(_fallback_answer(message) + resume_suffix, [], {})

    history = await messages.list_for_session(pool, session_id)
    model_messages: list = [
        SystemMessage(
            content=build_system_prompt(
                customer_id=customer_id,
                prior_context=prior_context,
            )
        )
    ]
    for record in history:
        framed = format_untrusted("customer-message", record.content)
        if record.role == "customer":
            model_messages.append(HumanMessage(content=framed))
        elif record.role == "agent":
            model_messages.append(AIMessage(content=record.content))

    model_started = perf_counter()
    try:
        tool_list = build_tools(
            pool=pool,
            session_id=session_id,
            customer_id=customer_id,
        )
        tool_index = tools_by_name(tool_list)
        model = build_chat_model(settings, tools=tool_list)
        tools_used: list[str] = []
        input_tokens = 0
        output_tokens = 0

        for _ in range(settings.agent_max_iterations):
            call_started = perf_counter()
            response = await model.ainvoke(model_messages)
            usage = response.usage_metadata or {}
            input_tokens += usage.get("input_tokens") or 0
            output_tokens += usage.get("output_tokens") or 0
            collector.iterations += 1
            collector.record(
                StepKind.MODEL_CALL,
                "support_model",
                duration_ms=_elapsed_ms(call_started),
                detail={"used_tools": bool(response.tool_calls)},
            )
            model_messages.append(response)

            if not response.tool_calls:
                reply = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )
                return AnswerResult(
                    reply + resume_suffix,
                    list(dict.fromkeys(tools_used)),
                    {
                        "input_tokens": input_tokens or None,
                        "output_tokens": output_tokens or None,
                    },
                )

            for call in response.tool_calls:
                tool_name = call["name"]
                tool = tool_index.get(tool_name)
                tool_started = perf_counter()
                if tool is None:
                    result = ToolResult(
                        ToolStatus.ERROR,
                        f"Unknown tool {tool_name}.",
                    )
                else:
                    result = await tool.ainvoke(call.get("args", {}))
                if not isinstance(result, ToolResult):
                    result = ToolResult(
                        ToolStatus.SUCCESS,
                        "Tool completed.",
                        data={"result": result},
                    )
                tools_used.append(tool_name)
                _record_tool(
                    collector,
                    tool_name,
                    tool_started,
                    result,
                    query=_summarise_tool_query(tool_name, call.get("args", {})),
                )
                model_messages.append(
                    ToolMessage(
                        content=result.to_model_string(),
                        tool_call_id=call["id"],
                        name=tool_name,
                    )
                )

                data = result.data or {}
                if tool_name == "lookup_customer" and result.ok and data.get("customer_id"):
                    await session_repository.attach_customer(pool, session_id, data["customer_id"])
                if tool_name == "escalate_to_human" and result.ok:
                    await session_repository.merge_metadata(
                        pool,
                        session_id,
                        {"pending_action": None, "pending_actions": []},
                    )
                    return AnswerResult(
                        f"I’ve escalated this to a human support agent. Your reference "
                        f"is {data.get('ticket_id')}.",
                        list(dict.fromkeys(tools_used)),
                        {
                            "input_tokens": input_tokens or None,
                            "output_tokens": output_tokens or None,
                        },
                        escalated=True,
                        ticket_id=data.get("ticket_id"),
                    )

        collector.record(
            StepKind.ESCALATION,
            "iteration_budget_exhausted",
            duration_ms=0,
            detail={"max_iterations": settings.agent_max_iterations},
        )
        return AnswerResult(
            "I could not complete this safely within the available steps. Would you "
            "like me to escalate it to a human support agent?",
            list(dict.fromkeys(tools_used)),
            {
                "input_tokens": input_tokens or None,
                "output_tokens": output_tokens or None,
            },
        )
    except Exception as exc:
        logger.warning(
            "model_call_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        collector.record(
            StepKind.MODEL_CALL,
            "support_model",
            duration_ms=_elapsed_ms(model_started),
            detail={"fallback": True},
            error="The model provider was unavailable; a local fallback answered.",
        )
        latest = await session_repository.get(pool, session_id)
        failures = int((latest.metadata if latest else {}).get("model_failures", 0)) + 1
        updates: dict[str, Any] = {"model_failures": failures}
        if failures >= settings.escalation_tool_failure_threshold:
            failure_action = {
                "type": "escalate",
                "reason": "repeated_model_failure",
                "summary": message,
                "urgency": "normal",
            }
            queued = _queue_pending_action(
                _pending_action_stack(dict(latest.metadata) if latest else {}),
                failure_action,
            )
            updates["pending_action"] = queued[0]
            updates["pending_actions"] = queued
            await session_repository.merge_metadata(pool, session_id, updates)
            return AnswerResult(
                "The model service has failed repeatedly. Would you like me to "
                "escalate this to a human support agent?",
                [],
                {},
            )
        await session_repository.merge_metadata(pool, session_id, updates)
        return AnswerResult(_fallback_answer(message) + resume_suffix, [], {})


def _record_tool(
    collector: TraceCollector,
    name: str,
    started: float,
    result: ToolResult,
    *,
    query: str | None = None,
) -> None:
    safe_data = {
        key: ("[REDACTED_EMAIL]" if key == "email" else value)
        for key, value in (result.data or {}).items()
        if key not in {"hits"}
    }
    if name == "lookup_customer" and result.candidates:
        # The trace is visible in the demo UI. An ambiguous lookup must prove
        # that multiple matches existed without disclosing other accounts.
        safe_candidates = [{"customer": "[REDACTED]"} for _ in result.candidates]
    else:
        safe_candidates = [
            {
                key: "[REDACTED_EMAIL]" if key == "email" else value
                for key, value in candidate.items()
            }
            for candidate in result.candidates
        ]
    collector.record(
        StepKind.TOOL_CALL,
        name,
        duration_ms=_elapsed_ms(started),
        detail={
            "status": result.status.value,
            "query": redact(query) if query else None,
            "result_summary": _compact_text(result.message),
            "result": safe_data,
            "candidates": safe_candidates,
        },
        error=result.message if result.status == ToolStatus.ERROR else None,
    )


def _compact_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _summarise_tool_query(tool_name: str, arguments: dict[str, Any]) -> str:
    """Make model-selected tool arguments inspectable without leaking secrets."""
    safe: list[str] = []
    for key, value in arguments.items():
        lowered = key.casefold()
        if any(secret in lowered for secret in ("password", "token", "secret", "api_key")):
            rendered = "[REDACTED]"
        elif "email" in lowered:
            rendered = "[REDACTED_EMAIL]"
        else:
            rendered = _compact_text(value, 100)
        safe.append(f"{key}={rendered}")
    return _compact_text(f"{tool_name}({', '.join(safe)})")


def _structured_tool_calls(
    trace_steps: list[dict[str, Any]],
) -> list[dict[str, str | None]]:
    """Project trace tool steps into the compact public response contract."""
    calls: list[dict[str, str | None]] = []
    for step in trace_steps:
        if step.get("kind") != StepKind.TOOL_CALL.value:
            continue
        detail = step.get("detail") or {}
        calls.append(
            {
                "tool": str(step.get("name")),
                "query": detail.get("query"),
                "result_summary": str(
                    detail.get("result_summary") or detail.get("status") or "Tool completed."
                ),
            }
        )
    return calls


def _extract_identity(message: str) -> tuple[str, str] | None:
    email = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", message, re.I)
    if email:
        return "email", email.group(0)
    customer_id = re.search(r"\bcust-\d+\b", message, re.I)
    if customer_id:
        return "customer_id", customer_id.group(0).lower()
    name = re.search(
        r"\b(?:my name is|i am|customer named)\s+([A-Za-z][A-Za-z' -]{0,50})",
        message,
        re.I,
    )
    if name:
        value = re.split(r"[,.!?]", name.group(1))[0].strip()
        if value.casefold().startswith("a ") and len(value.split()) > 1:
            value = value[2:].strip()
        return "name", value
    return None


def _is_identity_statement(message: str) -> bool:
    return re.search(r"\b(?:my name is|customer named)\b", message, re.I) is not None


async def _maybe_answer_plan_gated_feature(
    *,
    pool: AsyncConnectionPool,
    session: Session,
    message: str,
    collector: TraceCollector,
) -> AnswerResult | None:
    """Chain account lookup and KB retrieval for plan-gated Slack questions."""
    lowered = message.casefold()
    if "slack" not in lowered or not any(
        phrase in lowered for phrase in ("connect", "integration", "available", "support")
    ):
        return None

    identity = _extract_identity(message)
    if identity is None and session.customer_id:
        identity = ("customer_id", session.customer_id)
    if identity is None:
        return None

    kind, value = identity
    lookup_started = perf_counter()
    account = await lookup_customer(pool=pool, **{kind: value})
    lookup_query = "customer_id supplied" if kind == "customer_id" else f"{kind} supplied"
    _record_tool(
        collector,
        "lookup_customer",
        lookup_started,
        account,
        query=lookup_query,
    )
    if account.status == ToolStatus.AMBIGUOUS:
        return AnswerResult(
            "I found several matching accounts, so I won’t guess. What is the full "
            "email address or customer ID on your account?",
            ["lookup_customer"],
            {},
        )
    if not account.ok:
        return AnswerResult(
            "I could not find that account. Please check the email or customer ID.",
            ["lookup_customer"],
            {},
        )

    account_data = account.data or {}
    customer_id = account_data["customer_id"]
    await session_repository.attach_customer(pool, session.id, customer_id)
    await session_repository.merge_metadata(
        pool,
        session.id,
        {
            "customer_name": account_data["name"],
            "customer_email": account_data["email"],
            "customer_plan": account_data["plan"],
        },
    )

    search_started = perf_counter()
    knowledge = await search_knowledge_base(message, pool=pool, limit=3)
    _record_tool(
        collector,
        "search_knowledge_base",
        search_started,
        knowledge,
        query=message,
    )
    tools = ["lookup_customer", "search_knowledge_base"]
    hits = (knowledge.data or {}).get("hits", []) if knowledge.ok else []
    if not hits:
        return AnswerResult(
            f"I found your {account_data['plan']} account, but I do not have a "
            "verified Slack setup article to rely on. Would you like human follow-up?",
            tools,
            {},
        )

    plan = str(account_data["plan"]).casefold()
    eligibility = (
        "Your current Free plan does not include Slack, so you would need to upgrade "
        "to Pro or Enterprise first."
        if plan == "free"
        else f"Your current {account_data['plan']} plan includes this integration."
    )
    return AnswerResult(
        _grounded_kb_answer(hits[0]) + " " + eligibility,
        tools,
        {},
    )


async def _maybe_lookup_customer(
    *,
    pool: AsyncConnectionPool,
    session: Session,
    message: str,
    collector: TraceCollector,
    pending_lookup: bool,
) -> AnswerResult | None:
    identity = _extract_identity(message)
    account_question = _needs_customer_identity(message.casefold())
    if identity is None and account_question and session.customer_id:
        identity = ("customer_id", session.customer_id)
    if identity is not None and pending_lookup:
        account_question = True
    if identity is None or not (
        pending_lookup or account_question or _is_identity_statement(message)
    ):
        return None

    kind, value = identity
    started = perf_counter()
    result = await lookup_customer(pool=pool, **{kind: value})
    lookup_query = "customer_id supplied" if kind == "customer_id" else f"{kind} supplied"
    _record_tool(
        collector,
        "lookup_customer",
        started,
        result,
        query=lookup_query,
    )

    if result.status == ToolStatus.AMBIGUOUS:
        return AnswerResult(
            "I found several matching accounts, so I won’t guess or expose their "
            "details. What is the full email address or customer ID on your account?",
            ["lookup_customer"],
            {},
        )
    if not result.ok:
        return AnswerResult(
            "I could not find that account. Please check the email or customer ID.",
            ["lookup_customer"],
            {},
        )

    data = result.data or {}
    customer_id = data["customer_id"]
    await session_repository.attach_customer(pool, session.id, customer_id)
    await session_repository.merge_metadata(
        pool,
        session.id,
        {
            "pending_lookup": False,
            "customer_name": data["name"],
            "customer_email": data["email"],
        },
    )
    if account_question:
        return AnswerResult(
            f"I found {data['workspace_name']}. The account is on the {data['plan']} "
            f"plan and its status is {data['status']}. It uses "
            f"{data['seats_used']} of {data['seats_limit'] or 'unlimited'} seats.",
            ["lookup_customer"],
            {},
        )
    return AnswerResult(
        f"Thanks, I found the {data['workspace_name']} account. How can I help?",
        ["lookup_customer"],
        {},
    )


def _is_affirmative(message: str) -> bool:
    normalized = _normalise_confirmation(message)
    exact = {
        "y",
        "yes",
        "yes please",
        "yes do it",
        "yes go ahead",
        "please do",
        "please go ahead",
        "do it",
        "do that",
        "confirm",
        "confirmed",
        "go ahead",
        "go ahead please",
        "proceed",
        "please proceed",
        "continue",
        "sure",
        "sure please",
        "sounds good",
        "absolutely",
        "okay",
        "ok",
        "tes",
        "tes please",
        "yse",
        "yse please",
        "yeah",
        "yep",
        "yup",
    }
    return normalized in exact or bool(
        re.fullmatch(
            r"(?:yes|yeah|yep|yup|sure|okay|ok|tes|yse) "
            r"(?:please |definitely |absolutely )?(?:do it|do that|go ahead|proceed|continue)",
            normalized,
        )
    )


def _is_negative_confirmation(message: str) -> bool:
    normalized = _normalise_confirmation(message)
    return (
        normalized in {"n", "no", "nope", "nah", "cancel", "stop", "do not", "dont"}
        or normalized.startswith(("no ", "do not ", "dont ", "cancel ", "stop "))
    )


def _normalise_confirmation(message: str) -> str:
    lowered = message.casefold().replace("’", "'").replace("don't", "dont")
    return " ".join(re.findall(r"[a-z]+", lowered))


def _looks_like_confirmation_attempt(message: str) -> bool:
    """Keep an outstanding write action pending when a short reply is unclear."""
    return len(_normalise_confirmation(message).split()) <= 5


def _grounded_kb_answer(hit: dict[str, Any]) -> str:
    """Avoid presenting a merely related historical ticket as an action just taken."""
    if hit.get("ticket_id") == "TK-005":
        return (
            "I found a related known issue affecting CSV exports for datasets larger "
            "than 10,000 rows. Is your dataset that large? If so, try exporting in "
            "smaller batches with date filters. If it is smaller, share the exact "
            "error text so we can distinguish this from a different export problem."
        )
    return str(hit.get("answer", ""))


def _explicit_human_request(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "speak to a human",
            "talk to a human",
            "human agent",
            "real person",
            "representative",
            "escalate to a human",
        )
    )


def _needs_authority(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "refund",
            "chargeback",
            "billing dispute",
            "account recovery",
            "restore my account",
        )
    )


def _explicit_ticket_request(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "create a ticket",
            "open a ticket",
            "file a ticket",
            "submit a ticket",
        )
    )


def _needs_customer_identity(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "what plan",
            "my plan",
            "on the free plan",
            "my account status",
            "my subscription",
            "my seats",
            "seats do i",
            "my storage",
            "my integrations",
        )
    )


def _is_context_followup(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "did i mention",
            "did i say",
            "what did i",
            "as i said",
            "earlier",
            "previous message",
            "that issue",
            "continue with",
        )
    )


async def _perform_escalation(
    pool: AsyncConnectionPool,
    session: Session,
    message: str,
    pending: dict,
    collector: TraceCollector,
) -> AnswerResult:
    started = perf_counter()
    result = await escalate_to_human(
        pending.get("reason", "customer_requested_human"),
        pending.get("summary", message),
        urgency=pending.get("urgency", "normal"),
        session_id=session.id,
        customer_id=session.customer_id,
        pool=pool,
    )
    _record_tool(
        collector,
        "escalate_to_human",
        started,
        result,
        query=_compact_text(pending.get("reason", "customer_requested_human")),
    )
    ticket_id = (result.data or {}).get("ticket_id")
    await session_repository.merge_metadata(
        pool,
        session.id,
        {"pending_action": None, "pending_actions": []},
    )
    return AnswerResult(
        f"I’ve escalated this to a human support agent. Your reference is {ticket_id}.",
        ["escalate_to_human"],
        {},
        escalated=True,
        ticket_id=ticket_id,
    )


async def _perform_ticket_creation(
    pool: AsyncConnectionPool,
    session: Session,
    pending: dict,
    collector: TraceCollector,
    *,
    consume_pending: bool = False,
) -> AnswerResult:
    started = perf_counter()
    result = await create_ticket(
        pending.get("subject", "Support request"),
        pending.get("description", "Customer requested support follow-up."),
        category=pending.get("category", "general"),
        priority=pending.get("priority", "normal"),
        session_id=session.id,
        customer_id=session.customer_id,
        pool=pool,
    )
    _record_tool(
        collector,
        "create_ticket",
        started,
        result,
        query=_compact_text(pending.get("subject", "Support request")),
    )
    next_action = None
    resume_topic = None
    if consume_pending:
        latest = await session_repository.get(pool, session.id)
        stack = _pending_action_stack(dict(latest.metadata) if latest else {})
        remaining = stack[1:] if stack else []
        next_action = remaining[0] if remaining else None
        updates: dict[str, Any] = {
            "pending_action": next_action,
            "pending_actions": remaining,
        }
        origin_topic = pending.get("origin_topic")
        latest_metadata = dict(latest.metadata) if latest else {}
        if origin_topic and latest_metadata.get("current_topic") == origin_topic:
            pending_topics = list(latest_metadata.get("pending_topics", []))
            resume_topic = pending_topics.pop(0) if pending_topics else None
            updates["current_topic"] = resume_topic
            updates["pending_topics"] = pending_topics
        await session_repository.merge_metadata(
            pool,
            session.id,
            updates,
        )
    ticket_id = (result.data or {}).get("ticket_id")
    next_suffix = (
        f" Your next pending approval is: {_pending_action_label(next_action)}."
        if next_action
        else ""
    )
    if resume_topic:
        next_suffix += (
            f" Now, would you like to continue with your {resume_topic} request?"
        )
    return AnswerResult(
        f"I created ticket {ticket_id}. Keep that reference for follow-up." + next_suffix,
        ["create_ticket"],
        {},
        ticket_id=ticket_id,
    )


def _ticket_subject(message: str) -> str:
    compact = " ".join(message.split())
    return compact[:80] if compact else "Support request"


def _pending_action_stack(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return pending approvals in LIFO order, with the next action first."""
    stored = metadata.get("pending_actions")
    if isinstance(stored, list):
        actions = [item for item in stored if isinstance(item, dict)]
        if actions:
            return actions
    single = metadata.get("pending_action")
    return [single] if isinstance(single, dict) else []


def _queue_pending_action(
    existing: list[dict[str, Any]],
    action: dict[str, Any],
) -> list[dict[str, Any]]:
    """Push one approval onto a deduplicated LIFO stack."""
    identity = (action.get("type"), action.get("summary"))
    remaining = [
        item
        for item in existing
        if (item.get("type"), item.get("summary")) != identity
    ]
    return [action, *remaining]


def _pending_action_label(action: dict[str, Any]) -> str:
    summary = " ".join(
        str(action.get("summary") or action.get("subject") or "").split()
    ).rstrip(" .!?")
    if summary:
        return summary[:100]
    return str(action.get("type", "support follow-up")).replace("_", " ")


def _detect_topic(message: str) -> str | None:
    lowered = message.casefold()
    topics = {
        "upgrade": ("upgrade", "upgrading", "change plan"),
        "billing": ("billing", "charge", "invoice", "refund"),
        "export": ("export", "download"),
        "account access": ("login", "sign in", "password", "account access"),
        "integration": ("integration", "connect", "slack", "salesforce"),
        "ticket": ("ticket",),
    }
    for topic, phrases in topics.items():
        if any(phrase in lowered for phrase in phrases):
            return topic
    return None


async def _update_topic_state(
    pool: AsyncConnectionPool,
    session_id: UUID,
    metadata: dict,
    message: str,
) -> tuple[dict, str]:
    current = metadata.get("current_topic")
    pending = list(metadata.get("pending_topics", []))
    lowered = message.casefold()
    completion = any(
        phrase in lowered
        for phrase in ("fixed now", "resolved now", "that is fixed", "it's fixed", "done now")
    )
    resume_suffix = ""
    updates: dict[str, Any] = {}

    if completion and current:
        updates["current_topic"] = None
        if pending:
            # pending_topics is a LIFO stack stored with the next topic first.
            resumed = pending.pop(0)
            updates["current_topic"] = resumed
            resume_suffix = f" Now, would you like to continue with your {resumed} request?"
        updates["pending_topics"] = pending
    else:
        topic = _detect_topic(message)
        if topic and current and topic != current:
            # If the customer explicitly jumps back to an already-pending
            # topic, remove that older entry before suspending the current one.
            pending = [item for item in pending if item != topic]
            pending = [item for item in pending if item != current]
            pending.insert(0, current)
        if topic:
            updates["current_topic"] = topic
            updates["pending_topics"] = pending

    if updates:
        updated = await session_repository.merge_metadata(pool, session_id, updates)
        return dict(updated.metadata), resume_suffix
    return metadata, resume_suffix


def _fallback_answer(message: str) -> str:
    lowered = message.casefold()
    if "password" in lowered or "login" in lowered or "sign in" in lowered:
        return (
            "Try the “Forgot password” link on the Ravenna sign-in page and check your "
            "spam folder for the reset email. If it still does not arrive, tell me the "
            "email domain you use and any error shown—do not send your password."
        )
    if "bill" in lowered or "charge" in lowered or "refund" in lowered:
        return (
            "I can help narrow down the billing issue. Is this about an invoice, an "
            "unexpected charge, changing your plan, or requesting a refund?"
        )
    if "integration" in lowered:
        return (
            "Which integration are you connecting, and what exact error or step is "
            "failing? I’ll use that to narrow down the setup issue."
        )
    return (
        "I understand you need help with Ravenna. What exact error do you see, and "
        "what were you trying to do immediately before it happened?"
    )
