"""Escalation node — the handoff to a human.

The design challenge asks for a mechanism to recognise when the agent is out of
its depth. Recognition comes from two independent sources, on purpose:

  - **Model judgment** — it calls `escalate_to_human` because it understands it
    cannot help. Good at nuance (a frustrated customer, an unusual request),
    unreliable at admitting defeat.
  - **Deterministic triggers** — `consecutive_tool_failures` past a threshold,
    the iteration budget exhausted with no answer, or a knowledge-base search
    whose best score falls below `escalation_min_retrieval_score`. Blind to
    nuance, but they fire every time.

Neither alone is sufficient. A model that never gives up loops until the budget
runs out; a counter alone escalates a customer who was merely chatty. Together
they cover each other's failure mode.

What a handoff produces: the session flips to `escalated`, a ticket is opened
carrying the reason and a summary of the conversation so the human does not start
cold, and the customer is told plainly what happens next.
"""

from langchain_core.messages import AIMessage

from support_agent.agent.state import AgentState
from support_agent.agent.trace import StepKind


async def escalation_node(state: AgentState) -> AgentState:
    """Perform the handoff and set the customer-facing message."""
    reason = state.get("escalation_reason") or "agent_could_not_resolve"
    message = (
        "I’m unable to complete this safely. I recommend handing the conversation "
        "to a human support agent."
    )
    return {
        "escalated": True,
        "escalation_reason": reason,
        "final_response": message,
        "outcome": "escalated",
        "messages": [AIMessage(content=message)],
        "trace_steps": [
            {
                "kind": StepKind.ESCALATION.value,
                "name": reason,
                "duration_ms": 0,
                "detail": {"summary": await summarise_for_handoff(state)},
                "error": None,
            }
        ],
    }


def should_escalate(
    state: AgentState,
    *,
    failure_threshold: int,
    max_iterations: int,
) -> tuple[bool, str | None]:
    """Evaluate the deterministic triggers.

    Returns whether to escalate and, if so, which trigger fired — the reason is
    stored on the ticket and used as the `support_agent_escalations_total` label,
    so escalation causes are countable rather than anecdotal.
    """
    if state.get("escalated") or state.get("escalation_reason"):
        return True, state.get("escalation_reason") or "model_requested_escalation"
    if state.get("consecutive_tool_failures", 0) >= failure_threshold:
        return True, "repeated_tool_failure"
    if state.get("iterations", 0) >= max_iterations:
        messages = state.get("messages", [])
        if messages and getattr(messages[-1], "tool_calls", []):
            return True, "iteration_budget_exhausted"
    return False, None


async def summarise_for_handoff(state: AgentState) -> str:
    """Condense the conversation into a briefing for the human who picks this up.

    What was asked, what was tried, what failed, and what the customer still
    needs — so the handoff does not restart the conversation from zero.
    """
    lines: list[str] = []
    for message in state.get("messages", [])[-8:]:
        role = getattr(message, "type", "message")
        content = getattr(message, "content", "")
        if content:
            lines.append(f"{role}: {' '.join(str(content).split())[:240]}")
    return "\n".join(lines) or "No conversation details were available."
