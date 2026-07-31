"""The state carried through one pass of the agent graph.

`messages` uses LangGraph's `add_messages` reducer, so nodes return only what
they add and the framework handles append-and-dedupe semantics.

Two fields deserve explanation because they are not standard ReAct scaffolding:

`deferred_topics` is what makes topic shifts work. In the worked example the
customer asks about upgrading, then interrupts themselves to report a broken
export. A plain ReAct loop answers the second question and forgets the first.
Parking the unfinished intent here — and telling the model in the prompt to pick
it back up once the interruption is resolved — is what produces "Now, would you
still like to explore upgrading your plan?" at the end.

`consecutive_tool_failures` is the deterministic half of escalation. Model
judgment alone is unreliable about admitting defeat; a counter is not.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State threaded through every node in the graph."""

    # --- Conversation -------------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    turn_index: int

    # --- Identity -----------------------------------------------------------
    # Set once `lookup_customer` succeeds, so later turns skip the lookup.
    customer_id: str | None

    # --- Topic tracking -----------------------------------------------------
    # Intents raised but not yet resolved, oldest first. Drained as they are
    # addressed; the prompt instructs the model to return to them.
    deferred_topics: list[str]

    # --- Loop control -------------------------------------------------------
    # llm<->tools round trips used this turn, against settings.agent_max_iterations.
    iterations: int
    consecutive_tool_failures: int

    # --- Guardrail outcome --------------------------------------------------
    # Set by the guardrail node; when present the graph skips the model entirely
    # and goes straight to finalize with this canned refusal.
    refusal: str | None
    guardrail_flags: list[str]  # e.g. ['injection'], ['out_of_scope']

    # --- Escalation ---------------------------------------------------------
    escalated: bool
    escalation_reason: str | None
    ticket_id: str | None
    final_response: str | None
    outcome: str | None

    # --- Observability ------------------------------------------------------
    trace_id: str
    # Ordered node visits and tool calls; assembled by agent/trace.py.
    trace_steps: Annotated[list[dict[str, Any]], operator.add]


def initial_state(
    *,
    session_id: str,
    turn_index: int,
    trace_id: str,
    customer_id: str | None = None,
    deferred_topics: list[str] | None = None,
) -> AgentState:
    """Build the state for a fresh turn.

    Conversation history is *not* passed in — the checkpointer restores it from
    the previous turn under the same `thread_id`.
    """
    return AgentState(
        messages=[],
        session_id=session_id,
        turn_index=turn_index,
        customer_id=customer_id,
        deferred_topics=list(deferred_topics or []),
        iterations=0,
        consecutive_tool_failures=0,
        refusal=None,
        guardrail_flags=[],
        escalated=False,
        escalation_reason=None,
        ticket_id=None,
        final_response=None,
        outcome=None,
        trace_id=trace_id,
        trace_steps=[],
    )


def budget_exhausted(state: AgentState, max_iterations: int) -> bool:
    """True when the turn has used its iteration budget and must wrap up."""
    return state.get("iterations", 0) >= max_iterations
