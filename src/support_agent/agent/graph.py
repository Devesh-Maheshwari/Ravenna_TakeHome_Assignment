"""The agent loop: a LangGraph `StateGraph` wired by hand.

    entry → guardrail → llm ⇄ tools → finalize → end
                 ↓                        ↑
              (refuse) ──────────────────┘
         any node → escalation → finalize

`create_react_agent` would give us the `llm ⇄ tools` cycle for free, and nowhere
to put the rest of it: a guardrail that runs before the model ever sees the
input, an escalation path that can fire from anywhere, and per-node trace
capture. Those three are the substance of this assignment, so the loop is built
explicitly instead.

The `llm ⇄ tools` cycle is what produces multi-step tool chaining: the model
calls a tool, reads the result, and decides whether it needs another before it
answers. `route_after_llm` is the conditional edge that decides.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from support_agent.agent.llm import build_chat_model
from support_agent.agent.nodes.escalation import escalation_node, should_escalate
from support_agent.agent.nodes.guardrail import guardrail_node
from support_agent.agent.nodes.llm import llm_node
from support_agent.agent.nodes.tools import tools_node
from support_agent.agent.state import AgentState
from support_agent.agent.trace import StepKind
from support_agent.config import Settings
from support_agent.tools.registry import tools_by_name


def build_graph(
    *,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    tools: list,
) -> CompiledStateGraph:
    """Assemble and compile the graph.

    Compiled once at startup, not per request: compilation is not free and the
    graph is stateless between turns. Per-conversation state lives in the
    checkpointer, addressed by `thread_id` at invoke time.
    """
    model = build_chat_model(settings, tools=tools)
    tool_index = tools_by_name(tools)

    async def call_model(state: AgentState) -> AgentState:
        return await llm_node(state, model)

    async def call_tools(state: AgentState) -> AgentState:
        return await tools_node(state, tool_index)

    graph = StateGraph(AgentState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("llm", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("escalation", escalation_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"llm": "llm", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "llm",
        lambda state: route_after_llm(state, settings.agent_max_iterations),
        {"tools": "tools", "escalation": "escalation", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "tools",
        lambda state: route_after_tools(
            state, settings.escalation_tool_failure_threshold
        ),
        {"llm": "llm", "escalation": "escalation"},
    )
    graph.add_edge("escalation", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def route_after_guardrail(state: AgentState) -> str:
    """`finalize` if the guardrail refused the input, otherwise `llm`."""
    return "finalize" if state.get("refusal") else "llm"


def route_after_llm(state: AgentState, max_iterations: int) -> str:
    """Decide what follows the model's turn.

    - Model requested tool calls and budget remains → `tools`
    - Budget exhausted → `finalize`, forcing a graceful wrap-up rather than an
      unbounded loop
    - Escalation triggered → `escalation`
    - Otherwise the model produced a final answer → `finalize`
    """
    escalate, _ = should_escalate(
        state,
        failure_threshold=2**31 - 1,
        max_iterations=max_iterations,
    )
    if escalate:
        return "escalation"
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if getattr(last, "tool_calls", []):
        return "tools" if state.get("iterations", 0) < max_iterations else "escalation"
    return "finalize"


def route_after_tools(state: AgentState, failure_threshold: int) -> str:
    """`escalation` once tools have failed repeatedly, otherwise back to `llm`.

    This is the deterministic escalation trigger: it does not depend on the model
    noticing that it is stuck.
    """
    return (
        "escalation"
        if state.get("consecutive_tool_failures", 0) >= failure_threshold
        else "llm"
    )


async def finalize(state: AgentState) -> AgentState:
    """Close out the turn.

    Picks the customer-visible text (the model's answer, the guardrail refusal,
    or the escalation message), stamps the outcome, and seals the trace. Small
    enough to live here rather than in its own node module.
    """
    if state.get("refusal"):
        response = state["refusal"]
        outcome = "refused"
    elif state.get("escalated"):
        response = state.get("final_response") or (
            "I recommend handing this conversation to a human support agent."
        )
        outcome = "escalated"
    else:
        response = ""
        for message in reversed(state.get("messages", [])):
            if getattr(message, "type", "") == "ai" and not getattr(
                message, "tool_calls", []
            ):
                content = message.content
                response = content if isinstance(content, str) else str(content)
                break
        if not response:
            response = (
                "I could not complete this safely within the available steps. "
                "Would you like me to escalate it to a human support agent?"
            )
        outcome = "resolved"
    return {
        "final_response": response,
        "outcome": outcome,
        "trace_steps": [
            {
                "kind": StepKind.FINALIZE.value,
                "name": "finalize",
                "duration_ms": 0,
                "detail": {"outcome": outcome},
                "error": None,
            }
        ],
    }
