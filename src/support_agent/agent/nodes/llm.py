"""LLM node — one model call with tools bound.

This is where "decide when to use a tool versus respond directly from
conversation context" actually happens. The node does not decide; the model does,
by either emitting tool calls or emitting a final answer. The node's job is to
give it the right context and record what it chose.

The model receives: the system prompt (with the identified customer folded in),
the conversation history restored by the checkpointer, and any tool results from
earlier iterations of this same turn. It does *not* receive a fresh copy of the
transcript — the checkpointer already holds it, and passing it twice is how
duplicate-context bugs start.

Increments `iterations`, which the conditional edge uses to enforce the per-turn
budget.
"""

from time import perf_counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from support_agent.agent.prompts import build_system_prompt
from support_agent.agent.state import AgentState
from support_agent.agent.trace import StepKind


async def llm_node(state: AgentState, model: BaseChatModel) -> AgentState:
    """Call the model and append its message to state.

    Records a `MODEL_CALL` trace step with the duration, token usage, and whether
    the model chose to call tools or answer — that last flag is what makes
    "why didn't it search?" answerable after the fact.
    """
    started = perf_counter()
    response = await model.ainvoke(
        [
            SystemMessage(
                content=build_system_prompt(customer_id=state.get("customer_id"))
            ),
            *state.get("messages", []),
        ]
    )
    usage = response.usage_metadata or {}
    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1,
        "trace_steps": [
            {
                "kind": StepKind.MODEL_CALL.value,
                "name": "support_model",
                "duration_ms": max(0, round((perf_counter() - started) * 1000)),
                "detail": {
                    "used_tools": bool(getattr(response, "tool_calls", [])),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                },
                "error": None,
            }
        ],
    }
