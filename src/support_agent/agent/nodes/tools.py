"""Tool execution node.

Wraps LangGraph's `ToolNode` rather than using it directly, because the bare
version executes tools and returns results with no hook for the three things
this project needs: per-call timing, trace capture, and failure counting.

Failure handling is the important part. A tool that raises must not kill the
turn — the exception is converted into a tool *result* describing the failure, so
the model sees it, can tell the customer, and can try a different approach. An
unhandled exception here would surface to the customer as a 500, which is a worse
outcome than "I couldn't look that up, let me get you to someone who can."

`consecutive_tool_failures` is incremented on error and reset on success. The
conditional edge reads it to escalate deterministically, without relying on the
model to notice it is stuck.

Parallel tool calls are executed concurrently and their results returned together
— the model may legitimately ask for a customer lookup and a knowledge-base
search in one step.
"""

import asyncio
from time import perf_counter

from langchain_core.messages import ToolMessage

from support_agent.agent.state import AgentState
from support_agent.agent.trace import StepKind, summarise_tool_result
from support_agent.tools.base import ToolResult, ToolStatus


async def tools_node(state: AgentState, tools_by_name: dict) -> AgentState:
    """Execute every tool call in the model's last message.

    Records a `TOOL_CALL` trace step per call — name, arguments, summarised
    result, duration, and error if any — and updates the tool-call metrics.
    """
    messages = state.get("messages", [])
    calls = getattr(messages[-1], "tool_calls", []) if messages else []

    async def execute(call: dict) -> tuple[ToolMessage, dict, bool]:
        started = perf_counter()
        name = call["name"]
        tool = tools_by_name.get(name)
        error: str | None = None
        if tool is None:
            result = ToolResult(ToolStatus.ERROR, f"Unknown tool {name}.")
        else:
            try:
                result = await tool.ainvoke(call.get("args", {}))
            except Exception:
                result = ToolResult(
                    ToolStatus.ERROR,
                    "The support tool is temporarily unavailable. Do not guess.",
                )
        failed = isinstance(result, ToolResult) and result.status == ToolStatus.ERROR
        if failed:
            error = result.message
        model_content = (
            result.to_model_string() if isinstance(result, ToolResult) else str(result)
        )
        step = {
            "kind": StepKind.TOOL_CALL.value,
            "name": name,
            "duration_ms": max(0, round((perf_counter() - started) * 1000)),
            "detail": {
                "arguments": call.get("args", {}),
                "result": summarise_tool_result(name, result),
            },
            "error": error,
        }
        return (
            ToolMessage(
                content=model_content,
                tool_call_id=call.get("id", name),
                name=name,
            ),
            step,
            failed,
        )

    completed = await asyncio.gather(*(execute(call) for call in calls))
    failures = [failed for _, _, failed in completed]
    if failures and all(failures):
        failure_count = state.get("consecutive_tool_failures", 0) + 1
    else:
        failure_count = 0
    return {
        "messages": [message for message, _, _ in completed],
        "trace_steps": [step for _, step, _ in completed],
        "consecutive_tool_failures": failure_count,
    }
