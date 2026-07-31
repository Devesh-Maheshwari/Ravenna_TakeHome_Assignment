"""Guardrail node — runs before the model sees the input.

Placed first in the graph deliberately: screening after generation means paying
for tokens on input that was never going to be answered, and risks a partial
response leaking before the check completes.

Two checks, both cheap and heuristic (`security/guardrails.py` holds the logic):

  - **Injection** — attempts to override the system prompt, extract it, or
    re-cast the agent's role.
  - **Scope** — requests that have nothing to do with customer support.

This is the *second* line of defence against prompt injection, not the first.
The first is structural, in `agent/prompts.py`: the system prompt is the only
text with authority and everything else is framed as data. Heuristics catch
known phrasings; the framing is what holds when the phrasing is novel.

A refusal short-circuits to `finalize` with a canned message. No model call, no
tool access.
"""

from support_agent.agent.state import AgentState
from support_agent.agent.trace import StepKind
from support_agent.security.guardrails import screen_input


async def guardrail_node(state: AgentState) -> AgentState:
    """Screen the latest customer message.

    Returns state with `refusal` set to short-circuit the turn, or with
    `guardrail_flags` recorded and the turn allowed to continue. Records a
    `GUARDRAIL` trace step either way — a clean pass is worth seeing in a trace.
    """
    verdict = screen_input(_latest_customer_text(state))
    flags = [flag.value for flag in verdict.flags]
    return {
        "refusal": verdict.refusal_message,
        "guardrail_flags": flags,
        "trace_steps": [
            {
                "kind": StepKind.GUARDRAIL.value,
                "name": "screen_input",
                "duration_ms": 0,
                "detail": {"allowed": verdict.allowed, "flags": flags},
                "error": None,
            }
        ],
    }


def _latest_customer_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""
