"""Input screening and log redaction.

This is the **second** layer of the prompt-injection defence, and the weaker one.
The first is structural and lives in `agent/prompts.py`: the system prompt is the
only text with authority, and everything else — customer messages, knowledge-base
article bodies, tool output — is delimited and labelled as data. That property
holds against phrasings nobody anticipated. What follows here is pattern matching,
which by construction only catches what it has seen before.

Stating that honestly matters more than the regex list. A heuristic screen sold as
a solution is worse than one sold as defence in depth, because it invites trusting
it. The realistic claim: this catches the obvious attempts, raises the effort
required for the rest, and makes attempts visible in the metrics.

The knowledge-base path is the one worth thinking hardest about. Article bodies
are trusted-looking text that the model is asked to act on — the classic indirect
injection surface. Our corpus is fixed and curated, so the risk is theoretical
here, but a real deployment ingests tickets written by customers, and then it is
not.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from support_agent.agent.prompts import REFUSAL_MESSAGE


class GuardrailFlag(StrEnum):
    """Also the `reason` label on `support_agent_guardrail_blocks_total`."""

    INJECTION = "injection"
    OUT_OF_SCOPE = "out_of_scope"
    OVERSIZED = "oversized"  # far past any plausible support question


@dataclass(frozen=True, slots=True)
class GuardrailVerdict:
    allowed: bool
    flags: list[GuardrailFlag]
    # Only when blocked. Customer-facing, stays in character, and never explains
    # the detection rule — that would be a tutorial on evading it.
    refusal_message: str | None = None


def screen_input(text: str, *, max_chars: int = 4000) -> GuardrailVerdict:
    """Screen one customer message before the model sees it."""
    flags: list[GuardrailFlag] = []
    if len(text) > max_chars:
        flags.append(GuardrailFlag.OVERSIZED)
    if detect_injection(text):
        flags.append(GuardrailFlag.INJECTION)
    if is_out_of_scope(text):
        flags.append(GuardrailFlag.OUT_OF_SCOPE)
    return GuardrailVerdict(
        allowed=not flags,
        flags=flags,
        refusal_message=None if not flags else REFUSAL_MESSAGE.strip(),
    )


def detect_injection(text: str) -> bool:
    """Heuristics for instruction-override attempts.

    Targets the recognisable families: overriding prior instructions, requests to
    reveal the system prompt, and attempts to re-cast the agent's role. Deliberately
    tuned to under-block — a frustrated customer writing "ignore what I said
    before, my actual problem is..." is a real support conversation, and refusing
    it is a worse failure than the injection it resembles.
    """
    lowered = text.casefold()
    patterns = (
        r"\bignore (?:all |any )?(?:previous|prior|system) instructions?\b",
        r"\bdisregard (?:all |every |any )?(?:previous |prior )?(?:rule|instruction)",
        r"\boverride (?:the |your )?(?:system |developer )?(?:prompt|instructions)",
        r"\breveal (?:the |your )?(?:system prompt|instructions)\b",
        r"\bshow me (?:the |your )?(?:system prompt|hidden instructions)\b",
        r"\bprint (?:the |your )?(?:initial configuration|system prompt)\b",
        r"\byou are now\b.{0,40}\b(?:developer|system|unrestricted)\b",
        r"\bact as\b.{0,40}\b(?:without rules|unrestricted|system)\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def is_out_of_scope(text: str) -> bool:
    """Whether the message is unrelated to supporting this product.

    Also tuned to under-block. Customers open with small talk and describe
    problems obliquely; a narrow scope check refuses real support requests, and
    the system prompt already keeps the agent on topic without refusing anyone.
    """
    lowered = text.casefold().strip()
    obvious_non_support = (
        "write me a poem",
        "write a poem",
        "solve this equation",
        "weather forecast",
        "political opinion",
        "write python code",
        "write javascript code",
        "generate python code",
        "explain python",
        "homework",
        "recipe",
    )
    return any(phrase in lowered for phrase in obvious_non_support)


def redact(text: str) -> str:
    """Mask emails, card-like digit runs, and API-key-shaped tokens for logging.

    Applied on the path to logs and traces, never to the model's context — the
    agent needs the real email to look an account up. This is about what we
    persist, not what it sees.
    """
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[REDACTED_EMAIL]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:\d[ -]*?){13,19}\b", "[REDACTED_NUMBER]", text)
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED_API_KEY]", text)
