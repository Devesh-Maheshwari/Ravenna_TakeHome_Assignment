"""Prometheus metrics — the aggregate layer of observability.

Traces answer "why was *this* response wrong?"; metrics answer "is something
wrong right now?". They are complementary and neither substitutes for the other:
you cannot debug a specific bad answer from a histogram, and you cannot notice
that tool errors tripled from a single trace.

These are deliberately *domain* metrics rather than generic HTTP counts. Request
rate and 5xx rate say nothing about an agent that returns 200 with a confidently
wrong answer. Escalation rate, tool error rate, and iteration count do. That is
the difference between monitoring a web service and monitoring an agent.

Exposed at `GET /metrics` in Prometheus exposition format, which is why no
vendor SDK is needed — pointing Grafana, Datadog, or anything else at this is a
scrape config, not a code change. See docs/DESIGN_DECISIONS.md.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

# Own registry rather than the global default, so tests can build a clean one and
# assert on counter values without cross-test contamination.
REGISTRY = CollectorRegistry()

NAMESPACE = "support_agent"


def build_metrics(registry: CollectorRegistry = REGISTRY) -> dict:
    """Declare every metric against `registry`.

    Returns them keyed by name for injection. Declared in one function rather
    than as module-level globals so the registry is a parameter, which is what
    makes them testable.

    Defined here:

      turns_total{outcome}            resolved | escalated | refused | error
      turn_latency_seconds            end-to-end per turn
      tool_calls_total{tool,outcome}  success | not_found | ambiguous | error
      tool_latency_seconds{tool}      per tool
      llm_tokens_total{direction}     input | output
      agent_iterations                llm<->tools round trips; catches runaway loops
      guardrail_blocks_total{reason}  injection | out_of_scope | oversized
      escalations_total{reason}       which trigger fired
      active_sessions                 gauge
    """
    return {
        "turns_total": Counter(
            "turns_total",
            "Completed agent turns.",
            ("outcome",),
            namespace=NAMESPACE,
            registry=registry,
        ),
        "turn_latency_seconds": Histogram(
            "turn_latency_seconds",
            "End-to-end latency for an agent turn.",
            namespace=NAMESPACE,
            registry=registry,
        ),
        "tool_calls_total": Counter(
            "tool_calls_total",
            "Agent tool calls.",
            ("tool", "outcome"),
            namespace=NAMESPACE,
            registry=registry,
        ),
        "tool_latency_seconds": Histogram(
            "tool_latency_seconds",
            "Tool-call latency.",
            ("tool",),
            namespace=NAMESPACE,
            registry=registry,
        ),
        "llm_tokens_total": Counter(
            "llm_tokens_total",
            "Tokens consumed by model calls.",
            ("direction",),
            namespace=NAMESPACE,
            registry=registry,
        ),
        "agent_iterations": Histogram(
            "agent_iterations",
            "LLM-to-tools round trips per turn.",
            namespace=NAMESPACE,
            registry=registry,
        ),
        "guardrail_blocks_total": Counter(
            "guardrail_blocks_total",
            "Messages blocked by guardrails.",
            ("reason",),
            namespace=NAMESPACE,
            registry=registry,
        ),
        "escalations_total": Counter(
            "escalations_total",
            "Agent escalations.",
            ("reason",),
            namespace=NAMESPACE,
            registry=registry,
        ),
        "active_sessions": Gauge(
            "active_sessions",
            "Currently active sessions.",
            namespace=NAMESPACE,
            registry=registry,
        ),
    }


def render(registry: CollectorRegistry = REGISTRY) -> tuple[bytes, str]:
    """Serialise for `GET /metrics`. Returns the payload and its content type."""
    return generate_latest(registry), CONTENT_TYPE_LATEST
