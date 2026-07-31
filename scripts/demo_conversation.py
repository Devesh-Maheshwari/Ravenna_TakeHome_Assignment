"""Replay the worked example from the problem statement against a live API.

The conversation it drives, and what each turn is meant to demonstrate:

  1. "Hi, I need help with my Ravenna account"
        → vague opener; the agent should ask a clarifying question rather than
          guessing or calling a tool with nothing to go on.
  2. "I think I'm on the free plan but I want to upgrade"
        → account-specific; should trigger `lookup_customer`.
  3. "Actually, before that, the export feature hasn't been working..."
        → mid-conversation topic shift; should trigger `search_knowledge_base`
          (TK-005 covers the CSV export bug) and park the upgrade intent in
          `deferred_topics`.
  4. "Yes please"
        → confirmation; should trigger `create_ticket`, and the reply should
          then return to the parked upgrade topic unprompted.

Turn 4 is the interesting assertion. Answering the export question is ordinary
retrieval; coming back to the abandoned upgrade thread afterwards is what
separates a session-aware agent from a stateless question answerer.

Prints each turn's reply, tools used, and iteration count. Run against a seeded
database with the API up.
"""

import json
from uuid import uuid4

import httpx


def main() -> int:
    """Drive the scripted conversation and print the transcript with traces."""
    with httpx.Client(base_url="http://localhost:8000", timeout=60) as client:
        health = client.get("/health")
        health.raise_for_status()
        if not health.json()["database"]["reachable"]:
            print("Database is not reachable. Run `make up init` first.")
            return 1

        session = client.post(
            "/sessions",
            json={
                "metadata": {
                    "client_id": "demo",
                    "user_label": "Bob Smith",
                    "email": "bob@example.com",
                    "source": f"scripted-demo-{uuid4()}",
                }
            },
        )
        session.raise_for_status()
        session_id = session.json()["session_id"]
        print(f"session: {session_id}\n")

        turns = (
            "Hi, I need help with my Ravenna account.",
            "I think I'm on the free plan but I want to upgrade.",
            "Actually, before that, the export feature hasn't been working for me. "
            "Is that a known issue?",
            "Yes please.",
        )
        for message in turns:
            response = client.post(
                f"/sessions/{session_id}/messages",
                json={"message": message},
            )
            response.raise_for_status()
            body = response.json()
            print(f"customer: {message}")
            print(f"agent:    {body['response']}")
            print(
                "trace:    "
                + json.dumps(
                    {
                        "tools_used": body["tools_used"],
                        "tool_calls": body["tool_calls"],
                        "iterations": body["reasoning"]["iterations"],
                        "outcome": body["reasoning"]["outcome"],
                    }
                )
            )
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
