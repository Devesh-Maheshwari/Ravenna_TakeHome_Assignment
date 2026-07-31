"""HTTP client for the support agent API.

Imports nothing from `src/support_agent` — deliberately. The UI is an ordinary
consumer of the published API, so if something is awkward to do here it is
awkward for any client, and that is worth finding out before a reviewer does.
"""

from typing import Any

import httpx


class SupportAgentAPIError(RuntimeError):
    """An API failure with a message suitable for displaying in the UI."""


class SupportAgentClient:
    """Thin wrapper over the endpoints the UI needs."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0) -> None:
        # Generous timeout: a turn with several chained tool calls legitimately
        # takes tens of seconds, and a client that gives up mid-turn looks like
        # a server bug.
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            try:
                body = exc.response.json()
            except ValueError:
                body = {}
            message = body.get("message") or body.get("detail") or exc.response.reason_phrase
            raise SupportAgentAPIError(
                f"API request failed ({exc.response.status_code}): {message}"
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise SupportAgentAPIError(f"Could not reach the support API: {exc}") from exc

    def create_session(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """POST /sessions."""
        return self._request("POST", "/sessions", json={"metadata": metadata or {}})

    def send_message(self, session_id: str, message: str) -> dict[str, Any]:
        """POST /sessions/{id}/messages — reply plus tools_used plus trace."""
        return self._request(
            "POST",
            f"/sessions/{session_id}/messages",
            json={"message": message},
        )

    def get_session(self, session_id: str) -> dict[str, Any]:
        """GET /sessions/{id}."""
        return self._request("GET", f"/sessions/{session_id}")

    def list_sessions(
        self,
        *,
        client_id: str | None = None,
        customer_id: str | None = None,
        status: str | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent conversations scoped to a browser or selected demo customer."""
        params = {
            key: value
            for key, value in {
                "client_id": client_id,
                "customer_id": customer_id,
                "status": status,
                "source": source,
            }.items()
            if value is not None
        }
        result = self._request("GET", "/sessions", params=params)
        return result["sessions"]

    def list_demo_customers(self) -> list[dict[str, Any]]:
        """Seeded names available in the local demo customer picker."""
        return self._request("GET", "/demo/customers")["customers"]

    def close_session(self, session_id: str) -> dict[str, Any]:
        """Close a conversation without deleting its history."""
        return self._request("POST", f"/sessions/{session_id}/close")

    def health(self) -> bool:
        """Whether the API is up, so the UI can say so instead of throwing."""
        try:
            response = self._client.get("/health", timeout=2.0)
            return response.is_success
        except httpx.RequestError:
            return False
