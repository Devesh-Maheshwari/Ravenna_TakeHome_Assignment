"""Executable fixtures for the database-backed acceptance suite."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from support_agent.main import create_app


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Run the real app lifespan against the configured local Postgres."""
    with TestClient(create_app()) as test_client:
        health = test_client.get("/health")
        if not health.is_success or not health.json()["database"]["reachable"]:
            pytest.skip("Local Postgres is required: run `make up init` first")
        yield test_client


@pytest.fixture
def session_id(client: TestClient) -> str:
    response = client.post(
        "/sessions",
        json={"metadata": {"client_id": "pytest", "user_label": "Test customer"}},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "capability(number): maps a test to one requirement")
