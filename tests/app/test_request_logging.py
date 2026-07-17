"""Tests for request correlation middleware."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_echoes_request_id(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "test-correlation-id"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "test-correlation-id"


def test_health_generates_request_id(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")
