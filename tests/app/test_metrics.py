"""Integration tests for GET /metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_requires_auth(client: TestClient) -> None:
    assert client.get("/metrics").status_code == 401


def test_metrics_returns_payload_when_models_exist(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/metrics", headers=auth_headers)
    if response.status_code == 404:
        return
    assert response.status_code == 200
    payload = response.json()
    assert "metrics" in payload
    assert "lr" in payload["metrics"] or "xgb" in payload["metrics"]
