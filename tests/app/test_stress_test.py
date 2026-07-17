"""Integration tests for POST /stress_test."""

from __future__ import annotations


def test_stress_test_requires_auth(client, ensure_macro_data) -> None:
    assert client.post("/stress_test", json={"sample_size": 100}).status_code == 401


def test_stress_test_valid_request(client, auth_headers, ensure_macro_data) -> None:
    health = client.get("/health").json()
    if not health.get("model_loaded"):
        return

    response = client.post(
        "/stress_test",
        headers=auth_headers,
        json={
            "sample_size": 50,
            "scenarios": ["Normal", "Recession"],
            "model_name": "xgb",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["loan_count"] == 50
    assert len(payload["results"]) == 2
    assert len(payload["comparison"]) == 2


def test_stress_test_rejects_invalid_sample_size(client, auth_headers, ensure_macro_data) -> None:
    response = client.post(
        "/stress_test",
        headers=auth_headers,
        json={"sample_size": 0, "scenarios": ["Normal"]},
    )
    assert response.status_code == 422
