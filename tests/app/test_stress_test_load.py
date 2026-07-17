"""Load smoke tests for POST /stress_test at batch limits."""

from __future__ import annotations

import time

from src.config.constants import MAX_BATCH_LOANS


def test_stress_test_within_time_budget(client, auth_headers) -> None:
    health = client.get("/health").json()
    if not health.get("model_loaded"):
        return

    payload = {
        "sample_size": min(200, MAX_BATCH_LOANS),
        "scenarios": ["Normal", "Recession"],
        "model_name": "xgb",
    }
    start = time.perf_counter()
    response = client.post("/stress_test", headers=auth_headers, json=payload)
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    body = response.json()
    assert body["loan_count"] <= MAX_BATCH_LOANS
    assert elapsed < 180.0


def test_stress_test_rejects_sample_size_above_max(client, auth_headers) -> None:
    health = client.get("/health").json()
    if not health.get("model_loaded"):
        return

    payload = {
        "sample_size": MAX_BATCH_LOANS + 1,
        "scenarios": ["Normal"],
        "model_name": "xgb",
    }
    response = client.post("/stress_test", headers=auth_headers, json=payload)
    assert response.status_code == 422
