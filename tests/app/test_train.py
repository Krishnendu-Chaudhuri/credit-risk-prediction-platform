"""Integration tests for POST /train."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


def test_train_requires_auth(client: TestClient) -> None:
    assert client.post("/train", json={}).status_code == 401


def test_train_async_returns_job_and_completes(
    client: TestClient,
    auth_headers: dict[str, str],
    tiny_training_csv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_CSV_PATH", str(tiny_training_csv))
    start = client.post("/train", headers=auth_headers, json={"test_size": 0.2, "random_state": 42})
    assert start.status_code == 202
    job_id = start.json()["job_id"]
    assert start.json()["status"] == "pending"

    final_status = None
    for _ in range(120):
        status_resp = client.get(f"/train/status/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        final_status = status_resp.json()
        if final_status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.25)

    assert final_status is not None
    assert final_status["status"] == "completed"
    result = final_status["result"]
    assert result["best_model"] in {"lr", "xgb"}
    assert "metrics" in result
    assert "imputation_stats" in result["training_metadata"]


def test_train_status_not_found(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/train/status/does-not-exist", headers=auth_headers)
    assert response.status_code == 404
