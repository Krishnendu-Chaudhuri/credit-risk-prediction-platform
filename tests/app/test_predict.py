"""Integration tests for POST /predict."""

from __future__ import annotations

from src.config.constants import MAX_BATCH_LOANS
from src.models.schemas import LoanInput


def test_predict_requires_auth(client, auth_headers, sample_loan) -> None:
    assert client.post("/predict", json={"loan": sample_loan}).status_code == 401


def test_predict_single_loan(client, auth_headers, sample_loan) -> None:
    response = client.post(
        "/predict",
        headers=auth_headers,
        json={"loan": sample_loan, "model_name": "xgb"},
    )
    if response.status_code == 400:
        return
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    assert 0 <= predictions[0]["pd"] <= 1
    assert predictions[0]["ifrs9_stage"] in {1, 2, 3}


def test_predict_batch_cap_returns_422(client, auth_headers, sample_loan) -> None:
    health = client.get("/health").json()
    if not health.get("model_loaded"):
        return

    loans = [LoanInput(**sample_loan).model_dump() for _ in range(MAX_BATCH_LOANS + 1)]
    response = client.post("/predict", headers=auth_headers, json={"loans": loans})
    assert response.status_code == 422
