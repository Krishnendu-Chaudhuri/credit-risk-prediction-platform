"""Load smoke tests for POST /predict at batch limits."""

from __future__ import annotations

import time

import pytest
from app.api.main import PredictRequest
from src.config.constants import MAX_BATCH_LOANS
from src.models.schemas import LoanInput


def test_predict_request_accepts_max_batch_size(sample_loan) -> None:
    loans = [LoanInput(**sample_loan) for _ in range(MAX_BATCH_LOANS)]
    request = PredictRequest(loans=loans)
    assert request.loans is not None
    assert len(request.loans) == MAX_BATCH_LOANS


def test_predict_moderate_batch_within_time_budget(client, auth_headers, sample_loan) -> None:
    health = client.get("/health").json()
    if not health.get("model_loaded"):
        return

    batch_size = min(200, MAX_BATCH_LOANS)
    loans = [LoanInput(**sample_loan).model_dump() for _ in range(batch_size)]
    start = time.perf_counter()
    response = client.post("/predict", headers=auth_headers, json={"loans": loans})
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == batch_size
    assert elapsed < 60.0


def test_predict_rejects_above_max_batch(client, auth_headers, sample_loan) -> None:
    loans = [LoanInput(**sample_loan).model_dump() for _ in range(MAX_BATCH_LOANS + 1)]
    response = client.post("/predict", headers=auth_headers, json={"loans": loans})
    assert response.status_code == 422


@pytest.mark.slow
def test_predict_at_max_batch_limit(client, auth_headers, sample_loan) -> None:
    """Full MAX_BATCH_LOANS inference; run with pytest -m slow when models are trained."""
    health = client.get("/health").json()
    if not health.get("model_loaded"):
        pytest.skip("Models not trained")

    loans = [LoanInput(**sample_loan).model_dump() for _ in range(MAX_BATCH_LOANS)]
    response = client.post("/predict", headers=auth_headers, json={"loans": loans})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == MAX_BATCH_LOANS
