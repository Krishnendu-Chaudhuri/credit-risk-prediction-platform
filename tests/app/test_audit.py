"""Tests for prediction audit store."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.audit_store import get_prediction_audit, log_prediction


@pytest.fixture
def audit_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "audit.db"
    monkeypatch.setenv("AUDIT_DB_PATH", str(db))
    import app.api.audit_store as store

    monkeypatch.setattr(store, "DB_PATH", db)
    return db


def test_audit_round_trip(audit_db: Path) -> None:
    pid = log_prediction(
        inputs={"loan_amnt": 10000},
        outputs={"pd": 0.12},
        shap_contributions=[{"feature": "loan_amnt", "contribution": 0.01}],
        reason_codes=["EXCESSIVE_OBLIGATIONS"],
        prediction_id="test-id-1",
    )
    assert pid == "test-id-1"
    record = get_prediction_audit("test-id-1")
    assert record is not None
    assert record["inputs"]["loan_amnt"] == 10000
    assert record["reason_codes"] == ["EXCESSIVE_OBLIGATIONS"]


def test_audit_db_created_on_first_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "nested" / "audit.db"
    assert not db.exists()
    monkeypatch.setenv("AUDIT_DB_PATH", str(db))
    import app.api.audit_store as store

    monkeypatch.setattr(store, "DB_PATH", db)
    log_prediction(inputs={"loan_amnt": 1}, outputs={"pd": 0.1}, prediction_id="create-test")
    assert db.exists()
