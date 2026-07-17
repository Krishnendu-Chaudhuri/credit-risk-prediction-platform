"""Shared pytest fixtures for the credit risk PD engine."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def test_api_key() -> None:
    from app.api.settings import load_env

    load_env()
    os.environ["API_KEY"] = "test-key"
    os.environ.setdefault("ENVIRONMENT", "development")
    from app.api import auth as auth_module

    auth_module.API_KEY = "test-key"


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test-key"}


@pytest.fixture
def client() -> TestClient:
    from app.api.main import app

    return TestClient(app)


@pytest.fixture
def redis_client(monkeypatch: pytest.MonkeyPatch):
    import fakeredis

    from app.api import redis_store

    fake = fakeredis.FakeRedis(decode_responses=True)
    redis_store.reset_redis_client()
    monkeypatch.setenv("REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(redis_store, "get_redis", lambda: fake)
    yield fake
    fake.flushall()
    redis_store.reset_redis_client()
    monkeypatch.delenv("REDIS_URL", raising=False)


@pytest.fixture(scope="session")
def ensure_macro_data() -> Path:
    path = ROOT / "assets" / "data" / "US_Macro_Economic_Stress_Test_Data.xlsx"
    if not path.exists():
        from src.utils.generate_macro import main

        main()
    return path


@pytest.fixture(scope="session")
def tiny_training_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(42)
    n = 120
    grades = ["A", "B", "C", "D"]
    df = pd.DataFrame(
        {
            "person_age": rng.integers(22, 65, n),
            "person_income": rng.integers(20000, 150000, n),
            "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
            "person_emp_length": rng.uniform(0, 20, n).round(1),
            "loan_intent": rng.choice(["PERSONAL", "EDUCATION", "MEDICAL"], n),
            "loan_grade": rng.choice(grades, n),
            "loan_amnt": rng.integers(1000, 30000, n),
            "loan_int_rate": rng.uniform(5, 20, n).round(2),
            "loan_status": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "loan_percent_income": rng.uniform(0.05, 0.5, n).round(2),
            "cb_person_default_on_file": rng.choice(["Y", "N"], n, p=[0.1, 0.9]),
            "cb_person_cred_hist_length": rng.integers(1, 15, n),
        }
    )
    csv_path = tmp_path_factory.mktemp("data") / "tiny_credit.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_loan() -> dict:
    return {
        "person_age": 35,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 12.0,
        "loan_percent_income": 0.3,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3,
    }
