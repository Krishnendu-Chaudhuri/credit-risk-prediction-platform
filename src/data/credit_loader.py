"""Load and clean credit risk CSV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import CREDIT_CSV_FILE, require_file
from src.config.paths import MODEL_DIR as DEFAULT_MODEL_DIR
from src.features.engineering import clean_raw_data, engineer_features
from src.utils import get_env_path, load_json

REQUIRED_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]


def _load_imputation_stats() -> dict | None:
    model_dir = get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR)
    meta_path = model_dir / "training_metadata.json"
    if not meta_path.exists():
        return None
    return load_json(meta_path).get("imputation_stats")


def load_credit_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else get_env_path("DATA_CSV_PATH", CREDIT_CSV_FILE)
    require_file(path)
    df = pd.read_csv(path)
    df = clean_raw_data(df)

    imputation_stats = _load_imputation_stats()
    if imputation_stats:
        engineered = engineer_features(df, imputation_stats=imputation_stats)
        keep_cols = REQUIRED_COLUMNS + (["loan_status"] if "loan_status" in engineered.columns else [])
        return engineered[keep_cols].dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("unknown")
            else:
                df[col] = df[col].fillna(df[col].median())

    return df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)


def sample_portfolio(n: int = 500, csv_path: str | Path | None = None, random_state: int = 42) -> pd.DataFrame:
    df = load_credit_data(csv_path)
    n = min(n, len(df))
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)
