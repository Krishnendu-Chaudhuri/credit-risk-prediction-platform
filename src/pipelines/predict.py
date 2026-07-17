"""Inference for credit risk PD models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.config.constants import CREDIT_CSV_PATH
from src.config.paths import MODEL_DIR as DEFAULT_MODEL_DIR
from src.data.integrity import load_data_manifest, verify_manifest
from src.features.encoding_monitor import check_unknown_categories
from src.features.engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features, loan_input_to_frame
from src.models.calibration import apply_calibrator, load_calibrator
from src.models.registry import resolve_model_dir
from src.models.risk_scoring import enrich_prediction
from src.models.schemas import FeatureContribution, LoanInput, PredictionOutput
from src.utils import get_env_path, load_artifacts, models_exist
from src.utils.explain import explain_prediction
from src.utils.reason_codes import map_reason_codes

logger = logging.getLogger(__name__)


def _get_model(artifacts: dict, model_name: str):
    if model_name == "lr":
        return artifacts["lr_model"]
    return artifacts["xgb_model"]


def _get_calibrator(model_dir: Path, model_name: str):
    path = model_dir / f"calibrator_{model_name}.joblib"
    if path.exists():
        return load_calibrator(path)
    return None


def _feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _prepare_features(
    loans: list[dict] | pd.DataFrame,
    preprocessor,
    imputation_stats: dict | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    if isinstance(loans, pd.DataFrame):
        frame = loans
    else:
        frame = pd.DataFrame(loans)
    engineered = engineer_features(frame, imputation_stats=imputation_stats)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    transformed = preprocessor.transform(engineered[feature_cols])
    check_unknown_categories(transformed, preprocessor, engineered)
    return transformed, engineered


def predict_pd(
    loan: LoanInput | list[LoanInput] | pd.DataFrame,
    model_name: Literal["xgb", "lr"] = "xgb",
    lgd: float = 0.45,
    model_dir: str | None = None,
    audit: bool = True,
    user_id: str | None = None,
) -> PredictionOutput | list[PredictionOutput]:
    base_path = get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR) if model_dir is None else Path(model_dir)
    path = resolve_model_dir(base_path) if model_dir is None else base_path
    if not models_exist(path):
        raise FileNotFoundError(f"No trained models found in {path}. Run training first.")

    artifacts = load_artifacts(path)
    model = _get_model(artifacts, model_name)
    preprocessor = artifacts["preprocessor"]
    metadata = artifacts["metadata"]
    imputation_stats = metadata.get("imputation_stats")
    thresholds = metadata.get("optimal_threshold", {})
    threshold = float(thresholds.get(model_name, 0.5))
    calibrator = _get_calibrator(path, model_name)
    feature_names = _feature_names(preprocessor)

    manifest = metadata.get("data_manifest") or load_data_manifest(path)
    if manifest:
        csv_rel = metadata.get("csv_path", CREDIT_CSV_PATH)
        csv_path = get_env_path("DATA_CSV_PATH", csv_rel)
        verify_manifest(manifest, [csv_path])

    if isinstance(loan, pd.DataFrame):
        loans_df = loan
        single = False
    elif isinstance(loan, list):
        loans_df = pd.DataFrame([item.model_dump() for item in loan])
        single = False
    else:
        loans_df = loan_input_to_frame(loan.model_dump())
        single = True

    X, engineered = _prepare_features(loans_df, preprocessor, imputation_stats=imputation_stats)
    raw_probs = model.predict_proba(X)[:, 1]
    probs = apply_calibrator(calibrator, raw_probs) if calibrator is not None else raw_probs

    outputs: list[PredictionOutput] = []
    for idx, (raw_pd, cal_pd) in enumerate(zip(raw_probs, probs, strict=False)):
        row = loans_df.iloc[idx]
        enriched = enrich_prediction(
            float(raw_pd),
            float(row["loan_amnt"]),
            lgd=lgd,
            loan_int_rate=float(row.get("loan_int_rate", 0)),
            loan_intent=str(row.get("loan_intent", "")),
            loan_grade=str(row.get("loan_grade", "")),
            threshold=threshold,
            calibrated_pd=float(cal_pd),
        )
        shap_raw = explain_prediction(model, model_name, X[idx], feature_names, top_n=5)
        shap_contribs = [FeatureContribution(**item) for item in shap_raw]
        reason_codes = map_reason_codes(shap_raw)

        prediction_id = None
        if audit:
            from app.api.audit_store import log_prediction

            prediction_id = log_prediction(
                inputs=row.to_dict(),
                outputs=enriched,
                shap_contributions=shap_raw,
                reason_codes=reason_codes,
                user_id=user_id,
            )

        outputs.append(
            PredictionOutput(
                model_name=model_name,
                optimal_threshold_class=int(float(cal_pd) >= threshold),
                prediction_id=prediction_id,
                shap_contributions=shap_contribs,
                reason_codes=reason_codes,
                **enriched,
            )
        )

    return outputs[0] if single else outputs
