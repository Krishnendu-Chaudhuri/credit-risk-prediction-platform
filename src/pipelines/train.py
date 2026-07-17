"""Training pipeline for Logistic Regression and XGBoost PD models."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config.paths import (
    CREDIT_CSV_FILE,
    MACRO_XLSX_FILE,
    OUTPUT_DIR,
    ensure_directory,
    require_file,
)
from src.config.paths import (
    MODEL_DIR as DEFAULT_MODEL_DIR,
)
from src.data.integrity import compute_manifest, save_data_manifest, verify_manifest
from src.features.engineering import build_feature_matrix, fit_imputation_stats
from src.features.preprocessing import fit_preprocessor
from src.models.calibration import apply_calibrator, fit_calibrator
from src.models.registry import new_version_id, save_version_artifacts
from src.models.schemas import ModelMetrics, TrainConfig, TrainResponse
from src.models.thresholds import optimal_threshold
from src.pipelines.validation.cross_validate import repeated_stratified_cv_metrics
from src.utils import compute_metrics, get_env_path, save_artifacts, set_seed, to_relative_path
from src.utils.macro_regression import fit_macro_multiplier_regression


def _permutation_importance(model, X_test, y_test, feature_names: list[str]) -> dict[str, float]:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    scores = result.importances_mean
    total = scores.sum() or 1.0
    return {name: float(val / total) for name, val in zip(feature_names, scores, strict=False)}


def _get_transformed_feature_names(preprocessor, raw_features: list[str]) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return raw_features


def train_models(config: TrainConfig | None = None, csv_path: str | None = None) -> TrainResponse:
    config = config or TrainConfig()
    set_seed(config.random_state)

    data_path = Path(csv_path) if csv_path else get_env_path("DATA_CSV_PATH", CREDIT_CSV_FILE)
    model_dir = get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR)
    ensure_directory(model_dir)
    ensure_directory(OUTPUT_DIR)
    macro_path = get_env_path("MACRO_XLSX_PATH", MACRO_XLSX_FILE)

    require_file(data_path)
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["loan_status"],
    )

    imputation_stats = fit_imputation_stats(train_df)
    X_train, y_train = build_feature_matrix(train_df, imputation_stats)
    X_test, y_test = build_feature_matrix(test_df, imputation_stats)
    feature_cols = list(X_train.columns)

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.15, random_state=config.random_state, stratify=y_train
    )

    preprocessor = fit_preprocessor(X_tr)
    X_tr_t = preprocessor.transform(X_tr)
    X_cal_t = preprocessor.transform(X_cal)
    X_test_t = preprocessor.transform(X_test)

    pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)

    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)
    lr_model.fit(X_tr_t, y_tr)
    lr_cal = fit_calibrator(y_cal.values, lr_model.predict_proba(X_cal_t)[:, 1], method="platt")
    lr_prob = apply_calibrator(lr_cal, lr_model.predict_proba(X_test_t)[:, 1])
    lr_threshold = optimal_threshold(y_test.values, lr_prob, X_test["loan_amnt"].values)
    lr_metrics = compute_metrics(y_test.values, lr_prob, threshold=lr_threshold)

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=config.random_state,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr_t, y_tr)
    xgb_cal = fit_calibrator(y_cal.values, xgb_model.predict_proba(X_cal_t)[:, 1], method="isotonic")
    xgb_prob = apply_calibrator(xgb_cal, xgb_model.predict_proba(X_test_t)[:, 1])
    xgb_threshold = optimal_threshold(y_test.values, xgb_prob, X_test["loan_amnt"].values)
    xgb_metrics = compute_metrics(y_test.values, xgb_prob, threshold=xgb_threshold)

    cv_metrics: dict[str, Any] = {}
    if len(X_train) >= 50:

        def _cv_fold(train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, float]:
            fold_pre = fit_preprocessor(X_train.iloc[train_idx])
            X_tr_f = fold_pre.transform(X_train.iloc[train_idx])
            X_te_f = fold_pre.transform(X_train.iloc[test_idx])
            y_tr_f = y_train.iloc[train_idx].values
            y_te_f = y_train.iloc[test_idx].values
            fold_model = XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=config.random_state,
                n_jobs=-1,
            )
            fold_model.fit(X_tr_f, y_tr_f)
            prob = fold_model.predict_proba(X_te_f)[:, 1]
            return compute_metrics(y_te_f, prob)

        cv_metrics = repeated_stratified_cv_metrics(
            X_train.values,
            y_train.values,
            _cv_fold,
            n_splits=5,
            n_repeats=3,
        )

    oot_metrics: dict[str, Any] | None = None
    if os.getenv("OOT_VALIDATION") == "row_order_proxy":
        split_idx = int(len(df) * 0.8)
        oot_df = df.iloc[split_idx:].copy()
        train_oot_df = df.iloc[:split_idx].copy()
        oot_stats = fit_imputation_stats(train_oot_df)
        X_oot, y_oot = build_feature_matrix(oot_df, oot_stats)
        X_tr_oot, y_tr_oot = build_feature_matrix(train_oot_df, oot_stats)
        oot_pre = fit_preprocessor(X_tr_oot)
        oot_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=config.random_state,
            n_jobs=-1,
        )
        oot_model.fit(oot_pre.transform(X_tr_oot), y_tr_oot)
        oot_prob = oot_model.predict_proba(oot_pre.transform(X_oot))[:, 1]
        oot_metrics = {
            "xgb": compute_metrics(y_oot.values, oot_prob),
            "caveat": "row_order_proxy — no origination_date in benchmark CSV",
        }

    transformed_names = _get_transformed_feature_names(preprocessor, feature_cols)
    lr_importance = _permutation_importance(lr_model, X_test_t, y_test, transformed_names[: len(lr_model.coef_[0])])
    xgb_importance = _permutation_importance(
        xgb_model, X_test_t, y_test, transformed_names[: len(xgb_model.feature_importances_)]
    )

    best_model = "xgb" if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "lr"
    best_importance = xgb_importance if best_model == "xgb" else lr_importance

    macro_reg = None
    try:
        if macro_path.exists():
            require_file(macro_path, label="Macro data file")
            normal = pd.read_excel(macro_path, sheet_name="Normal")
            macro_dict = normal.set_index("variable")["base_value"].to_dict()
            macro_X = pd.DataFrame([macro_dict] * len(train_df))
            macro_reg = fit_macro_multiplier_regression(train_df["loan_status"], macro_X)
    except Exception:
        macro_reg = None

    data_manifest = compute_manifest([data_path, macro_path] if macro_path.exists() else [data_path])

    metadata: dict[str, Any] = {
        "trained_at": datetime.now(UTC).isoformat(),
        "best_model": best_model,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": feature_cols,
        "csv_path": to_relative_path(data_path),
        "imputation_stats": imputation_stats,
        "data_manifest": data_manifest,
        "optimal_threshold": {"lr": lr_threshold, "xgb": xgb_threshold},
        "cv_metrics": cv_metrics or None,
        "oot_metrics": oot_metrics,
        "macro_regression": (
            {"spec": macro_reg.spec, "r2": macro_reg.r2, "coefficients": macro_reg.coefficients} if macro_reg else None
        ),
    }

    artifacts = {
        "preprocessor.joblib": preprocessor,
        "lr_model.joblib": lr_model,
        "xgb_model.joblib": xgb_model,
        "calibrator_lr.joblib": lr_cal,
        "calibrator_xgb.joblib": xgb_cal,
        "training_metadata.json": metadata,
        "metrics.json": {"lr": lr_metrics, "xgb": xgb_metrics, "cv": cv_metrics, "oot": oot_metrics},
        "feature_importance.json": {"lr": lr_importance, "xgb": xgb_importance},
        "data_manifest.json": data_manifest,
    }
    save_artifacts(model_dir, artifacts)

    version_id = new_version_id()
    metadata["version_id"] = version_id
    artifacts["training_metadata.json"] = metadata
    save_version_artifacts(version_id, artifacts, model_dir)
    save_data_manifest(model_dir, data_manifest)
    verify_manifest(data_manifest, [data_path])
    return TrainResponse(
        best_model=best_model,
        metrics={
            "lr": ModelMetrics(model_name="lr", **lr_metrics),
            "xgb": ModelMetrics(model_name="xgb", **xgb_metrics),
        },
        feature_importance=best_importance,
        training_metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk PD models")
    parser.add_argument("--csv", default=None, help="Path to training CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_models(
        TrainConfig(test_size=args.test_size, random_state=args.seed),
        csv_path=args.csv,
    )
    print(f"Best model: {result.best_model}")
    for name, metrics in result.metrics.items():
        print(f"{name}: ROC AUC={metrics.roc_auc:.4f}, KS={metrics.ks:.4f}, Gini={metrics.gini:.4f}")


if __name__ == "__main__":
    main()
