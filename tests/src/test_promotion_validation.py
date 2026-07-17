"""Tests for challenger/champion promotion validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.models.promotion import PromotionBlockedError, validate_promotion
from src.models.registry import ModelStatus, get_current_version, promote_version, save_version_artifacts, version_dir
from src.utils import save_json


def _stub_metrics(roc_auc: float, ks: float) -> dict:
    return {
        "lr": {"roc_auc": roc_auc, "ks": ks},
        "xgb": {"roc_auc": roc_auc, "ks": ks},
        "cv": {},
        "oot": None,
    }


def _save_version(model_dir: Path, version_id: str, roc_auc: float, ks: float) -> None:
    save_version_artifacts(
        version_id,
        {
            "preprocessor.joblib": {"stub": True},
            "lr_model.joblib": {"stub": True},
            "xgb_model.joblib": {"stub": True},
            "training_metadata.json": {"best_model": "xgb"},
            "metrics.json": _stub_metrics(roc_auc, ks),
        },
        model_dir,
    )


def test_promote_allowed_when_challenger_beats_champion(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    champion_id = "20250101T120000Z"
    challenger_id = "20250102T120000Z"
    _save_version(model_dir, champion_id, roc_auc=0.70, ks=0.30)
    _save_version(model_dir, challenger_id, roc_auc=0.75, ks=0.35)
    save_json(
        version_dir(champion_id, model_dir) / "version.json", {"version_id": champion_id, "status": ModelStatus.current}
    )
    save_json(model_dir / "current.json", {"version_id": champion_id, "status": ModelStatus.current})

    promote_version(challenger_id, model_dir)
    assert get_current_version(model_dir) == challenger_id


def test_promote_blocked_when_challenger_underperforms(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    champion_id = "20250101T120000Z"
    challenger_id = "20250102T120000Z"
    _save_version(model_dir, champion_id, roc_auc=0.80, ks=0.40)
    _save_version(model_dir, challenger_id, roc_auc=0.70, ks=0.20)
    save_json(
        version_dir(champion_id, model_dir) / "version.json", {"version_id": champion_id, "status": ModelStatus.current}
    )
    save_json(model_dir / "current.json", {"version_id": champion_id, "status": ModelStatus.current})
    champion_dir = version_dir(champion_id, model_dir)
    challenger_dir = version_dir(challenger_id, model_dir)

    with pytest.raises(PromotionBlockedError):
        validate_promotion(challenger_dir, champion_dir)

    with pytest.raises(PromotionBlockedError):
        promote_version(challenger_id, model_dir)


def test_promote_force_overrides_validation(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    champion_id = "20250101T120000Z"
    challenger_id = "20250102T120000Z"
    _save_version(model_dir, champion_id, roc_auc=0.80, ks=0.40)
    _save_version(model_dir, challenger_id, roc_auc=0.50, ks=0.10)
    save_json(
        version_dir(champion_id, model_dir) / "version.json", {"version_id": champion_id, "status": ModelStatus.current}
    )
    save_json(model_dir / "current.json", {"version_id": champion_id, "status": ModelStatus.current})

    promote_version(challenger_id, model_dir, force=True)
    assert get_current_version(model_dir) == challenger_id
