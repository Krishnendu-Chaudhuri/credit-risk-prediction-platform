"""Challenger/champion validation before model promotion."""

from __future__ import annotations

import os
from pathlib import Path

from src.utils import load_json


class PromotionBlockedError(RuntimeError):
    """Raised when challenger underperforms champion beyond configured thresholds."""


def _best_model_metrics(metrics: dict, metadata: dict) -> dict:
    best = metadata.get("best_model", "xgb")
    return metrics.get(best, metrics.get("xgb", {}))


def validate_promotion(
    challenger_dir: Path,
    champion_dir: Path | None,
    *,
    force: bool = False,
) -> None:
    if force or champion_dir is None or not champion_dir.exists():
        return

    challenger_metrics_path = challenger_dir / "metrics.json"
    champion_metrics_path = champion_dir / "metrics.json"
    if not challenger_metrics_path.exists() or not champion_metrics_path.exists():
        return

    challenger_meta = load_json(challenger_dir / "training_metadata.json")
    champion_meta = load_json(champion_dir / "training_metadata.json")
    challenger = _best_model_metrics(load_json(challenger_metrics_path), challenger_meta)
    champion = _best_model_metrics(load_json(champion_metrics_path), champion_meta)

    max_auc_delta = float(os.getenv("PROMOTION_MAX_AUC_DELTA", "-0.01"))
    max_ks_delta = float(os.getenv("PROMOTION_MAX_KS_DELTA", "-0.02"))

    auc_delta = float(challenger.get("roc_auc", 0)) - float(champion.get("roc_auc", 0))
    ks_delta = float(challenger.get("ks", 0)) - float(champion.get("ks", 0))

    if auc_delta < max_auc_delta or ks_delta < max_ks_delta:
        raise PromotionBlockedError(
            f"Challenger underperforms champion: auc_delta={auc_delta:.4f} (min {max_auc_delta}), "
            f"ks_delta={ks_delta:.4f} (min {max_ks_delta})"
        )
