"""Monitor unseen categorical values after OneHotEncoder transform."""

from __future__ import annotations

from threading import Lock

import numpy as np
from app.api.logging_config import get_logger

from src.features.engineering import CATEGORICAL_FEATURES

logger = get_logger(__name__)

_lock = Lock()
_unknown_category_count = 0


def get_unknown_category_count() -> int:
    with _lock:
        return _unknown_category_count


def reset_unknown_category_count() -> None:
    global _unknown_category_count  # noqa: PLW0603
    with _lock:
        _unknown_category_count = 0


def _categorical_feature_names(preprocessor) -> dict[str, list[str]]:
    """Map raw categorical column -> list of transformed column names."""
    cat_transformer = preprocessor.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]
    try:
        names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception:
        return {}

    mapping: dict[str, list[str]] = {col: [] for col in CATEGORICAL_FEATURES}
    for name in names:
        for col in CATEGORICAL_FEATURES:
            if name.startswith(f"{col}_") or name == col:
                mapping[col].append(name)
                break
    return mapping


def check_unknown_categories(
    transformed: np.ndarray,
    preprocessor,
    raw_frame,
) -> int:
    """Detect rows where a categorical feature has all-zero one-hot block."""
    global _unknown_category_count  # noqa: PLW0603

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return 0

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    mapping = _categorical_feature_names(preprocessor)
    new_unknown = 0

    for col in CATEGORICAL_FEATURES:
        if col not in raw_frame.columns:
            continue
        cols = [name_to_idx[n] for n in mapping.get(col, []) if n in name_to_idx]
        if not cols:
            continue
        blocks = transformed[:, cols]
        unknown_rows = np.all(blocks == 0, axis=1)
        count = int(unknown_rows.sum())
        if count:
            logger.warning("Unseen category detected for %s in %d row(s)", col, count)
            new_unknown += count

    if new_unknown:
        with _lock:
            _unknown_category_count += new_unknown
    return new_unknown
