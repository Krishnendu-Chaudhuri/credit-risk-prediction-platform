"""SQLite audit store for prediction explanations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from src.config.paths import PROJECT_ROOT
from src.utils import get_env_path

DB_PATH = get_env_path("AUDIT_DB_PATH", PROJECT_ROOT / "app" / "api" / "data" / "audit.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL,
            shap_json TEXT,
            reason_codes_json TEXT,
            user_id TEXT
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "user_id" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN user_id TEXT")
    return conn


def log_prediction(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    shap_contributions: list[dict[str, Any]] | None = None,
    reason_codes: list[str] | None = None,
    prediction_id: str | None = None,
    user_id: str | None = None,
) -> str:
    pid = prediction_id or str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO predictions
            (prediction_id, created_at, inputs_json, outputs_json, shap_json, reason_codes_json, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                datetime.now(UTC).isoformat(),
                json.dumps(inputs),
                json.dumps(outputs),
                json.dumps(shap_contributions or []),
                json.dumps(reason_codes or []),
                user_id,
            ),
        )
        conn.commit()
    return pid


def get_prediction_audit(prediction_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT prediction_id, created_at, inputs_json, outputs_json, shap_json, reason_codes_json, user_id FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "prediction_id": row[0],
        "created_at": row[1],
        "inputs": json.loads(row[2]),
        "outputs": json.loads(row[3]),
        "shap_contributions": json.loads(row[4]),
        "reason_codes": json.loads(row[5]),
        "user_id": row[6],
    }


def recent_inputs(limit: int = 500) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT inputs_json FROM predictions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [json.loads(r[0]) for r in rows]
