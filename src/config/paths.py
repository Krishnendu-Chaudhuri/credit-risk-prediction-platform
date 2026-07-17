"""Central project-relative path configuration."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "assets" / "data"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

CREDIT_CSV_FILE = DATA_DIR / "credit_risk_dataset_new.csv"
MACRO_XLSX_FILE = DATA_DIR / "US_Macro_Economic_Stress_Test_Data.xlsx"

# String aliases for metadata/env backward compatibility
CREDIT_CSV_PATH = CREDIT_CSV_FILE.relative_to(PROJECT_ROOT).as_posix()
MACRO_XLSX_PATH = MACRO_XLSX_FILE.relative_to(PROJECT_ROOT).as_posix()
MODEL_DIR_REL = MODEL_DIR.relative_to(PROJECT_ROOT).as_posix()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, *, label: str = "Dataset") -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        candidate = default
    else:
        candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
