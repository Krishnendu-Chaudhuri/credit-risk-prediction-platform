"""Shared constants for the credit risk PD engine."""

from src.config.paths import CREDIT_CSV_PATH, MACRO_XLSX_PATH, MODEL_DIR_REL

MAX_BATCH_LOANS = 10_000

# Re-export path string aliases for backward compatibility
MODEL_DIR = MODEL_DIR_REL
