"""Unit tests for data integrity manifest."""

from __future__ import annotations

from pathlib import Path

from src.data.integrity import compute_manifest, file_sha256, verify_manifest


def test_file_sha256_stable(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    h1 = file_sha256(p)
    h2 = file_sha256(p)
    assert h1["sha256"] == h2["sha256"]


def test_verify_manifest_detects_drift(tmp_path: Path, caplog) -> None:
    p = tmp_path / "data.csv"
    p.write_text("original", encoding="utf-8")
    manifest = compute_manifest([p])
    p.write_text("tampered", encoding="utf-8")
    warnings = verify_manifest(manifest, [p])
    assert any("drift" in w.lower() for w in warnings)
