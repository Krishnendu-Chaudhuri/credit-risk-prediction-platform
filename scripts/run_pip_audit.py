#!/usr/bin/env python3
"""Run pip-audit with optional vulnerability allowlist."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "security" / "pip-audit-allowlist.json"


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pip_audit",
        "-r",
        str(ROOT / "requirements.txt"),
        "--desc",
        "on",
    ]
    if ALLOWLIST.exists():
        data = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
        for vuln_id in data.get("ignored_vulnerabilities", []):
            cmd.extend(["--ignore-vuln", vuln_id])
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
