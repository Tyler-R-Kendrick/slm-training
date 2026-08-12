"""RESEARCH-15 / SLM-569: runner script tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_runner_default_off_json() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.run_research_15_anytime_valid_promotion", "--json"],
        cwd=ROOT,
        env={**dict(**{}), "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "skipped_default_off" in proc.stdout
