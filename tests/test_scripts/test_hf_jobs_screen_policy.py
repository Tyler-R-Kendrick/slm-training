"""Repository policy: automation must never invoke scripts.hf_jobs_screen.

Screening fan-out submits PAID HF Jobs. Spend is strictly opt-in and human:
the only tracked files allowed to reference the screening entrypoint are the
script itself, its tests, its design doc, and the two generated manifests that
list measured or watched paths rather than invocations: the version registry
and the code-quality baseline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED = {
    "scripts/hf_jobs_screen.py",
    "docs/design/hf-jobs-train.md",
    # Generated manifests: each names the file as a measurement subject, which
    # is not a call site. See docs/design/code-quality-contract.md.
    "src/slm_training/resources/versions.json",
    "src/slm_training/resources/code_quality_baseline.json",
}
ALLOWED_PREFIXES = ("tests/test_scripts/test_hf_jobs_screen_",)


def test_no_other_tracked_file_invokes_hf_jobs_screen() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    offenders: list[str] = []
    for rel in tracked:
        if rel in ALLOWED or rel.startswith(ALLOWED_PREFIXES):
            continue
        path = REPO_ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable — cannot invoke a python module
        if "hf_jobs_screen" in text:
            offenders.append(rel)
    assert offenders == [], (
        "automation must never call scripts.hf_jobs_screen (paid HF Jobs "
        f"spend is human opt-in only); found references in: {offenders}"
    )
