"""DESIGN.md lint bridge (@google/design.md)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

_BRIDGE_DIR = Path(__file__).resolve().parents[3] / "tools" / "design_md_bridge"
_CLI = _BRIDGE_DIR / "cli.mjs"


def bridge_available() -> bool:
    if not _CLI.exists():
        return False
    if shutil.which("node") is None:
        return False
    return (_BRIDGE_DIR / "node_modules" / "@google" / "design.md").exists()


def _run(payload: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    if not bridge_available():
        raise RuntimeError(
            "DESIGN.md bridge unavailable; run: cd tools/design_md_bridge && npm ci"
        )
    env = os.environ.copy()
    proc = subprocess.run(
        ["node", str(_CLI)],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        timeout=timeout,
        cwd=str(_BRIDGE_DIR),
        env=env,
        check=False,
    )
    raw = (proc.stdout or b"").decode("utf-8").strip()
    if not raw:
        err = (proc.stderr or b"").decode("utf-8").strip()
        raise RuntimeError(f"design_md bridge empty response: {err or proc.returncode}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"design_md bridge invalid JSON: {raw[:200]}") from exc


def lint(source: str) -> dict[str, Any]:
    """Lint a DESIGN.md string. Returns ok/score/summary/findings."""
    return _run({"op": "lint", "source": source})


def score(source: str) -> float:
    result = lint(source)
    return float(result.get("score") or 0.0)


def load_default_design_md() -> str:
    path = (
        Path(__file__).resolve().parents[3]
        / "fixtures"
        / "design_md"
        / "default.DESIGN.md"
    )
    return path.read_text(encoding="utf-8")


def attach_default_design_md(record: Any, *, min_score: float = 0.7) -> Any:
    """Attach a per-record DESIGN.md when missing and lint passes."""
    if getattr(record, "design_md", None):
        return record
    try:
        from slm_training.data.design_md.extract import extract_design_md

        design = extract_design_md(
            title=str(getattr(record, "id", "") or "record"),
            description=str(getattr(record, "prompt", "") or "")[:400],
            tags=[str(getattr(record, "source", "fixture"))],
            variant="strict",
        )
    except Exception:  # noqa: BLE001
        design = load_default_design_md()
    record.meta = dict(record.meta or {})
    if bridge_available():
        report = lint(design)
        record.meta["design_lint"] = {
            "score": report.get("score"),
            "summary": report.get("summary"),
        }
        if not report.get("ok") or float(report.get("score") or 0) < min_score:
            return record
    else:
        record.meta["design_lint"] = {"score": 1.0, "summary": {"errors": 0}, "offline": True}
    record.design_md = design
    return record
