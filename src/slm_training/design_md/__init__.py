"""DESIGN.md lint bridge (@google/design.md).

Python adapter only — the linter kernel lives in Node (`tools/design_md_bridge`).
Uses a persistent REPL when available; results are hashed-cached.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

_BRIDGE_DIR = Path(__file__).resolve().parents[3] / "tools" / "design_md_bridge"
_CLI = _BRIDGE_DIR / "cli.mjs"

_REPL_LOCK = threading.Lock()
_REPL_PROC: subprocess.Popen[str] | None = None
_LINT_CACHE: dict[str, dict[str, Any]] = {}
_LINT_CACHE_MAX = 256
_BASE_LINT_CACHE: dict[str, Any] | None = None


def bridge_available() -> bool:
    if not _CLI.exists():
        return False
    if shutil.which("node") is None:
        return False
    return (_BRIDGE_DIR / "node_modules" / "@google" / "design.md").exists()


def _close_repl() -> None:
    global _REPL_PROC
    proc = _REPL_PROC
    _REPL_PROC = None
    if proc is None:
        return
    try:
        if proc.stdin and proc.poll() is None:
            proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
            proc.stdin.flush()
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


atexit.register(_close_repl)


def _ensure_repl() -> subprocess.Popen[str]:
    global _REPL_PROC
    if _REPL_PROC is not None and _REPL_PROC.poll() is None:
        return _REPL_PROC
    if not bridge_available():
        raise RuntimeError(
            "DESIGN.md bridge unavailable; run: cd tools/design_md_bridge && npm ci"
        )
    _REPL_PROC = subprocess.Popen(
        ["node", str(_CLI), "--repl"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(_BRIDGE_DIR),
    )
    return _REPL_PROC


def _invoke_once(payload: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
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


def _invoke_repl(payload: dict[str, Any]) -> dict[str, Any]:
    with _REPL_LOCK:
        proc = _ensure_repl()
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            _close_repl()
            proc = _ensure_repl()
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            _close_repl()
            raise RuntimeError("design_md bridge REPL died")
        return json.loads(line)


def _run(payload: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    use_repl = os.getenv("DESIGN_MD_BRIDGE_NO_REPL", "").strip() not in {
        "1",
        "true",
        "yes",
    }
    if use_repl:
        try:
            return _invoke_repl(payload)
        except Exception:  # noqa: BLE001
            _close_repl()
            return _invoke_once(payload, timeout=timeout)
    return _invoke_once(payload, timeout=timeout)


def _cache_key(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def lint(source: str) -> dict[str, Any]:
    """Lint a DESIGN.md string. Returns ok/score/summary/findings."""
    key = _cache_key(source)
    cached = _LINT_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    result = _run({"op": "lint", "source": source})
    if len(_LINT_CACHE) >= _LINT_CACHE_MAX:
        # Drop an arbitrary old entry (insertion order).
        _LINT_CACHE.pop(next(iter(_LINT_CACHE)))
    _LINT_CACHE[key] = result
    return dict(result)


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
    global _BASE_LINT_CACHE
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
        # Specialized DESIGN.md files share the same tokens as the fixture base.
        # Lint the base once and reuse the score for specialized variants so
        # corpus builds stay deterministic and fast.
        if _BASE_LINT_CACHE is None:
            _BASE_LINT_CACHE = lint(load_default_design_md())
        report = _BASE_LINT_CACHE
        record.meta["design_lint"] = {
            "score": report.get("score"),
            "summary": report.get("summary"),
            "specialized": True,
        }
        if not report.get("ok") or float(report.get("score") or 0) < min_score:
            return record
    else:
        record.meta["design_lint"] = {
            "score": 1.0,
            "summary": {"errors": 0},
            "offline": True,
            "specialized": True,
        }
    record.design_md = design
    return record
