"""Playwright evidence adapter for the bundled OpenUI preview."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.bridge_utils import repo_root


@dataclass(frozen=True)
class RuntimeEvidence:
    rendered: bool
    console_errors: tuple[str, ...] = ()
    behavior_errors: tuple[str, ...] = ()
    interaction_trace: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeEvidence:
        return cls(
            rendered=bool(data.get("rendered")),
            console_errors=tuple(str(x) for x in data.get("console_errors") or ()),
            behavior_errors=tuple(str(x) for x in data.get("behavior_errors") or ()),
            interaction_trace=tuple(
                str(x) for x in data.get("interaction_trace") or ()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rendered": self.rendered,
            "console_errors": list(self.console_errors),
            "behavior_errors": list(self.behavior_errors),
            "interaction_trace": list(self.interaction_trace),
        }


def run_preview_verifier(
    source: str,
    *,
    seed_console_error: bool = False,
    seed_behavior_error: bool = False,
    timeout_s: float = 30.0,
    root: Path | None = None,
) -> RuntimeEvidence:
    """Render one program in Chromium and return runtime/interaction evidence."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required for the OpenUI preview verifier")
    root = root or repo_root()
    script = root / "src" / "apps" / "openui_preview" / "verify.mjs"
    if not script.is_file():
        raise RuntimeError(f"OpenUI preview verifier not found: {script}")
    payload = {
        "source": source,
        "seed_console_error": seed_console_error,
        "seed_behavior_error": seed_behavior_error,
    }
    proc = subprocess.run(
        [node, str(script)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=root,
        timeout=timeout_s,
        check=False,
    )
    raw = (proc.stdout or "").strip()
    if proc.returncode or not raw:
        detail = (proc.stderr or raw or f"exit {proc.returncode}").strip()
        raise RuntimeError(f"OpenUI preview verifier failed: {detail[:500]}")
    try:
        return RuntimeEvidence.from_dict(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenUI preview verifier returned invalid JSON: {raw[:500]}") from exc
