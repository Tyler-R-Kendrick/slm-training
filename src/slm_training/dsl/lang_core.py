"""Python adapter over official @openuidev/lang-core (Node bridge)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm_training.dsl.placeholders import extract_placeholders

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BRIDGE_DIR = REPO_ROOT / "tools" / "openui_bridge"
DEFAULT_CLI = DEFAULT_BRIDGE_DIR / "cli.mjs"


class ParseError(ValueError):
    """Raised when OpenUI source fails official parse or content policy."""


@dataclass
class Program:
    """Normalized parse result compatible with harness callers."""

    source: str
    root: dict[str, Any] | None = None
    placeholders: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    serialized: str | None = None
    policy_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def statements(self) -> list[Any]:
        # Compatibility shim for older custom-parser callers.
        return []


def _node_bin() -> str:
    return shutil.which("node") or ""


def bridge_available() -> bool:
    node = _node_bin()
    cli = Path(os.getenv("OPENUI_BRIDGE_CLI") or DEFAULT_CLI)
    node_modules = (cli.parent / "node_modules" / "@openuidev" / "lang-core")
    return bool(node) and cli.is_file() and node_modules.is_dir()


def _invoke(payload: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    node = _node_bin()
    if not node:
        raise RuntimeError(
            "Node.js is required for @openuidev/lang-core bridge. Install Node 20+."
        )
    cli = Path(os.getenv("OPENUI_BRIDGE_CLI") or DEFAULT_CLI)
    if not cli.is_file():
        raise RuntimeError(f"OpenUI bridge CLI not found at {cli}")
    if not (cli.parent / "node_modules" / "@openuidev" / "lang-core").is_dir():
        raise RuntimeError(
            f"Install bridge deps: cd {cli.parent} && npm ci"
        )

    proc = subprocess.run(
        [node, str(cli)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise RuntimeError(
            f"OpenUI bridge returned empty output (exit={proc.returncode}): {proc.stderr}"
        )
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenUI bridge non-JSON output: {stdout[:500]}") from exc
    if proc.returncode not in (0, 2) and not result.get("ok", False):
        raise RuntimeError(result.get("error") or proc.stderr or "bridge failed")
    return result


def parse(source: str) -> Program:
    """Parse with official lang-core (does not enforce placeholder content policy)."""
    result = _invoke({"op": "parse", "source": source})
    if result.get("error"):
        raise ParseError(result["error"])
    placeholders = list(result.get("placeholders") or extract_placeholders(source))
    return Program(
        source=source,
        root=result.get("root"),
        placeholders=placeholders,
        meta=dict(result.get("meta") or {}),
        serialized=result.get("serialized"),
        policy_errors=list(result.get("policy_errors") or []),
    )


def validate(source: str) -> Program:
    """Parse + official schema validation + placeholder content policy."""
    result = _invoke({"op": "validate", "source": source})
    if result.get("error"):
        raise ParseError(result["error"])
    policy = list(result.get("policy_errors") or [])
    meta_errors = list((result.get("meta") or {}).get("errors") or [])
    if not result.get("ok"):
        messages = [e.get("message") for e in policy if e.get("message")]
        messages += [str(e) for e in meta_errors]
        raise ParseError("; ".join(messages) or "OpenUI validation failed")
    placeholders = list(result.get("placeholders") or extract_placeholders(source))
    return Program(
        source=source,
        root=result.get("root"),
        placeholders=placeholders,
        meta=dict(result.get("meta") or {}),
        serialized=result.get("serialized"),
        policy_errors=policy,
    )


def serialize(program: Program) -> str:
    if program.serialized:
        return program.serialized
    if program.root is None:
        raise ParseError("cannot serialize program without root")
    result = _invoke({"op": "serialize", "root": program.root})
    if not result.get("ok"):
        raise ParseError(result.get("error") or "serialize failed")
    return str(result["source"])


def generate_system_prompt(**options: Any) -> str:
    """Official library.prompt() for training / teacher synthesis."""
    result = _invoke({"op": "prompt", "options": options or {}})
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "prompt generation failed")
    return str(result["prompt"])


def stream_check(source: str) -> dict[str, Any]:
    """Incremental/partial parse via official createStreamingParser."""
    return _invoke({"op": "stream_check", "source": source})


def library_schema() -> dict[str, Any]:
    result = _invoke({"op": "schema"})
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "schema failed")
    return dict(result.get("schema") or {})
