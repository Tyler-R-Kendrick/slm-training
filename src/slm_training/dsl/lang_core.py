"""Python adapter over official @openuidev/lang-core (Node bridge)."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm_training.bridge_utils import readline_with_timeout, repo_root
from slm_training.dsl.placeholders import extract_placeholders

REPO_ROOT = repo_root()
DEFAULT_BRIDGE_DIR = REPO_ROOT / "tools" / "openui_bridge"
DEFAULT_CLI = DEFAULT_BRIDGE_DIR / "cli.mjs"

_REPL_LOCK = threading.Lock()
_REPL_PROC: subprocess.Popen[str] | None = None
_RESULT_CACHE: dict[str, Any] = {}
_RESULT_CACHE_MAX = 2048
_CACHE_LOCK = threading.Lock()


def _cache_put(key: str, value: Any) -> None:
    with _CACHE_LOCK:
        if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
            _RESULT_CACHE.pop(next(iter(_RESULT_CACHE)))
        _RESULT_CACHE[key] = value


def _cache_get(key: str) -> Any:
    with _CACHE_LOCK:
        return _RESULT_CACHE.get(key)


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
    state_declarations: dict[str, Any] = field(default_factory=dict)
    query_statements: list[dict[str, Any]] = field(default_factory=list)
    mutation_statements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def statements(self) -> list[Any]:
        # Compatibility shim for older custom-parser callers.
        return []


def _node_bin() -> str:
    return shutil.which("node") or ""


def bridge_available() -> bool:
    node = _node_bin()
    cli = Path(os.getenv("OPENUI_BRIDGE_CLI") or DEFAULT_CLI)
    node_modules = cli.parent / "node_modules" / "@openuidev" / "lang-core"
    return bool(node) and cli.is_file() and node_modules.is_dir()


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
    node = _node_bin()
    cli = Path(os.getenv("OPENUI_BRIDGE_CLI") or DEFAULT_CLI)
    if not node:
        raise RuntimeError(
            "Node.js is required for @openuidev/lang-core bridge. Install Node 20+."
        )
    if not cli.is_file():
        raise RuntimeError(f"OpenUI bridge CLI not found at {cli}")
    if not (cli.parent / "node_modules" / "@openuidev" / "lang-core").is_dir():
        raise RuntimeError(f"Install bridge deps: cd {cli.parent} && npm ci")
    _REPL_PROC = subprocess.Popen(
        [node, str(cli), "--repl"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    return _REPL_PROC


def _invoke_once(payload: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    node = _node_bin()
    if not node:
        raise RuntimeError(
            "Node.js is required for @openuidev/lang-core bridge. Install Node 20+."
        )
    cli = Path(os.getenv("OPENUI_BRIDGE_CLI") or DEFAULT_CLI)
    if not cli.is_file():
        raise RuntimeError(f"OpenUI bridge CLI not found at {cli}")
    if not (cli.parent / "node_modules" / "@openuidev" / "lang-core").is_dir():
        raise RuntimeError(f"Install bridge deps: cd {cli.parent} && npm ci")

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


def _readline_with_timeout(proc: subprocess.Popen[str], timeout_s: float) -> str:
    return readline_with_timeout(
        proc, timeout_s, error_message="OpenUI bridge REPL read failed"
    )


def _invoke_repl(payload: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
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

        line = _readline_with_timeout(proc, timeout_s)
        if not line:
            err = ""
            if proc.stderr is not None:
                try:
                    err = proc.stderr.read()
                except Exception:  # noqa: BLE001
                    err = ""
            _close_repl()
            raise RuntimeError(f"OpenUI bridge REPL died: {err}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenUI bridge non-JSON output: {line[:500]}") from exc


def _invoke(payload: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    """Prefer persistent REPL; fall back to one-shot subprocess."""
    use_repl = os.getenv("OPENUI_BRIDGE_NO_REPL", "").strip() not in {
        "1",
        "true",
        "yes",
    }
    if use_repl:
        try:
            return _invoke_repl(payload, timeout_s=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _close_repl()
            raise RuntimeError(
                f"OpenUI bridge timed out after {timeout_s:.3f}s"
            ) from exc
        except Exception:  # noqa: BLE001
            # Fall back so CI / broken REPL still works.
            _close_repl()
            return _invoke_once(payload, timeout_s=timeout_s)
    return _invoke_once(payload, timeout_s=timeout_s)


def parse(source: str) -> Program:
    """Parse with official lang-core (does not enforce placeholder content policy)."""
    cache_key = "parse:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    result = _invoke({"op": "parse", "source": source})
    if result.get("error"):
        raise ParseError(result["error"])
    placeholders = list(result.get("placeholders") or extract_placeholders(source))
    program = Program(
        source=source,
        root=result.get("root"),
        placeholders=placeholders,
        meta={
            **dict(result.get("meta") or {}),
            "contract_id": result.get("contract_id"),
            "contract_inputs": result.get("contract_inputs"),
        },
        serialized=result.get("serialized"),
        policy_errors=list(result.get("policy_errors") or []),
        state_declarations=dict(result.get("state_declarations") or {}),
        query_statements=list(result.get("query_statements") or []),
        mutation_statements=list(result.get("mutation_statements") or []),
    )
    _cache_put(cache_key, program)
    return program


def validate(source: str) -> Program:
    """Parse + official schema validation + placeholder content policy."""
    cache_key = "validate:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
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
    program = Program(
        source=source,
        root=result.get("root"),
        placeholders=placeholders,
        meta={
            **dict(result.get("meta") or {}),
            "contract_id": result.get("contract_id"),
            "contract_inputs": result.get("contract_inputs"),
        },
        serialized=result.get("serialized"),
        policy_errors=policy,
        state_declarations=dict(result.get("state_declarations") or {}),
        query_statements=list(result.get("query_statements") or []),
        mutation_statements=list(result.get("mutation_statements") or []),
    )
    _cache_put(cache_key, program)
    return program


def serialize(program: Program) -> str:
    if program.serialized:
        return program.serialized
    if program.root is None:
        raise ParseError("cannot serialize program without root")
    result = _invoke(
        {
            "op": "serialize",
            "root": program.root,
            "source": program.source,
            "state_declarations": program.state_declarations,
        }
    )
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
