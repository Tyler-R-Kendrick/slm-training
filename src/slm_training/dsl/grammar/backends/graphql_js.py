"""GraphQL grammar backend — graphql-js as the validity oracle (F2, SLM-43).

Mirrors the OpenUI lang-core sidecar pattern: a Node CLI
(`src/apps/graphql_bridge/cli.mjs`) wraps the official `graphql` package for
parse / schema-validate / canonical print. The introspection schema is the
scope/symbol context: `validate` enforces that every selected field exists on
the schema type it is selected from (the F2 scope rule), which OpenUI 0.2.x's
small binding surface cannot exercise.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from slm_training.dsl.lang_core import ParseError, Program
from slm_training.dsl.grammar.backends.types import GrammarInfo, REPO_ROOT
from slm_training.dsl.stream_types import StreamStatus

BRIDGE_DIR = REPO_ROOT / "src" / "apps" / "graphql_bridge"
BRIDGE_CLI = BRIDGE_DIR / "cli.mjs"
DEFAULT_SCHEMA_PATH = (
    REPO_ROOT / "src" / "slm_training" / "resources" / "graphql" / "demo_schema.graphql"
)


def _node_bin() -> str:
    return shutil.which("node") or ""


def bridge_available() -> bool:
    return bool(
        _node_bin()
        and BRIDGE_CLI.is_file()
        and (BRIDGE_DIR / "node_modules" / "graphql").is_dir()
    )


def _sanitized_env() -> dict[str, str]:
    # Session environments may inject NODE_OPTIONS entries (e.g. --import tsx)
    # that this Node build rejects with exit 9, silently killing the bridge.
    env = dict(os.environ)
    env["NODE_OPTIONS"] = ""
    return env


def invoke_bridge(payload: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    node = _node_bin()
    if not node:
        raise RuntimeError("Node.js is required for the graphql-js bridge.")
    if not BRIDGE_CLI.is_file():
        raise RuntimeError(f"GraphQL bridge CLI not found at {BRIDGE_CLI}")
    if not (BRIDGE_DIR / "node_modules" / "graphql").is_dir():
        raise RuntimeError(f"Install bridge deps: cd {BRIDGE_DIR} && npm ci")
    proc = subprocess.run(
        [node, str(BRIDGE_CLI)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env=_sanitized_env(),
    )
    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise RuntimeError(
            f"GraphQL bridge returned empty output (exit={proc.returncode}): {proc.stderr}"
        )
    return json.loads(stdout)


@lru_cache(maxsize=1)
def default_schema_sdl() -> str:
    return DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8")


def schema_symbols(schema_sdl: str | None = None) -> dict[str, list[str]]:
    """{type name: [field names]} — the schema-as-symbol-table view."""
    result = invoke_bridge(
        {"op": "schema_symbols", "schema_sdl": schema_sdl or default_schema_sdl()}
    )
    if not result.get("ok"):
        raise ParseError("; ".join(result.get("errors", ["schema error"])))
    return {str(k): [str(f) for f in v] for k, v in result.get("types", {}).items()}


class GraphQLJsBackend:
    """GrammarBackend over the graphql-js sidecar."""

    def __init__(self, *, schema_path: Path | None = None) -> None:
        self._schema_path = schema_path or DEFAULT_SCHEMA_PATH

    @property
    def info(self) -> GrammarInfo:
        return GrammarInfo(
            id="graphql",
            kind="graphql-js",
            description="GraphQL via the official graphql-js parser/validator",
            grammar_path=self._schema_path,
            root_component=None,
        )

    def available(self) -> bool:
        return bridge_available() and self._schema_path.is_file()

    def _schema_sdl(self) -> str:
        if self._schema_path == DEFAULT_SCHEMA_PATH:
            return default_schema_sdl()
        return self._schema_path.read_text(encoding="utf-8")

    def parse(self, source: str) -> Program:
        result = invoke_bridge({"op": "parse", "source": source})
        if not result.get("ok"):
            raise ParseError("; ".join(result.get("errors", ["parse error"])))
        return Program(
            source=source,
            root=None,
            placeholders=extract_variables(source),
            meta={"backend": "graphql", "kind": "graphql-js"},
            serialized=str(result.get("canonical") or source.strip()),
        )

    def validate(self, source: str) -> Program:
        result = invoke_bridge(
            {"op": "validate", "source": source, "schema_sdl": self._schema_sdl()}
        )
        if not result.get("ok"):
            raise ParseError("; ".join(result.get("errors", ["validation error"])))
        return Program(
            source=source,
            root=None,
            placeholders=extract_variables(source),
            meta={"backend": "graphql", "kind": "graphql-js", "schema_checked": True},
            serialized=str(result.get("canonical") or source.strip()),
        )

    def serialize(self, program: Program) -> str:
        return (program.serialized or program.source).strip()

    def stream_check(self, source: str) -> StreamStatus:
        try:
            result = invoke_bridge({"op": "parse", "source": source})
        except Exception as exc:  # noqa: BLE001
            return StreamStatus(
                ok=False,
                incomplete=False,
                has_root=False,
                error_codes=("bridge-error",),
                unresolved=(str(exc),),
            )
        if result.get("ok"):
            return StreamStatus(
                ok=True,
                incomplete=False,
                has_root=True,
                error_codes=(),
                unresolved=(),
                serialized=str(result.get("canonical") or "") or None,
            )
        errors = [str(e) for e in result.get("errors", [])]
        # graphql-js reports truncated input as '... found <EOF>.' — that is
        # an incomplete prefix, not a hard error.
        incomplete = any("<EOF>" in e for e in errors)
        return StreamStatus(
            ok=False,
            incomplete=incomplete,
            has_root=False,
            error_codes=() if incomplete else ("syntax-error",),
            unresolved=tuple(errors),
        )

    def structural_tokens(self) -> frozenset[str]:
        return frozenset(
            {"{", "}", "(", ")", ":", ",", "...", "$", "@", "query", "mutation", "fragment", "on"}
        )

    def component_names(self) -> frozenset[str]:
        try:
            return frozenset(schema_symbols(self._schema_sdl()))
        except Exception:  # noqa: BLE001
            return frozenset()

    def content_props(self) -> frozenset[str]:
        # Variables ($name) are the routed-content channel; no copy props.
        return frozenset()

    def library_schema(self) -> dict[str, Any]:
        try:
            return {"types": schema_symbols(self._schema_sdl())}
        except Exception:  # noqa: BLE001
            return {"types": {}}

    def generate_system_prompt(self, **options: Any) -> str:  # noqa: ARG002
        return (
            "Emit a single GraphQL operation that validates against the "
            "provided schema. Select only fields that exist on the schema "
            "types; pass values through variables ($name), never literals."
        )


def extract_variables(source: str) -> list[str]:
    """GraphQL's routed-content channel: ``$variable`` references, in order."""
    import re

    seen: list[str] = []
    for match in re.finditer(r"\$([A-Za-z_][A-Za-z0-9_]*)", source):
        name = f"${match.group(1)}"
        if name not in seen:
            seen.append(name)
    return seen
