"""Reproducible OpenUI dataset contract identity (``contract_id``).

Every training / eval record is produced against a specific *language contract*:
the OpenUI Lang spec version, the parser, the component + tool schemas, the
canonicalizer, the renderer, and the DSL tokenizer. If **any** of these change,
records built before and after are no longer comparable — a component-library
bump is a new dataset version. ``contract_id`` is a stable hash of those seven
inputs, stamped into ``ExampleRecord.meta['contract_id']`` so splits / evals
never silently mix incompatible contracts::

    contract_id = hash(lang_spec_version + parser_commit + component_schema
                       + tool_schema + canonicalizer + renderer
                       + tokenizer_version)

The value is derived entirely from static repo artifacts (the bridge
``package.json``, the vendored grammars, and in-source version constants) so it
is deterministic and requires **neither** the Node bridge **nor** torch to be
installed — the same tree always yields the same id.

**Scope:** OpenUI Lang 0.2.x (the layout subset). The v0.5 upgrade
(``state`` / ``query`` / ``mutation`` / ``action`` / ``tool`` constructs) is
pending an upstream ``@openuidev/lang-core`` release, so ``tool_schema`` is
``none`` in this contract. See ``docs/design/openui-contract-id.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from slm_training.bridge_utils import repo_root

# Bump when the canonicalizer (``strip_style_literals`` + lang-core serialize +
# production codec) changes in a way that alters the bytes emitted for a program.
CANONICALIZER_VERSION = 1

_REPO = repo_root()
_BRIDGE_PKG = _REPO / "tools" / "openui_bridge" / "package.json"
_LARK_GRAMMAR = _REPO / "grammars" / "openui.lark"
_PROP_ORDER = _REPO / "grammars" / "openui_prop_order.json"
_DSL_TOKENIZER = _REPO / "src" / "slm_training" / "models" / "dsl_tokenizer.py"


def _sha12(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _file_sha12(path: Path) -> str:
    try:
        return _sha12(path.read_bytes())
    except OSError:
        return "missing"


def _clean_semver(spec: str) -> str:
    """Strip npm range operators (``^`` / ``~`` / ``>=`` …) to a bare version."""
    return spec.lstrip("^~>=< ").strip() or "unknown"


def _read_int_const(path: Path, name: str, default: int) -> int:
    """Read ``NAME = <int>`` from a source file without importing it.

    Importing ``models.dsl_tokenizer`` would run ``models/__init__`` and pull in
    torch; the contract id must not depend on optional heavy deps, so we read the
    single source of truth (the constant) straight from the file instead.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return default
    match = re.search(rf"^{re.escape(name)}\s*=\s*(\d+)", text, re.MULTILINE)
    return int(match.group(1)) if match else default


@lru_cache(maxsize=1)
def _bridge_versions() -> dict[str, str]:
    try:
        deps = json.loads(_BRIDGE_PKG.read_text(encoding="utf-8")).get(
            "dependencies", {}
        )
    except (OSError, json.JSONDecodeError):
        deps = {}
    return {
        "lang_core": _clean_semver(deps.get("@openuidev/lang-core", "unknown")),
        "react_ui": _clean_semver(deps.get("@openuidev/react-ui", "unknown")),
        "react_lang": _clean_semver(deps.get("@openuidev/react-lang", "unknown")),
    }


@lru_cache(maxsize=1)
def contract_components() -> dict[str, str]:
    """The seven contract inputs, each a stable, human-readable string."""
    versions = _bridge_versions()
    lang_core = versions["lang_core"]
    react_ui = versions["react_ui"]
    react_lang = versions["react_lang"]
    tok_version = _read_int_const(_DSL_TOKENIZER, "DSL_TOKENIZER_VERSION", 0)
    return {
        "lang_spec_version": lang_core,
        # Parser identity: official lang-core version + vendored Lark grammar.
        "parser_commit": f"lang-core@{lang_core}+lark:{_file_sha12(_LARK_GRAMMAR)}",
        # Component schema proxy: renderer lib version + positional-arg order map.
        "component_schema": (
            f"react-ui@{react_ui}+prop_order:{_file_sha12(_PROP_ORDER)}"
        ),
        # 0.2.x layout subset has no tool constructs (bridge sets toolCalls=false).
        "tool_schema": "none@0.2.x",
        "canonicalizer": (
            f"strip_style_literals+lang-core-serialize@v{CANONICALIZER_VERSION}"
        ),
        "renderer": f"react-ui@{react_ui}+react-lang@{react_lang}",
        "tokenizer_version": f"dsl-tok@v{tok_version}",
    }


@lru_cache(maxsize=1)
def contract_id() -> str:
    """Stable id hashing all seven contract inputs (``oc-<16 hex>``)."""
    payload = json.dumps(
        contract_components(), sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return "oc-" + digest[:16]


def contract_fingerprint() -> dict[str, object]:
    """``{"contract_id": ..., "components": {...}}`` for build manifests."""
    return {"contract_id": contract_id(), "components": contract_components()}


def stamp(meta: dict | None = None) -> dict:
    """Return a copy of ``meta`` with ``contract_id`` stamped in."""
    out = dict(meta or {})
    out["contract_id"] = contract_id()
    return out


__all__ = [
    "CANONICALIZER_VERSION",
    "contract_components",
    "contract_fingerprint",
    "contract_id",
    "stamp",
]
