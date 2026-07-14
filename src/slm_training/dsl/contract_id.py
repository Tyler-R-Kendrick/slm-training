"""Deterministic identity for the full OpenUI language contract."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from slm_training.bridge_utils import repo_root

CONTRACT_MANIFEST_PATH = repo_root() / "grammars" / "openui_contract.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


@lru_cache(maxsize=1)
def _base_manifest() -> dict[str, Any]:
    return json.loads(CONTRACT_MANIFEST_PATH.read_text(encoding="utf-8"))


def contract_manifest(*, tool_schema: Any = None) -> dict[str, Any]:
    """Return every versioned input used to identify a compatible dataset."""
    return {
        **_base_manifest(),
        "tool_schema_sha256": _sha256([] if tool_schema is None else tool_schema),
    }


def compute_contract_id(*, tool_schema: Any = None) -> str:
    manifest = contract_manifest(tool_schema=tool_schema)
    return f"openui-v{manifest['lang_spec_version']}-{_sha256(manifest)}"


OPENUI_CONTRACT_ID = compute_contract_id()

__all__ = [
    "CONTRACT_MANIFEST_PATH",
    "OPENUI_CONTRACT_ID",
    "compute_contract_id",
    "contract_manifest",
]
