"""Shared projection envelope helpers (stdlib JSON only)."""

from __future__ import annotations

import json
from typing import Any, Mapping

from slm_training.evidence_interop.authority import AuthorityRecord
from slm_training.evidence_interop.profile import (
    AUTHORITY_BLOCK_KEY,
    KIND_KEY,
    PROFILE_BLOCK_KEY,
    PROFILE_ID,
    SEMANTIC_AUTHORITY_DEFAULT,
    SEMANTIC_AUTHORITY_KEY,
)


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def base_envelope(
    *,
    kind: str,
    authority: AuthorityRecord,
    format_name: str,
) -> dict[str, Any]:
    """Common slm:* authority block present in every projection format."""
    return {
        PROFILE_BLOCK_KEY: PROFILE_ID,
        KIND_KEY: kind,
        AUTHORITY_BLOCK_KEY: authority.to_dict(),
        SEMANTIC_AUTHORITY_KEY: SEMANTIC_AUTHORITY_DEFAULT,
        "slm:format": format_name,
        "slm:projection_only": True,
    }
