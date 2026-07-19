"""Versioned grammar/profile family manifest for CAP5-01 (SLM-100).

This module introduces a deterministic, serializable ``GrammarProfile`` record and
a versioned manifest builder. It is intentionally plan-only / train-free: it
consumes existing grammar decision traces or decision-difficulty records and
emits an immutable summary that later slices can turn into ladder cells,
cutoff-sensitivity neighbors, or curriculum bands.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from slm_training.dsl.analysis.arity import DecisionDifficulty

MANIFEST_SCHEMA = "grammar_profile/v1"


def _stable_hash(parts: Mapping[str, Any]) -> str:
    """Return a stable 16-hex hash of a JSON-sortable mapping."""
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class GrammarProfile:
    """One point in a grammar/profile family.

    A profile summarizes the structural/decision complexity of a grammar or a
    subset of its traces. It stores exact counts where available and ``None``
    where evidence is missing, so incomplete profiles are honest rather than
    silently interpolated.
    """

    profile_id: str
    signature: str
    decision_count: int
    mean_arity: float
    max_arity: int
    mean_entropy_bits: float | None
    max_entropy_bits: float | None
    source_hash: str | None = None
    schema_version: str = MANIFEST_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "signature": self.signature,
            "decision_count": self.decision_count,
            "mean_arity": self.mean_arity,
            "max_arity": self.max_arity,
            "mean_entropy_bits": self.mean_entropy_bits,
            "max_entropy_bits": self.max_entropy_bits,
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GrammarProfile":
        return cls(
            profile_id=str(data["profile_id"]),
            signature=str(data["signature"]),
            decision_count=int(data["decision_count"]),
            mean_arity=float(data["mean_arity"]),
            max_arity=int(data["max_arity"]),
            mean_entropy_bits=data.get("mean_entropy_bits"),
            max_entropy_bits=data.get("max_entropy_bits"),
            source_hash=data.get("source_hash"),
            schema_version=str(data.get("schema_version", MANIFEST_SCHEMA)),
        )


def build_grammar_profile(
    difficulties: Iterable[DecisionDifficulty],
    *,
    profile_id: str,
    signature: str,
) -> GrammarProfile:
    """Aggregate ``DecisionDifficulty`` records into one ``GrammarProfile``.

    The aggregation is deterministic and does not reorder records. Missing
    entropy values are excluded from means and maxima.
    """
    diffs = list(difficulties)
    if not diffs:
        return GrammarProfile(
            profile_id=profile_id,
            signature=signature,
            decision_count=0,
            mean_arity=0.0,
            max_arity=0,
            mean_entropy_bits=None,
            max_entropy_bits=None,
        )

    arities = [d.live_legal_action_count for d in diffs]
    entropies = [
        d.posterior_entropy_bits
        for d in diffs
        if d.posterior_entropy_bits is not None
    ]
    source_hashes = sorted({d.source_hash for d in diffs if d.source_hash})
    return GrammarProfile(
        profile_id=profile_id,
        signature=signature,
        decision_count=len(diffs),
        mean_arity=sum(arities) / len(arities),
        max_arity=max(arities),
        mean_entropy_bits=(sum(entropies) / len(entropies)) if entropies else None,
        max_entropy_bits=max(entropies) if entropies else None,
        source_hash=_stable_hash({"sources": source_hashes}) if source_hashes else None,
    )


def build_grammar_profile_manifest(
    profiles: Iterable[GrammarProfile],
    *,
    run_id: str,
    source_manifest_sha: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Build a versioned manifest over one or more grammar profiles."""
    profile_list = list(profiles)
    profile_hashes = [p.source_hash for p in profile_list if p.source_hash]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": run_id,
        "source_manifest_sha": source_manifest_sha,
        "profile_count": len(profile_list),
        "profiles": [p.to_dict() for p in profile_list],
        "manifest_hash": _stable_hash(
            {
                "run_id": run_id,
                "source_manifest_sha": source_manifest_sha or "",
                "profiles": profile_hashes,
            }
        ),
        "note": note,
    }


def validate_grammar_profile_manifest(
    manifest: Mapping[str, Any],
) -> list[str]:
    """Return a list of validation errors; empty means valid."""
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if not isinstance(manifest.get("run_id"), str) or not manifest.get("run_id"):
        errors.append("run_id must be a non-empty string")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list):
        errors.append("profiles must be a list")
        return errors
    for idx, prof in enumerate(profiles):
        prefix = f"profiles[{idx}]"
        if not isinstance(prof, dict):
            errors.append(f"{prefix} must be an object")
            continue
        required = ("profile_id", "signature", "decision_count", "mean_arity", "max_arity")
        for key in required:
            if key not in prof:
                errors.append(f"{prefix} missing {key!r}")
    return errors
