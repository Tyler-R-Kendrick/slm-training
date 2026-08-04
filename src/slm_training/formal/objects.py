"""Portable formal objects for VSS certificates and Lean claim modules.

A :class:`FormalObjectV1` is a content-addressed, JSON-serializable statement
plus payload that **independent provers** can check without sharing a single
kernel trust root. Lean is one optional checker, never the sole authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

FORMAL_OBJECT_SCHEMA = "formal_object/v1"
SUPPORT_CERTIFICATE_STATEMENT = "vss.support_certificate/v1"
LEAN_CLAIM_STATEMENT = "lean.claim_module/v1"
CLOSURE_LAW_STATEMENT = "vss.exact_closure_law/v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FormalObjectKind(str, Enum):
    SUPPORT_CERTIFICATE = "support_certificate"
    LEAN_CLAIM = "lean_claim"
    CLOSURE_LAW = "closure_law"


@dataclass(frozen=True)
class FormalObjectV1:
    """Exportable formal object checked by independent provers."""

    kind: FormalObjectKind
    object_id: str
    statement: dict[str, Any]
    payload: dict[str, Any]
    content_digest: str
    required_checkers: tuple[str, ...]
    tier: str
    schema: str = FORMAL_OBJECT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "object_id": self.object_id,
            "statement": self.statement,
            "payload": self.payload,
            "content_digest": self.content_digest,
            "required_checkers": list(self.required_checkers),
            "tier": self.tier,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FormalObjectV1:
        if data.get("schema") != FORMAL_OBJECT_SCHEMA:
            raise ValueError(
                f"expected schema {FORMAL_OBJECT_SCHEMA!r}, got {data.get('schema')!r}"
            )
        required = tuple(str(item) for item in data.get("required_checkers", ()))
        obj = cls(
            kind=FormalObjectKind(str(data["kind"])),
            object_id=str(data["object_id"]),
            statement=dict(data["statement"]),
            payload=dict(data["payload"]),
            content_digest=str(data["content_digest"]),
            required_checkers=required,
            tier=str(data.get("tier", "production_core")),
            schema=str(data["schema"]),
        )
        recomputed = _object_digest(obj.kind, obj.object_id, obj.statement, obj.payload)
        if recomputed != obj.content_digest:
            raise ValueError(
                "formal object content_digest mismatch "
                f"(declared={obj.content_digest}, recomputed={recomputed})"
            )
        return obj


def _object_digest(
    kind: FormalObjectKind,
    object_id: str,
    statement: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    body = {
        "kind": kind.value,
        "object_id": object_id,
        "statement": statement,
        "payload": payload,
    }
    return _sha256(_canonical_json(body))


def export_support_certificate(
    certificate: Any,
    *,
    object_id: str | None = None,
) -> FormalObjectV1:
    """Lift a VSS :class:`SupportCertificate` into an exportable formal object.

    The payload embeds the certificate JSON and its digest. Independent checkers
    re-validate honesty laws and optionally re-run search replay.
    """

    from slm_training.dsl.solver.support import SupportCertificate

    if not isinstance(certificate, SupportCertificate):
        raise TypeError("export_support_certificate requires a SupportCertificate")

    cert_dict = certificate.to_dict()
    cert_digest = certificate.digest
    statement = {
        "family": SUPPORT_CERTIFICATE_STATEMENT,
        "verdict": certificate.verdict.value,
        "state_fingerprint": certificate.query.state_fingerprint,
        "problem_id": certificate.problem_id,
        "pack_id": certificate.pack_id,
        "constraint_version": certificate.constraint_version,
        "certificate_digest": cert_digest,
        "laws": [
            "unsupported_requires_exhaustion",
            "unsupported_forbids_budget_stop",
            "unsupported_forbids_partial_coverage",
            "supported_requires_witness_digest",
            "unknown_never_authorizes_removal",
            "digest_matches_payload",
        ],
    }
    payload = {
        "certificate": cert_dict,
        "certificate_digest": cert_digest,
        "schema_version": certificate.schema_version,
    }
    oid = object_id or f"support-cert:{cert_digest[:16]}"
    digest = _object_digest(FormalObjectKind.SUPPORT_CERTIFICATE, oid, statement, payload)
    return FormalObjectV1(
        kind=FormalObjectKind.SUPPORT_CERTIFICATE,
        object_id=oid,
        statement=statement,
        payload=payload,
        content_digest=digest,
        # Two pure backends always; full search replay is an optional third.
        required_checkers=("python_structural", "python_reference"),
        tier="production_core",
    )


# Lean claim catalog: theorem id → (module, statement family, structural law ids).
# Structural checkers implement these laws in pure Python so the loop does not
# depend on a Lean binary.
_LEAN_CLAIM_CATALOG: dict[str, dict[str, Any]] = {
    "Forest.close_monotone": {
        "module": "LeverProofLean.Forest",
        "tier": "production_core",
        "laws": ["close_monotone", "close_never_adds_live"],
    },
    "Forest.close_idempotent": {
        "module": "LeverProofLean.Forest",
        "tier": "production_core",
        "laws": ["close_idempotent"],
    },
    "Forest.close_history_preserved": {
        "module": "LeverProofLean.Forest",
        "tier": "production_core",
        "laws": ["close_history_preserved"],
    },
    "ExactClosure.closePass_subset": {
        "module": "LeverProofLean.ExactClosure",
        "tier": "production_core",
        "laws": ["close_pass_subset", "supported_not_removable"],
    },
    "ExactClosure.honest_when_no_accepted_unsupported": {
        "module": "LeverProofLean.ExactClosure",
        "tier": "production_core",
        "laws": ["honest_fixed_point", "unknown_not_removable"],
    },
    "DecodeInvariants.singleton_bypasses_ranker": {
        "module": "LeverProofLean.DecodeInvariants",
        "tier": "production_core",
        "laws": ["singleton_bypass", "empty_dead_end"],
    },
    "EcosystemTier.core_success_ignores_library_size": {
        "module": "LeverProofLean.EcosystemTier",
        "tier": "ecosystem_library",
        "laws": ["core_ignores_library_size", "core_ecosystem_disjoint"],
    },
    "StructuralMetrics.structural_similarity_mono": {
        "module": "LeverProofLean.StructuralMetrics",
        "tier": "ecosystem_library",
        "laws": ["structural_similarity_mono", "recall_mono"],
    },
}


def lean_claim_catalog() -> dict[str, dict[str, Any]]:
    """Return the exportable Lean claim catalog (theorem id → metadata)."""

    return {key: dict(value) for key, value in _LEAN_CLAIM_CATALOG.items()}


def export_lean_claim(
    theorem_id: str,
    *,
    object_id: str | None = None,
    proof_status: str = "proved",
) -> FormalObjectV1:
    """Export a Lean theorem as a formal object with structural law hooks.

    ``proof_status`` records the Lean-side status when known; structural and
    multi-backend checks do not trust that field alone.
    """

    if theorem_id not in _LEAN_CLAIM_CATALOG:
        raise KeyError(f"unknown lean claim theorem_id: {theorem_id!r}")
    meta = _LEAN_CLAIM_CATALOG[theorem_id]
    statement = {
        "family": LEAN_CLAIM_STATEMENT,
        "theorem_id": theorem_id,
        "module": meta["module"],
        "laws": list(meta["laws"]),
        "proof_status_declared": proof_status,
    }
    payload = {
        "theorem_id": theorem_id,
        "module": meta["module"],
        "laws": list(meta["laws"]),
        "lean_source_hint": f"src/leverproof_lean/{meta['module'].replace('.', '/')}.lean",
    }
    oid = object_id or f"lean-claim:{theorem_id}"
    digest = _object_digest(FormalObjectKind.LEAN_CLAIM, oid, statement, payload)
    return FormalObjectV1(
        kind=FormalObjectKind.LEAN_CLAIM,
        object_id=oid,
        statement=statement,
        payload=payload,
        content_digest=digest,
        # Two independent pure backends; Lean kernel is optional third.
        required_checkers=("python_structural", "python_reference"),
        tier=str(meta["tier"]),
    )


def export_closure_law(
    law_id: str,
    *,
    object_id: str | None = None,
) -> FormalObjectV1:
    """Export a VSS exact-closure honesty law as a formal object."""

    laws = {
        "removable_only_replay_valid_unsupported": [
            "supported_not_removable",
            "unknown_not_removable",
            "failed_replay_not_removable",
        ],
        "close_pass_only_shrinks": ["close_pass_subset"],
        "fixed_point_when_no_removals": ["honest_fixed_point"],
    }
    if law_id not in laws:
        raise KeyError(f"unknown closure law: {law_id!r}")
    statement = {
        "family": CLOSURE_LAW_STATEMENT,
        "law_id": law_id,
        "laws": list(laws[law_id]),
    }
    payload = {"law_id": law_id, "laws": list(laws[law_id])}
    oid = object_id or f"closure-law:{law_id}"
    digest = _object_digest(FormalObjectKind.CLOSURE_LAW, oid, statement, payload)
    return FormalObjectV1(
        kind=FormalObjectKind.CLOSURE_LAW,
        object_id=oid,
        statement=statement,
        payload=payload,
        content_digest=digest,
        required_checkers=("python_structural", "python_reference"),
        tier="production_core",
    )


__all__ = [
    "CLOSURE_LAW_STATEMENT",
    "FORMAL_OBJECT_SCHEMA",
    "LEAN_CLAIM_STATEMENT",
    "SUPPORT_CERTIFICATE_STATEMENT",
    "FormalObjectKind",
    "FormalObjectV1",
    "export_closure_law",
    "export_lean_claim",
    "export_support_certificate",
    "lean_claim_catalog",
]
