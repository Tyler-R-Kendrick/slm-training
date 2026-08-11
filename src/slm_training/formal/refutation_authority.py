"""Checked refutation authority (EVID-09 / SLM-542).

Destructive ``refuted`` / UNSUPPORTED authority requires either:

* exact bounded semantic replay bound to state/problem/source/tool identity, or
* a checked proof-certificate path with capability + trust evidence.

Self-attested Booleans (``exhausted``, ``coverage=complete``, ``replay_ok``)
are telemetry only and never authorize candidate removal alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

REFUTATION_EVIDENCE_SCHEMA = "checked_refutation_evidence/v1"

RefutationKindName = Literal["exact_replay", "checked_certificate"]


@dataclass(frozen=True)
class BindingIdsV1:
    """Identity binding for replay/certificate checks."""

    state_id: str
    problem_id: str
    source_id: str
    tool_id: str

    def nonempty(self) -> bool:
        return all(
            bool(part)
            for part in (self.state_id, self.problem_id, self.source_id, self.tool_id)
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "state_id": self.state_id,
            "problem_id": self.problem_id,
            "source_id": self.source_id,
            "tool_id": self.tool_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BindingIdsV1:
        return cls(
            state_id=str(data["state_id"]),
            problem_id=str(data["problem_id"]),
            source_id=str(data["source_id"]),
            tool_id=str(data["tool_id"]),
        )


@dataclass(frozen=True)
class CheckedRefutationEvidenceV1:
    """Authority-critical checked refutation evidence reference."""

    kind: RefutationKindName
    binding: BindingIdsV1
    evidence_digest: str
    capability_present: bool
    trust_checked: bool

    def well_formed(self) -> bool:
        if not self.binding.nonempty() or not self.evidence_digest or not self.trust_checked:
            return False
        if self.kind == "checked_certificate":
            return self.capability_present
        return self.kind == "exact_replay"

    def binds_expected(self, expected: BindingIdsV1) -> bool:
        return self.well_formed() and self.binding == expected

    def authorizes_removal(self, expected: BindingIdsV1) -> bool:
        return self.binds_expected(expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REFUTATION_EVIDENCE_SCHEMA,
            "kind": self.kind,
            "binding": self.binding.to_dict(),
            "evidence_digest": self.evidence_digest,
            "capability_present": self.capability_present,
            "trust_checked": self.trust_checked,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckedRefutationEvidenceV1:
        if data.get("schema") not in (None, REFUTATION_EVIDENCE_SCHEMA):
            raise ValueError(f"expected schema {REFUTATION_EVIDENCE_SCHEMA!r}")
        kind = str(data["kind"])
        if kind not in {"exact_replay", "checked_certificate"}:
            raise ValueError(f"invalid refutation kind: {kind!r}")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            binding=BindingIdsV1.from_dict(data["binding"]),
            evidence_digest=str(data["evidence_digest"]),
            capability_present=bool(data.get("capability_present", False)),
            trust_checked=bool(data.get("trust_checked", False)),
        )


@dataclass(frozen=True)
class LegacyExhaustionFlagV1:
    """Telemetry-only self-attested exhaustion / coverage / replay bits."""

    exhausted: bool = False
    replay_ok: bool = False
    coverage_complete: bool = False


def legacy_exhaustion_authorizes_removal(_flag: LegacyExhaustionFlagV1) -> bool:
    """Always false — mirrors Lean ``legacyExhaustionAuthorizesRemoval``."""

    return False


def evidence_authorizes_removal(
    evidence: CheckedRefutationEvidenceV1 | None,
    expected: BindingIdsV1 | None,
) -> bool:
    if evidence is None or expected is None:
        return False
    return evidence.authorizes_removal(expected)


def make_exact_replay_evidence(
    binding: BindingIdsV1,
    *,
    evidence_digest: str,
) -> CheckedRefutationEvidenceV1:
    return CheckedRefutationEvidenceV1(
        kind="exact_replay",
        binding=binding,
        evidence_digest=evidence_digest,
        capability_present=True,
        trust_checked=True,
    )


def make_checked_certificate_evidence(
    binding: BindingIdsV1,
    *,
    evidence_digest: str,
    capability_present: bool = True,
    trust_checked: bool = True,
) -> CheckedRefutationEvidenceV1:
    return CheckedRefutationEvidenceV1(
        kind="checked_certificate",
        binding=binding,
        evidence_digest=evidence_digest,
        capability_present=capability_present,
        trust_checked=trust_checked,
    )


def parse_binding(raw: Mapping[str, Any] | None) -> BindingIdsV1 | None:
    if raw is None:
        return None
    return BindingIdsV1.from_dict(raw)


def parse_refutation_evidence(
    raw: Mapping[str, Any] | None,
) -> CheckedRefutationEvidenceV1 | None:
    if raw is None:
        return None
    return CheckedRefutationEvidenceV1.from_dict(raw)


__all__ = [
    "REFUTATION_EVIDENCE_SCHEMA",
    "BindingIdsV1",
    "CheckedRefutationEvidenceV1",
    "LegacyExhaustionFlagV1",
    "RefutationKindName",
    "evidence_authorizes_removal",
    "legacy_exhaustion_authorizes_removal",
    "make_checked_certificate_evidence",
    "make_exact_replay_evidence",
    "parse_binding",
    "parse_refutation_evidence",
]
