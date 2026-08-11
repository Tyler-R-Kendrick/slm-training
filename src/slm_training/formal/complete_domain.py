"""Proof-bearing finite domains (KERN-02 / SLM-517).

Mirrors ``LeverProofLean.CompleteDomain``: a finite candidate list bound to a
state identity and checked sound/complete against an explicit legality
extension. A bare ``coverage_complete`` / ``coverageComplete`` Boolean is
telemetry only and cannot manufacture completeness authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

COMPLETE_DOMAIN_SCHEMA = "complete_domain/v1"

DomainAuthorityName = Literal[
    "proved_complete",
    "partial_coverage",
    "unknown_coverage",
]


@dataclass(frozen=True)
class CompleteDomainEvidenceV1:
    """Checked complete-domain evidence at one decode/closure state.

    Authority requires ``checked_sound`` and ``checked_complete`` both true,
    nodup candidates, and a non-empty state identity. Legacy Boolean coverage
    flags are preserved for artifact readability only.
    """

    state_id: str
    candidates: tuple[int, ...]
    legal_members: tuple[int, ...]
    checked_sound: bool
    checked_complete: bool
    legacy_coverage_complete: bool | None = None

    def __bool__(self) -> bool:
        raise TypeError(
            "CompleteDomainEvidenceV1 forbids truthy/falsy coercion; "
            "call is_proved_complete() or authorizes_* explicitly"
        )

    def has_duplicate_candidates(self) -> bool:
        return len(self.candidates) != len(set(self.candidates))

    def is_proved_complete(self) -> bool:
        return (
            bool(self.state_id)
            and self.checked_sound
            and self.checked_complete
            and not self.has_duplicate_candidates()
        )

    def binds_state(self, expected_state_id: str) -> bool:
        return self.is_proved_complete() and self.state_id == expected_state_id

    def authorizes_pruning(self, *, expected_state_id: str | None = None) -> bool:
        if expected_state_id is None:
            return self.is_proved_complete()
        return self.binds_state(expected_state_id)

    def authorizes_forced_emission(self, *, expected_state_id: str | None = None) -> bool:
        if not self.authorizes_pruning(expected_state_id=expected_state_id):
            return False
        return len(self.candidates) == 1

    def singleton_token(self, *, expected_state_id: str | None = None) -> int | None:
        if not self.authorizes_forced_emission(expected_state_id=expected_state_id):
            return None
        return self.candidates[0]

    def authority_name(self) -> DomainAuthorityName:
        if self.is_proved_complete():
            return "proved_complete"
        if self.legacy_coverage_complete is True:
            return "partial_coverage"
        return "unknown_coverage"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPLETE_DOMAIN_SCHEMA,
            "state_id": self.state_id,
            "candidates": list(self.candidates),
            "legal_members": list(self.legal_members),
            "checked_sound": self.checked_sound,
            "checked_complete": self.checked_complete,
            "legacy_coverage_complete": self.legacy_coverage_complete,
            "authority": self.authority_name(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompleteDomainEvidenceV1:
        if data.get("schema") != COMPLETE_DOMAIN_SCHEMA:
            raise ValueError(f"expected schema {COMPLETE_DOMAIN_SCHEMA!r}")
        evidence = cls(
            state_id=str(data["state_id"]),
            candidates=tuple(int(x) for x in data["candidates"]),
            legal_members=tuple(int(x) for x in data["legal_members"]),
            checked_sound=bool(data["checked_sound"]),
            checked_complete=bool(data["checked_complete"]),
            legacy_coverage_complete=(
                None
                if data.get("legacy_coverage_complete") is None
                else bool(data["legacy_coverage_complete"])
            ),
        )
        # Recompute — never trust serialized checked_* bits alone.
        rebuilt = build_complete_domain(
            evidence.state_id,
            evidence.candidates,
            evidence.legal_members,
            legacy_coverage_complete=evidence.legacy_coverage_complete,
        )
        if (
            rebuilt.checked_sound != evidence.checked_sound
            or rebuilt.checked_complete != evidence.checked_complete
        ):
            raise ValueError("checked_sound/complete disagree with legality evidence")
        return rebuilt


def build_complete_domain(
    state_id: str,
    candidates: Sequence[int],
    legal_members: Sequence[int],
    *,
    legacy_coverage_complete: bool | None = None,
) -> CompleteDomainEvidenceV1:
    """Build evidence by checking soundness and completeness against legality."""

    cands = tuple(int(x) for x in candidates)
    legal = tuple(int(x) for x in legal_members)
    legal_set = set(legal)
    cand_set = set(cands)
    nodup = len(cands) == len(cand_set)
    sound = nodup and all(c in legal_set for c in cands)
    complete = nodup and all(member in cand_set for member in legal)
    return CompleteDomainEvidenceV1(
        state_id=str(state_id),
        candidates=cands,
        legal_members=legal,
        checked_sound=sound,
        checked_complete=complete,
        legacy_coverage_complete=legacy_coverage_complete,
    )


def from_legacy_coverage_flag(
    state_id: str,
    candidates: Sequence[int],
    coverage_complete: bool,
) -> CompleteDomainEvidenceV1:
    """Migration adapter: preserve telemetry; never manufacture completeness.

    A forged ``coverage_complete=True`` bit alone has zero semantic authority.
    """

    return CompleteDomainEvidenceV1(
        state_id=str(state_id),
        candidates=tuple(int(x) for x in candidates),
        legal_members=(),
        checked_sound=False,
        checked_complete=False,
        legacy_coverage_complete=bool(coverage_complete),
    )


def legacy_flag_authorizes_completeness(coverage_complete: bool) -> bool:
    """Always false — mirrors Lean ``legacyFlagAuthorizesCompleteness``."""

    del coverage_complete
    return False


def finite_search_authorized(
    evidence: CompleteDomainEvidenceV1, expected_state_id: str
) -> bool:
    """Exact-closure / finite-search gate: proved domain bound to state."""

    return evidence.binds_state(expected_state_id)


__all__ = [
    "COMPLETE_DOMAIN_SCHEMA",
    "CompleteDomainEvidenceV1",
    "DomainAuthorityName",
    "build_complete_domain",
    "finite_search_authorized",
    "from_legacy_coverage_flag",
    "legacy_flag_authorizes_completeness",
]
