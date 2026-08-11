"""Deterministic bounded domain-obstruction cores (PGS-C02 / SLM-500).

Private implementation module; public symbols are re-exported from
``goal_support.py``.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slm_training.data.progspec.goal_constraints import canonical_json_bytes
from slm_training.dsl.solver.goal_support import (
    GOAL_SUPPORT_IMPLEMENTATION_VERSION,
    GoalTerminalEvidenceV1,
    GoalVerifierProfileV1,
    exact_atom_allowed_for_obstruction,
    exact_mandatory_failure_atom_ids,
)
from slm_training.dsl.solver.state import FiniteDomainState, SupportVerdict
from slm_training.dsl.solver.support import SupportCertificate, replay_support_certificate

DOMAIN_OBSTRUCTION_CORE_SCHEMA_VERSION = "domain_obstruction_core/v1"
DOMAIN_OBSTRUCTION_CLAIM = (
    "Every fully observed rejected terminal in the declared bounded domain "
    "fails at least one exact mandatory atom in this recorded hitting set."
)
OBSTRUCTION_ALGORITHM_ID = "goal_support_obstruction/v1"
OBSTRUCTION_ALGORITHM_VERSION = "v1"
MIN_CARDINALITY_EXACT_ATOM_THRESHOLD = 16
SUBSET_MINIMAL_DELETION_ATOM_THRESHOLD = 64

ObstructionCoreMode = Literal[
    "minimum_cardinality_exact",
    "subset_minimal_exact",
    "sound_overapprox",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TerminalFailureSetV1(_StrictModel):
    """One rejected terminal and its exact mandatory failure atom ids."""

    terminal_evidence_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    exact_mandatory_failure_atom_ids: tuple[str, ...]


class ObstructionCoreStatsV1(_StrictModel):
    atom_universe_size: int = Field(ge=0)
    terminal_count: int = Field(ge=0)
    subsets_examined: int = Field(default=0, ge=0)
    deletion_steps: int = Field(default=0, ge=0)


class DomainObstructionCoreV1(_StrictModel):
    """Digest-bound hitting set over exact mandatory failure atoms."""

    schema_version: str = Field(
        default=DOMAIN_OBSTRUCTION_CORE_SCHEMA_VERSION,
        pattern=r"^domain_obstruction_core/v1$",
    )
    claim: str = Field(default=DOMAIN_OBSTRUCTION_CLAIM, min_length=1)
    mode: ObstructionCoreMode
    core_atoms: tuple[str, ...]
    terminal_failure_sets: tuple[TerminalFailureSetV1, ...]
    algorithm_id: str = Field(default=OBSTRUCTION_ALGORITHM_ID, min_length=1)
    algorithm_version: str = Field(default=OBSTRUCTION_ALGORITHM_VERSION, min_length=1)
    min_cardinality_exact_threshold: int = Field(
        default=MIN_CARDINALITY_EXACT_ATOM_THRESHOLD, ge=1
    )
    subset_minimal_deletion_threshold: int = Field(
        default=SUBSET_MINIMAL_DELETION_ATOM_THRESHOLD, ge=1
    )
    profile_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    state_fingerprint: str = Field(min_length=1)
    bounds_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    base_certificate_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    terminal_set_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    atom_vocabulary_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    implementation_version: str = Field(
        default=GOAL_SUPPORT_IMPLEMENTATION_VERSION, min_length=1
    )
    stats: ObstructionCoreStatsV1
    core_digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")

    @model_validator(mode="after")
    def _check_integrity(self) -> "DomainObstructionCoreV1":
        if len(self.core_atoms) != len(set(self.core_atoms)):
            raise ValueError("core_atoms must be unique")
        terminal_digests = [
            item.terminal_evidence_digest for item in self.terminal_failure_sets
        ]
        if len(terminal_digests) != len(set(terminal_digests)):
            raise ValueError("terminal_failure_sets digests must be unique")
        for item in self.terminal_failure_sets:
            if len(item.exact_mandatory_failure_atom_ids) != len(
                set(item.exact_mandatory_failure_atom_ids)
            ):
                raise ValueError(
                    f"duplicate atom ids for terminal {item.terminal_evidence_digest}"
                )
        if self.core_digest:
            expected = self.compute_digest()
            if self.core_digest != expected:
                raise ValueError("core_digest does not match canonical payload")
        return self

    def _canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("core_digest", None)
        payload["core_atoms"] = sorted(payload["core_atoms"])
        payload["terminal_failure_sets"] = sorted(
            [
                {
                    "terminal_evidence_digest": item["terminal_evidence_digest"],
                    "exact_mandatory_failure_atom_ids": sorted(
                        item["exact_mandatory_failure_atom_ids"]
                    ),
                }
                for item in payload["terminal_failure_sets"]
            ],
            key=lambda item: item["terminal_evidence_digest"],
        )
        return payload

    def compute_digest(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self._canonical_payload())
        ).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_digest:
            payload.pop("core_digest", None)
        elif not payload.get("core_digest"):
            payload["core_digest"] = self.compute_digest()
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DomainObstructionCoreV1":
        if str(value.get("schema_version")) != DOMAIN_OBSTRUCTION_CORE_SCHEMA_VERSION:
            raise ValueError("unsupported DomainObstructionCoreV1 version")
        if not value.get("core_digest"):
            raise ValueError("persisted DomainObstructionCoreV1 requires core_digest")
        return cls.model_validate(value)


def terminal_failure_set_from_evidence(
    record: GoalTerminalEvidenceV1,
) -> TerminalFailureSetV1 | None:
    """Build a failure set when the terminal is a fully observed REJECT."""
    if record.overall_status != "REJECT":
        return None
    if record.mandatory_unknown_atoms:
        return None
    atom_ids = exact_mandatory_failure_atom_ids(record)
    if not atom_ids:
        return None
    digest = record.evidence_digest or record.compute_digest()
    return TerminalFailureSetV1(
        terminal_evidence_digest=digest,
        exact_mandatory_failure_atom_ids=atom_ids,
    )


def _terminal_set_digest(failure_sets: tuple[TerminalFailureSetV1, ...]) -> str:
    canonical = sorted(
        [
            {
                "terminal_evidence_digest": item.terminal_evidence_digest,
                "exact_mandatory_failure_atom_ids": sorted(
                    item.exact_mandatory_failure_atom_ids
                ),
            }
            for item in failure_sets
        ],
        key=lambda item: item["terminal_evidence_digest"],
    )
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def _atom_vocabulary_digest(atom_universe: tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(sorted(atom_universe))).hexdigest()


def _bounds_digest(certificate: SupportCertificate) -> str:
    return hashlib.sha256(
        canonical_json_bytes(certificate.bounds.to_dict())
    ).hexdigest()


def _certificate_coverage_complete(certificate: SupportCertificate) -> bool:
    if any(cov in {"partial", "none"} for cov in certificate.coverage_observations):
        return False
    return bool(certificate.coverage_observations)


def obstruction_core_emission_allowed(
    *,
    certificate: SupportCertificate,
    terminal_records: tuple[GoalTerminalEvidenceV1, ...],
    replay_ok: bool,
) -> bool:
    """Return True only when all legal emission preconditions hold."""
    if certificate.verdict is not SupportVerdict.UNSUPPORTED:
        return False
    if not replay_ok:
        return False
    if not certificate.exhausted:
        return False
    if certificate.stop_reason is not None:
        return False
    if not _certificate_coverage_complete(certificate):
        return False

    rejected = 0
    for record in terminal_records:
        if record.overall_status != "REJECT":
            continue
        rejected += 1
        if record.mandatory_unknown_atoms:
            return False
        if not exact_mandatory_failure_atom_ids(record):
            return False
        for atom in record.mandatory_failure_atoms:
            if not exact_atom_allowed_for_obstruction(atom):
                return False
    return rejected >= 1


def _collect_failure_sets(
    terminal_records: tuple[GoalTerminalEvidenceV1, ...],
) -> tuple[TerminalFailureSetV1, ...]:
    sets: list[TerminalFailureSetV1] = []
    for record in terminal_records:
        failure_set = terminal_failure_set_from_evidence(record)
        if failure_set is not None:
            sets.append(failure_set)
    return tuple(
        sorted(sets, key=lambda item: item.terminal_evidence_digest)
    )


def _atom_universe(
    failure_sets: tuple[TerminalFailureSetV1, ...],
) -> tuple[str, ...]:
    atoms: set[str] = set()
    for item in failure_sets:
        atoms.update(item.exact_mandatory_failure_atom_ids)
    return tuple(sorted(atoms))


def _hits_all(core: tuple[str, ...], failure_sets: tuple[TerminalFailureSetV1, ...]) -> bool:
    core_set = set(core)
    return all(core_set & set(item.exact_mandatory_failure_atom_ids) for item in failure_sets)


def _minimum_cardinality_exact_core(
    failure_sets: tuple[TerminalFailureSetV1, ...],
    atom_universe: tuple[str, ...],
) -> tuple[tuple[str, ...], int]:
    subsets_examined = 0
    best: tuple[str, ...] | None = None
    for size in range(1, len(atom_universe) + 1):
        for combo in itertools.combinations(atom_universe, size):
            subsets_examined += 1
            candidate = tuple(sorted(combo))
            if not _hits_all(candidate, failure_sets):
                continue
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            break
    if best is None:
        raise ValueError("no hitting set exists for declared failure sets")
    return best, subsets_examined


def _subset_minimal_deletion_core(
    failure_sets: tuple[TerminalFailureSetV1, ...],
    atom_universe: tuple[str, ...],
) -> tuple[tuple[str, ...], int]:
    core = list(atom_universe)
    deletion_steps = 0
    changed = True
    while changed:
        changed = False
        for atom in sorted(core):
            trial = tuple(item for item in core if item != atom)
            if trial and _hits_all(trial, failure_sets):
                core.remove(atom)
                deletion_steps += 1
                changed = True
                break
    if not _hits_all(tuple(core), failure_sets):
        raise ValueError("subset-minimal deletion did not produce a hitting set")
    return tuple(sorted(core)), deletion_steps


def _sound_overapprox_core(
    failure_sets: tuple[TerminalFailureSetV1, ...],
) -> tuple[str, ...]:
    atoms: set[str] = set()
    for item in failure_sets:
        atoms.add(item.exact_mandatory_failure_atom_ids[0])
    return tuple(sorted(atoms))


def _select_mode(
    atom_universe: tuple[str, ...],
    *,
    requested: ObstructionCoreMode | None,
) -> ObstructionCoreMode:
    if requested is not None:
        return requested
    size = len(atom_universe)
    if size <= MIN_CARDINALITY_EXACT_ATOM_THRESHOLD:
        return "minimum_cardinality_exact"
    if size <= SUBSET_MINIMAL_DELETION_ATOM_THRESHOLD:
        return "subset_minimal_exact"
    return "sound_overapprox"


def compute_domain_obstruction_core(
    *,
    certificate: SupportCertificate,
    terminal_records: tuple[GoalTerminalEvidenceV1, ...],
    profile: GoalVerifierProfileV1,
    state: FiniteDomainState,
    replay_ok: bool,
    mode: ObstructionCoreMode | None = None,
) -> DomainObstructionCoreV1 | None:
    """Compute an exact obstruction core when emission preconditions hold."""
    if not obstruction_core_emission_allowed(
        certificate=certificate,
        terminal_records=terminal_records,
        replay_ok=replay_ok,
    ):
        return None

    failure_sets = _collect_failure_sets(terminal_records)
    if not failure_sets:
        return None

    atom_universe = _atom_universe(failure_sets)
    selected_mode = _select_mode(atom_universe, requested=mode)
    subsets_examined = 0
    deletion_steps = 0

    if selected_mode == "minimum_cardinality_exact":
        core_atoms, subsets_examined = _minimum_cardinality_exact_core(
            failure_sets, atom_universe
        )
    elif selected_mode == "subset_minimal_exact":
        core_atoms, deletion_steps = _subset_minimal_deletion_core(
            failure_sets, atom_universe
        )
    else:
        core_atoms = _sound_overapprox_core(failure_sets)

    profile_digest = profile.digest or profile.compute_digest()
    stats = ObstructionCoreStatsV1(
        atom_universe_size=len(atom_universe),
        terminal_count=len(failure_sets),
        subsets_examined=subsets_examined,
        deletion_steps=deletion_steps,
    )
    payload = {
        "mode": selected_mode,
        "core_atoms": core_atoms,
        "terminal_failure_sets": failure_sets,
        "profile_digest": profile_digest,
        "state_fingerprint": state.fingerprint,
        "bounds_digest": _bounds_digest(certificate),
        "base_certificate_digest": certificate.digest,
        "terminal_set_digest": _terminal_set_digest(failure_sets),
        "atom_vocabulary_digest": _atom_vocabulary_digest(atom_universe),
        "stats": stats,
    }
    core = DomainObstructionCoreV1(**payload)
    return DomainObstructionCoreV1.from_dict(core.to_dict(include_digest=True))


def validate_domain_obstruction_core(
    core: DomainObstructionCoreV1,
    *,
    certificate: SupportCertificate,
    terminal_records: tuple[GoalTerminalEvidenceV1, ...],
    profile: GoalVerifierProfileV1,
    state: FiniteDomainState,
    replay_ok: bool,
) -> tuple[bool, tuple[str, ...]]:
    """Validate a recorded core against live inputs; return (ok, violations)."""
    violations: list[str] = []

    if core.claim != DOMAIN_OBSTRUCTION_CLAIM:
        violations.append("claim wording mismatch")

    expected_digest = core.compute_digest()
    if core.core_digest and core.core_digest != expected_digest:
        violations.append("tampered core_digest")

    profile_digest = profile.digest or profile.compute_digest()
    if core.profile_digest != profile_digest:
        violations.append("stale profile_digest")
    if core.state_fingerprint != state.fingerprint:
        violations.append("stale state_fingerprint")
    if core.bounds_digest != _bounds_digest(certificate):
        violations.append("stale bounds_digest")
    if core.base_certificate_digest != certificate.digest:
        violations.append("stale base_certificate_digest")

    if not obstruction_core_emission_allowed(
        certificate=certificate,
        terminal_records=terminal_records,
        replay_ok=replay_ok,
    ):
        violations.append("emission preconditions no longer hold")

    failure_sets = _collect_failure_sets(terminal_records)
    if _terminal_set_digest(failure_sets) != core.terminal_set_digest:
        violations.append("terminal_set_digest mismatch")
    atom_universe = _atom_universe(failure_sets)
    if _atom_vocabulary_digest(atom_universe) != core.atom_vocabulary_digest:
        violations.append("atom_vocabulary_digest mismatch")

    if not core.core_atoms:
        violations.append("empty core_atoms")
    if len(core.core_atoms) != len(set(core.core_atoms)):
        violations.append("duplicate core_atoms")

    dangling = set(core.core_atoms) - set(atom_universe)
    if dangling:
        violations.append(f"dangling core atoms: {sorted(dangling)}")

    if not _hits_all(core.core_atoms, failure_sets):
        violations.append("core does not hit every rejected terminal")

    for record in terminal_records:
        if record.mandatory_unknown_atoms:
            violations.append("mandatory unknown atoms present")
            break
        for atom in record.mandatory_failure_atoms:
            if not exact_atom_allowed_for_obstruction(atom):
                violations.append("heuristic/advisory atom laundered into core inputs")
                break

    if core.mode == "minimum_cardinality_exact":
        best, _ = _minimum_cardinality_exact_core(failure_sets, atom_universe)
        if tuple(core.core_atoms) != best:
            violations.append("minimum_cardinality_exact claim violated")
    elif core.mode == "subset_minimal_exact":
        for atom in core.core_atoms:
            trial = tuple(item for item in core.core_atoms if item != atom)
            if _hits_all(trial, failure_sets):
                violations.append(f"subset_minimal_exact: atom {atom!r} is removable")
    elif core.mode != "sound_overapprox":
        violations.append(f"unknown mode {core.mode!r}")

    recorded_sets = {
        item.terminal_evidence_digest: tuple(item.exact_mandatory_failure_atom_ids)
        for item in core.terminal_failure_sets
    }
    live_sets = {
        item.terminal_evidence_digest: tuple(item.exact_mandatory_failure_atom_ids)
        for item in failure_sets
    }
    if recorded_sets != live_sets:
        violations.append("terminal_failure_sets payload mismatch")

    return not violations, tuple(violations)


def obstruction_core_algorithm_table() -> dict[str, Any]:
    """Document thresholds, modes, and complexity boundaries."""
    return {
        "algorithm_id": OBSTRUCTION_ALGORITHM_ID,
        "algorithm_version": OBSTRUCTION_ALGORITHM_VERSION,
        "claim": DOMAIN_OBSTRUCTION_CLAIM,
        "modes": [
            "minimum_cardinality_exact",
            "subset_minimal_exact",
            "sound_overapprox",
        ],
        "min_cardinality_exact_atom_threshold": MIN_CARDINALITY_EXACT_ATOM_THRESHOLD,
        "subset_minimal_deletion_atom_threshold": SUBSET_MINIMAL_DELETION_ATOM_THRESHOLD,
        "complexity": {
            "minimum_cardinality_exact": "O(C(n,k)) brute force over atom universe",
            "subset_minimal_exact": "O(n^2) deterministic deletion from sorted union",
            "sound_overapprox": "O(terminals) one atom per terminal",
        },
    }


def replay_obstruction_core(
    core: DomainObstructionCoreV1,
    *,
    certificate: SupportCertificate,
    terminal_records: tuple[GoalTerminalEvidenceV1, ...],
    profile: GoalVerifierProfileV1,
    state: FiniteDomainState,
    expander: Any,
    verifier: Any,
) -> tuple[bool, tuple[str, ...]]:
    """Replay base certificate then validate obstruction core identities."""
    base_replay = replay_support_certificate(
        certificate,
        state=state,
        expander=expander,
        verifier=verifier,
    )
    ok, violations = validate_domain_obstruction_core(
        core,
        certificate=certificate,
        terminal_records=terminal_records,
        profile=profile,
        state=state,
        replay_ok=base_replay.ok,
    )
    merged = list(violations)
    if not base_replay.ok:
        merged.extend(base_replay.violations)
    return ok and base_replay.ok, tuple(merged)
