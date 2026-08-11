"""Finite structural mapping for PGS-F01 goal-support Lean laws.

Mirrors ``LeverProofLean.GoalSupport`` definitions against Python
``goal_support_domain_adequacy`` / ``goal_support_obstruction`` contracts.
Does **not** prove OpenUI verifier meaning or NL correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PartitionName = Literal["supported", "unsupported", "unknown", "unobserved"]
DomainAdequacyName = Literal[
    "domain_adequate_selected_supported",
    "selection_regret",
    "selection_unresolved",
    "domain_inadequate_under_bounds",
    "coverage_unknown",
]
ObstructionMode = Literal[
    "minimum_cardinality_exact",
    "subset_minimal_exact",
    "sound_overapprox",
]

GOAL_SUPPORT_FORMAL_SCHEMA = "goal_support_formal_mapping/v1"


@dataclass(frozen=True)
class PartitionsV1:
    legal: tuple[int, ...]
    supported: tuple[int, ...]
    unsupported: tuple[int, ...]
    unknown: tuple[int, ...]
    unobserved: tuple[int, ...]

    def cover(self) -> tuple[int, ...]:
        return self.supported + self.unsupported + self.unknown + self.unobserved


@dataclass(frozen=True)
class ActionEvidenceV1:
    action: int
    partition: PartitionName
    replay_ok: bool
    hard_profile: bool


@dataclass(frozen=True)
class ClassificationInputsV1:
    selected: int
    hard_profile: bool
    cap_applied: bool
    domain_complete: bool
    all_unsupported_replay_valid: bool
    obstruction_present: bool


@dataclass(frozen=True)
class TerminalFailureSetV1:
    terminal: int
    atoms: tuple[int, ...]


def certified_removable(evidence: ActionEvidenceV1) -> bool:
    if evidence.partition != "unsupported":
        return False
    return evidence.replay_ok and evidence.hard_profile


def certified_removal_set(
    partitions: PartitionsV1, evidence: tuple[ActionEvidenceV1, ...]
) -> tuple[int, ...]:
    by_action = {item.action: item for item in evidence}
    removed: list[int] = []
    for action in partitions.unsupported:
        item = by_action.get(action)
        if item is not None and certified_removable(item):
            removed.append(action)
    return tuple(removed)


def classify_domain_adequacy(
    partitions: PartitionsV1, inputs: ClassificationInputsV1
) -> DomainAdequacyName:
    if partitions.supported:
        if inputs.selected in partitions.supported:
            return "domain_adequate_selected_supported"
        if inputs.selected in partitions.unsupported:
            return "selection_regret"
        if inputs.selected in partitions.unknown or inputs.selected in partitions.unobserved:
            return "selection_unresolved"
        return "coverage_unknown"
    if (
        not partitions.unknown
        and not partitions.unobserved
        and partitions.unsupported
        and inputs.all_unsupported_replay_valid
        and inputs.hard_profile
        and not inputs.cap_applied
        and inputs.domain_complete
    ):
        return "domain_inadequate_under_bounds"
    return "coverage_unknown"


def hits_all(core: tuple[int, ...], failures: tuple[TerminalFailureSetV1, ...]) -> bool:
    core_set = set(core)
    return all(core_set & set(item.atoms) for item in failures)


def subset_minimal_exact(
    core: tuple[int, ...], failures: tuple[TerminalFailureSetV1, ...]
) -> bool:
    if not hits_all(core, failures):
        return False
    for atom in core:
        trial = tuple(item for item in core if item != atom)
        if trial and hits_all(trial, failures):
            return False
    return True


def is_witness(satisfied: tuple[int, ...], required: tuple[int, ...]) -> bool:
    return set(required).issubset(set(satisfied))


def apply_certified_removal(
    live: tuple[int, ...], removed: tuple[int, ...]
) -> tuple[int, ...]:
    rem = set(removed)
    return tuple(item for item in live if item not in rem)


def validate_well_formed_partitions(partitions: PartitionsV1) -> list[str]:
    violations: list[str] = []
    buckets = (
        partitions.supported,
        partitions.unsupported,
        partitions.unknown,
        partitions.unobserved,
    )
    seen: set[int] = set()
    for name, bucket in zip(
        ("supported", "unsupported", "unknown", "unobserved"), buckets, strict=True
    ):
        if len(bucket) != len(set(bucket)):
            violations.append(f"{name} contains duplicate action ids")
        overlap = seen & set(bucket)
        if overlap:
            violations.append(f"partition overlap on {sorted(overlap)}")
        seen.update(bucket)
        if bucket != tuple(sorted(bucket)):
            violations.append(f"{name} must be canonically sorted")
    if set(partitions.cover()) != set(partitions.legal):
        violations.append("partition cover != legal set")
    return violations


def serialize_structural_case(
    *,
    partitions: PartitionsV1,
    evidence: tuple[ActionEvidenceV1, ...],
    inputs: ClassificationInputsV1,
    failures: tuple[TerminalFailureSetV1, ...] = (),
    core: tuple[int, ...] = (),
    required_constraints: tuple[int, ...] = (),
    satisfied_constraints: tuple[int, ...] = (),
    live: tuple[int, ...] = (),
    removed: tuple[int, ...] = (),
) -> dict[str, Any]:
    return {
        "schema": GOAL_SUPPORT_FORMAL_SCHEMA,
        "partitions": {
            "legal": list(partitions.legal),
            "supported": list(partitions.supported),
            "unsupported": list(partitions.unsupported),
            "unknown": list(partitions.unknown),
            "unobserved": list(partitions.unobserved),
        },
        "evidence": [
            {
                "action": item.action,
                "partition": item.partition,
                "replay_ok": item.replay_ok,
                "hard_profile": item.hard_profile,
            }
            for item in evidence
        ],
        "inputs": {
            "selected": inputs.selected,
            "hard_profile": inputs.hard_profile,
            "cap_applied": inputs.cap_applied,
            "domain_complete": inputs.domain_complete,
            "all_unsupported_replay_valid": inputs.all_unsupported_replay_valid,
            "obstruction_present": inputs.obstruction_present,
        },
        "failures": [
            {"terminal": item.terminal, "atoms": list(item.atoms)} for item in failures
        ],
        "core": list(core),
        "required_constraints": list(required_constraints),
        "satisfied_constraints": list(satisfied_constraints),
        "live": list(live),
        "removed": list(removed),
    }


def evaluate_structural_case(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != GOAL_SUPPORT_FORMAL_SCHEMA:
        raise ValueError("unsupported goal_support_formal_mapping schema")
    parts_raw = payload["partitions"]
    partitions = PartitionsV1(
        legal=tuple(parts_raw["legal"]),
        supported=tuple(parts_raw["supported"]),
        unsupported=tuple(parts_raw["unsupported"]),
        unknown=tuple(parts_raw["unknown"]),
        unobserved=tuple(parts_raw["unobserved"]),
    )
    evidence = tuple(
        ActionEvidenceV1(
            action=int(item["action"]),
            partition=item["partition"],
            replay_ok=bool(item["replay_ok"]),
            hard_profile=bool(item["hard_profile"]),
        )
        for item in payload.get("evidence", [])
    )
    inputs_raw = payload["inputs"]
    inputs = ClassificationInputsV1(
        selected=int(inputs_raw["selected"]),
        hard_profile=bool(inputs_raw["hard_profile"]),
        cap_applied=bool(inputs_raw["cap_applied"]),
        domain_complete=bool(inputs_raw["domain_complete"]),
        all_unsupported_replay_valid=bool(inputs_raw["all_unsupported_replay_valid"]),
        obstruction_present=bool(inputs_raw["obstruction_present"]),
    )
    failures = tuple(
        TerminalFailureSetV1(
            terminal=int(item["terminal"]),
            atoms=tuple(int(atom) for atom in item["atoms"]),
        )
        for item in payload.get("failures", [])
    )
    core = tuple(int(atom) for atom in payload.get("core", []))
    required = tuple(int(item) for item in payload.get("required_constraints", []))
    satisfied = tuple(int(item) for item in payload.get("satisfied_constraints", []))
    live = tuple(int(item) for item in payload.get("live", []))
    removed = tuple(int(item) for item in payload.get("removed", []))
    return {
        "partition_violations": validate_well_formed_partitions(partitions),
        "certified_removal": list(certified_removal_set(partitions, evidence)),
        "classification": classify_domain_adequacy(partitions, inputs),
        "hits_all": hits_all(core, failures) if failures else None,
        "subset_minimal_exact": subset_minimal_exact(core, failures) if failures else None,
        "witness_ok": is_witness(satisfied, required) if required else None,
        "witness_ok_after_required_subset": (
            is_witness(satisfied, required[:-1])
            if len(required) > 1
            else is_witness(satisfied, required)
        )
        if required
        else None,
        "survivors": list(apply_certified_removal(live, removed)) if live else [],
    }


__all__ = [
    "GOAL_SUPPORT_FORMAL_SCHEMA",
    "ActionEvidenceV1",
    "ClassificationInputsV1",
    "DomainAdequacyName",
    "ObstructionMode",
    "PartitionsV1",
    "TerminalFailureSetV1",
    "apply_certified_removal",
    "certified_removal_set",
    "certified_removable",
    "classify_domain_adequacy",
    "evaluate_structural_case",
    "hits_all",
    "is_witness",
    "serialize_structural_case",
    "subset_minimal_exact",
    "validate_well_formed_partitions",
]
