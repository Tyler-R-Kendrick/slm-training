"""Factor-wise oracle substitution for SemanticPlanV1.

``PlanInterventionRecordV1``/``apply_plan_intervention`` extend this module
into a governed, matched paired-comparison harness (VCE-004): they capture
an immutable, replayable before/after observation of one
``PlanOracleSubstitutor.apply`` call -- declared vs. observed changed
factors, frozen per-comparison identity, and caller-supplied downstream
evidence -- without introducing a second substitution authority or touching
compiler/verifier legality.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Sequence

from slm_training.data.progspec.semantic_evidence import canonical_json
from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanBinding,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)

PlanSource = Literal["none", "predicted", "gold"]
OracleFactor = Literal["archetype", "roles", "topology", "bindings"]
UseMode = Literal["seed", "features", "soft_bias", "certified_restrictions"]

_ALLOWED_FACTORS: set[str] = {"archetype", "roles", "topology", "bindings"}
_FACTOR_FIELDS: dict[OracleFactor, str] = {
    "archetype": "archetype",
    "roles": "role_slots",
    "topology": "topology",
    "bindings": "bindings",
}


class PlanOracleSubstitutor:
    """Fail-closed factor-wise oracle substitution for ceiling experiments.

    Gold plans are accepted only when ``honesty_mode == "oracle_diagnostic"``.
    Oracle arms must never enter a production/ship manifest.
    """

    def __init__(
        self,
        *,
        plan_source: PlanSource = "none",
        oracle_factors: tuple[OracleFactor, ...] | None = None,
        use_mode: UseMode = "seed",
        honesty_mode: str = "production",
    ) -> None:
        if plan_source not in {"none", "predicted", "gold"}:
            raise ValueError(f"invalid plan_source: {plan_source!r}")
        if use_mode not in {"seed", "features", "soft_bias", "certified_restrictions"}:
            raise ValueError(f"invalid use_mode: {use_mode!r}")
        if honesty_mode not in {"production", "oracle_diagnostic"}:
            raise ValueError(f"invalid honesty_mode: {honesty_mode!r}")
        if oracle_factors is not None:
            bad = set(oracle_factors) - _ALLOWED_FACTORS
            if bad:
                raise ValueError(f"invalid oracle factors: {sorted(bad)}")
        self.plan_source = plan_source
        self.oracle_factors = tuple(oracle_factors) if oracle_factors else ()
        self.use_mode = use_mode
        self.honesty_mode = honesty_mode

    def apply(self, baseline: SemanticPlanV1, oracle: SemanticPlanV1) -> SemanticPlanV1:
        """Return a plan with selected oracle factors substituted into baseline.

        Unknown factors preserve baseline behavior. Gold source is rejected
        outside oracle_diagnostic mode.
        """
        if self.plan_source == "none":
            return baseline
        if self.plan_source == "gold" and self.honesty_mode != "oracle_diagnostic":
            raise ValueError(
                "gold plan source requires honesty_mode=oracle_diagnostic"
            )
        if self.plan_source == "predicted" and oracle.identity.provenance == "gold":
            raise ValueError(
                "predicted plan source cannot consume a gold oracle plan"
            )

        updates: dict[str, Any] = {}
        for factor in self.oracle_factors:
            if factor == "archetype":
                updates["archetype"] = PlanArchetype(
                    id=oracle.archetype.id,
                    distribution=oracle.archetype.distribution,
                    confidence=oracle.archetype.confidence,
                )
            elif factor == "roles":
                updates["role_slots"] = tuple(
                    RoleSlot(
                        role_id=slot.role_id,
                        component_family=slot.component_family,
                        candidate_distribution=slot.candidate_distribution,
                        min_cardinality=slot.min_cardinality,
                        max_cardinality=slot.max_cardinality,
                        required=slot.required,
                        evidence_spans=slot.evidence_spans,
                    )
                    for slot in oracle.role_slots
                )
            elif factor == "topology":
                updates["topology"] = PlanTopology(
                    parent_relation_candidates=oracle.topology.parent_relation_candidates,
                    sibling_order_groups=oracle.topology.sibling_order_groups,
                    depth_bounds=oracle.topology.depth_bounds,
                    cardinality_bounds=oracle.topology.cardinality_bounds,
                    partial_order_constraints=oracle.topology.partial_order_constraints,
                )
            elif factor == "bindings":
                updates["bindings"] = tuple(
                    PlanBinding(
                        role_slot_id=binding.role_slot_id,
                        candidate_symbols=binding.candidate_symbols,
                        placeholder_fallback=binding.placeholder_fallback,
                    )
                    for binding in oracle.bindings
                )
        if not updates:
            return baseline
        return baseline.model_copy(update=updates)

    def contamination_banner(self) -> str | None:
        """Banner to attach to any artifact produced by an oracle arm."""
        if self.plan_source in {"gold", "oracle_override"}:
            return (
                f"ORACLE_DIAGNOSTIC: plan_source={self.plan_source} "
                f"factors={list(self.oracle_factors)} use_mode={self.use_mode}"
            )
        return None


def _plan_hash(plan: SemanticPlanV1) -> str:
    return hashlib.sha256(canonical_json(plan.to_dict())).hexdigest()


def _changed_factors(
    baseline: SemanticPlanV1, mutated: SemanticPlanV1
) -> tuple[OracleFactor, ...]:
    """Return the factors that actually differ between ``baseline`` and ``mutated``.

    Only ever inspects the four substitutable fields ``apply`` can touch --
    never a proxy for legal/grammar support.
    """
    return tuple(
        factor
        for factor, field in _FACTOR_FIELDS.items()
        if getattr(baseline, field) != getattr(mutated, field)
    )


_INTERVENTION_IDENTITY_FIELDS = (
    "model_id",
    "request_fingerprint",
    "candidate_budget",
    "search_budget",
    "seed",
    "verifier_version",
    "hardware_id",
)


@dataclass(frozen=True)
class InterventionIdentityV1:
    """Identities frozen for one paired baseline/intervention comparison.

    Every field defaults to ``None`` -- an identity the caller did not
    supply stays explicitly absent rather than being fabricated, mirroring
    ``DecodeIdentityV1``'s no-fabrication contract.
    """

    model_id: str | None = None
    request_fingerprint: str | None = None
    candidate_budget: int | None = None
    search_budget: int | None = None
    seed: int | None = None
    verifier_version: str | None = None
    hardware_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterventionIdentityV1":
        return cls(**{key: data.get(key) for key in _INTERVENTION_IDENTITY_FIELDS})


INTERVENTION_RECORD_SCHEMA_VERSION = "plan_intervention_record/v1"


def _intervention_record_digest(record: "PlanInterventionRecordV1") -> str:
    payload = record.to_dict()
    payload.pop("record_hash", None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class PlanInterventionRecordV1:
    """Immutable, replayable paired baseline/intervention observation.

    Produced only by ``apply_plan_intervention``. Records exactly what
    VCE-004 requires: before/after plan hashes, the declared vs. observed
    changed-factor set, the frozen per-comparison identity, and
    caller-supplied downstream-decision/quality/compute evidence. It never
    widens legal support and never itself authorizes production use of an
    intervened plan -- callers still route the intervened plan through
    ``SemanticPlanV1.to_production_dict`` (fail-closed on oracle-only
    provenance) or ``filter_manifest_safe`` before any manifest.

    ``PlanOracleSubstitutor.apply`` intentionally never overwrites
    ``identity`` -- an intervened plan keeps the *baseline's* provenance, so
    ``SemanticPlanV1.is_oracle_only`` on the mutated plan alone cannot be
    trusted to detect oracle contamination. ``contamination_banner`` here
    (derived from the substitutor's declared ``plan_source``, not the
    mutated plan) is the authoritative signal; always gate manifest
    inclusion on the record, via ``filter_manifest_safe``, not on the raw
    mutated plan.
    """

    schema_version: str
    plan_source: str
    declared_factors: tuple[str, ...]
    changed_factors: tuple[str, ...]
    baseline_plan_hash: str
    intervened_plan_hash: str
    identity: dict[str, Any]
    downstream_decision: dict[str, Any]
    quality_metrics: dict[str, Any]
    compute: dict[str, Any]
    contamination_banner: str | None
    record_hash: str

    def touches_only_declared_factors(self) -> bool:
        """True when no factor outside ``declared_factors`` was mutated."""
        return set(self.changed_factors) <= set(self.declared_factors)

    def is_no_op(self) -> bool:
        """True when the intervened plan is hash-identical to the baseline."""
        return self.baseline_plan_hash == self.intervened_plan_hash

    @property
    def is_contaminated(self) -> bool:
        return self.contamination_banner is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_source": self.plan_source,
            "declared_factors": list(self.declared_factors),
            "changed_factors": list(self.changed_factors),
            "baseline_plan_hash": self.baseline_plan_hash,
            "intervened_plan_hash": self.intervened_plan_hash,
            "identity": dict(self.identity),
            "downstream_decision": dict(self.downstream_decision),
            "quality_metrics": dict(self.quality_metrics),
            "compute": dict(self.compute),
            "contamination_banner": self.contamination_banner,
            "record_hash": self.record_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanInterventionRecordV1":
        if data.get("schema_version") != INTERVENTION_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported plan intervention record schema: {data.get('schema_version')!r}"
            )
        record = cls(
            schema_version=data.get("schema_version", INTERVENTION_RECORD_SCHEMA_VERSION),
            plan_source=data.get("plan_source", "none"),
            declared_factors=tuple(data.get("declared_factors", [])),
            changed_factors=tuple(data.get("changed_factors", [])),
            baseline_plan_hash=data.get("baseline_plan_hash", ""),
            intervened_plan_hash=data.get("intervened_plan_hash", ""),
            identity=dict(data.get("identity", {})),
            downstream_decision=dict(data.get("downstream_decision", {})),
            quality_metrics=dict(data.get("quality_metrics", {})),
            compute=dict(data.get("compute", {})),
            contamination_banner=data.get("contamination_banner"),
            record_hash=data.get("record_hash", ""),
        )
        if _intervention_record_digest(record) != record.record_hash:
            raise ValueError(
                "plan intervention record hash mismatch (tampered or corrupted payload)"
            )
        return record


def apply_plan_intervention(
    substitutor: PlanOracleSubstitutor,
    baseline: SemanticPlanV1,
    oracle: SemanticPlanV1,
    *,
    identity: InterventionIdentityV1 | None = None,
    downstream_decision: dict[str, Any] | None = None,
    quality_metrics: dict[str, Any] | None = None,
    compute: dict[str, Any] | None = None,
) -> PlanInterventionRecordV1:
    """Apply ``substitutor`` and capture an immutable paired observation.

    Pure: never mutates ``baseline``/``oracle`` (``PlanOracleSubstitutor.apply``
    already returns a new plan via ``model_copy``), never touches
    compiler/verifier legality, and never authorizes production use of the
    resulting plan by itself.
    """
    mutated = substitutor.apply(baseline, oracle)
    record = PlanInterventionRecordV1(
        schema_version=INTERVENTION_RECORD_SCHEMA_VERSION,
        plan_source=substitutor.plan_source,
        declared_factors=substitutor.oracle_factors,
        changed_factors=_changed_factors(baseline, mutated),
        baseline_plan_hash=_plan_hash(baseline),
        intervened_plan_hash=_plan_hash(mutated),
        identity=(identity or InterventionIdentityV1()).to_dict(),
        downstream_decision=dict(downstream_decision or {}),
        quality_metrics=dict(quality_metrics or {}),
        compute=dict(compute or {}),
        contamination_banner=substitutor.contamination_banner(),
        record_hash="",
    )
    return replace(record, record_hash=_intervention_record_digest(record))


def intervention_record_integrity_ok(record: PlanInterventionRecordV1) -> bool:
    """Recompute the record hash and compare; used by replay/tamper checks."""
    return _intervention_record_digest(record) == record.record_hash


def filter_manifest_safe(
    records: Sequence[PlanInterventionRecordV1],
) -> tuple[PlanInterventionRecordV1, ...]:
    """Drop every contamination-bannered record -- oracle arms never ship."""
    return tuple(record for record in records if not record.is_contaminated)
