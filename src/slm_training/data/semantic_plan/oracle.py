"""Factor-wise oracle substitution for SemanticPlanV1.

``PlanInterventionRecordV1``/``apply_plan_intervention`` extend this module
into a governed, matched paired-comparison harness (VCE-004): they capture
an immutable, replayable before/after observation of one
``PlanOracleSubstitutor.apply`` call -- declared vs. observed changed
factors, frozen per-comparison identity, and caller-supplied downstream
evidence -- without introducing a second substitution authority or touching
compiler/verifier legality.

VCE-005 adds the remaining control arms (``destructive``/``random`` plan
sources, ``select_shuffled_oracle``, ``build_baseline_intervention``) so an
oracle-localization result cannot be mistaken for a generic extra-compute,
leakage, or intervention-pipeline effect: every arm below still routes
through the same ``PlanOracleSubstitutor.apply`` / ``apply_plan_intervention``
/ ``PlanInterventionRecordV1`` machinery -- no second substitution authority,
no new record schema.
"""

from __future__ import annotations

import hashlib
import random
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

PlanSource = Literal["none", "predicted", "gold", "destructive", "random"]
OracleFactor = Literal["archetype", "roles", "topology", "bindings"]
UseMode = Literal["seed", "features", "soft_bias", "certified_restrictions"]

_ALLOWED_FACTORS: set[str] = {"archetype", "roles", "topology", "bindings"}
_FACTOR_FIELDS: dict[OracleFactor, str] = {
    "archetype": "archetype",
    "roles": "role_slots",
    "topology": "topology",
    "bindings": "bindings",
}
_SYNTHETIC_SOURCES: set[str] = {"destructive", "random"}


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
        if plan_source not in {"none", "predicted", "gold", "destructive", "random"}:
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

    def apply(
        self,
        baseline: SemanticPlanV1,
        oracle: SemanticPlanV1 | None = None,
        *,
        rng_seed: int = 0,
    ) -> SemanticPlanV1:
        """Return a plan with selected oracle factors substituted into baseline.

        Unknown factors preserve baseline behavior. Gold source is rejected
        outside oracle_diagnostic mode. ``plan_source in {"destructive",
        "random"}`` ignores ``oracle`` entirely and instead permutes each
        declared factor's *existing* content within ``baseline`` (see
        ``_synthetic_factor_value``) -- a wrong-but-structurally-matched
        value, never content read from any real oracle/other record.
        """
        if self.plan_source == "none":
            return baseline
        if self.plan_source == "gold" and self.honesty_mode != "oracle_diagnostic":
            raise ValueError(
                "gold plan source requires honesty_mode=oracle_diagnostic"
            )
        if self.plan_source in _SYNTHETIC_SOURCES:
            updates: dict[str, Any] = {}
            for factor in self.oracle_factors:
                value = _synthetic_factor_value(
                    factor,
                    baseline,
                    destructive=self.plan_source == "destructive",
                    rng_seed=rng_seed,
                )
                if value is not None:
                    updates[_FACTOR_FIELDS[factor]] = value
            if not updates:
                return baseline
            return baseline.model_copy(update=updates)
        if oracle is None:
            raise ValueError(f"plan_source={self.plan_source!r} requires an oracle plan")
        if self.plan_source == "predicted" and oracle.identity.provenance == "gold":
            raise ValueError(
                "predicted plan source cannot consume a gold oracle plan"
            )

        updates = {}
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


_ALTERNATE_ARCHETYPE_IDS = ("stack_column", "stack_row", "form", "card", "list", "grid")


def _synthetic_archetype(
    baseline: SemanticPlanV1, *, destructive: bool, rng_seed: int
) -> PlanArchetype | None:
    original_id = baseline.archetype.id
    if destructive:
        new_id = "mismatched_archetype" if original_id != "mismatched_archetype" else "mismatched_archetype_alt"
    else:
        rng = random.Random(rng_seed)
        candidates = [c for c in _ALTERNATE_ARCHETYPE_IDS if c != original_id]
        if not candidates:
            return None
        new_id = candidates[rng.randrange(len(candidates))]
    return PlanArchetype(
        id=new_id,
        distribution=baseline.archetype.distribution,
        confidence=baseline.archetype.confidence,
    )


def _permute(values: list[Any], *, destructive: bool, rng_seed: int) -> list[Any] | None:
    """A derangement-like reordering of ``values`` in place-count -- never new content.

    ``destructive`` rotates by one position (deterministic, reproducible
    without a seed); otherwise a seeded random shuffle is used. Returns
    ``None`` when fewer than two values exist (nothing to permute) or the
    only achievable reordering happens to equal the original order.
    """
    if len(values) < 2:
        return None
    if destructive:
        permuted = values[1:] + values[:1]
    else:
        rng = random.Random(rng_seed)
        permuted = list(values)
        rng.shuffle(permuted)
        if permuted == values:
            permuted = values[1:] + values[:1]
    return None if permuted == values else permuted


def _synthetic_role_slots(
    baseline: SemanticPlanV1, *, destructive: bool, rng_seed: int
) -> tuple[RoleSlot, ...] | None:
    slots = list(baseline.role_slots)
    families = [slot.component_family for slot in slots]
    permuted = _permute(families, destructive=destructive, rng_seed=rng_seed)
    if permuted is None:
        return None
    return tuple(
        RoleSlot(
            role_id=slot.role_id,
            component_family=new_family,
            candidate_distribution=slot.candidate_distribution,
            min_cardinality=slot.min_cardinality,
            max_cardinality=slot.max_cardinality,
            required=slot.required,
            evidence_spans=slot.evidence_spans,
        )
        for slot, new_family in zip(slots, permuted)
    )


def _synthetic_topology(
    baseline: SemanticPlanV1, *, destructive: bool, rng_seed: int
) -> PlanTopology | None:
    edges = baseline.topology.parent_relation_candidates
    if not edges:
        return None
    parents = [str(edge.get("parent_role_id") or "") for edge in edges]
    permuted = _permute(parents, destructive=destructive, rng_seed=rng_seed)
    if permuted is None:
        return None
    new_edges = tuple(
        {**edge, "parent_role_id": new_parent}
        for edge, new_parent in zip(edges, permuted)
    )
    return PlanTopology(
        parent_relation_candidates=new_edges,
        sibling_order_groups=baseline.topology.sibling_order_groups,
        depth_bounds=baseline.topology.depth_bounds,
        cardinality_bounds=baseline.topology.cardinality_bounds,
        partial_order_constraints=baseline.topology.partial_order_constraints,
    )


def _synthetic_bindings(
    baseline: SemanticPlanV1, *, destructive: bool, rng_seed: int
) -> tuple[PlanBinding, ...] | None:
    bindings = list(baseline.bindings)
    symbol_sets = [binding.candidate_symbols for binding in bindings]
    permuted = _permute(symbol_sets, destructive=destructive, rng_seed=rng_seed)
    if permuted is None:
        return None
    return tuple(
        PlanBinding(
            role_slot_id=binding.role_slot_id,
            candidate_symbols=new_symbols,
            placeholder_fallback=binding.placeholder_fallback,
        )
        for binding, new_symbols in zip(bindings, permuted)
    )


def _synthetic_factor_value(
    factor: OracleFactor, baseline: SemanticPlanV1, *, destructive: bool, rng_seed: int
) -> Any | None:
    """A wrong-but-structurally-matched value for ``factor``.

    Always a *permutation* of ``baseline``'s own existing content (same
    element count, same serialization shape) -- never new content sourced
    from anywhere else -- so a destructive/random arm's compute and
    serialization cost matches a real one-factor substitution of the same
    factor. Returns ``None`` when the factor has too little structure to
    permute (e.g. a single role slot); ``apply()`` then leaves that factor
    unchanged, matching the existing "unknown factors preserve baseline
    behavior" convention rather than fabricating content.
    """
    if factor == "archetype":
        return _synthetic_archetype(baseline, destructive=destructive, rng_seed=rng_seed)
    if factor == "roles":
        return _synthetic_role_slots(baseline, destructive=destructive, rng_seed=rng_seed)
    if factor == "topology":
        return _synthetic_topology(baseline, destructive=destructive, rng_seed=rng_seed)
    if factor == "bindings":
        return _synthetic_bindings(baseline, destructive=destructive, rng_seed=rng_seed)
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
    oracle: SemanticPlanV1 | None = None,
    *,
    rng_seed: int = 0,
    identity: InterventionIdentityV1 | None = None,
    downstream_decision: dict[str, Any] | None = None,
    quality_metrics: dict[str, Any] | None = None,
    compute: dict[str, Any] | None = None,
) -> PlanInterventionRecordV1:
    """Apply ``substitutor`` and capture an immutable paired observation.

    ``oracle`` is optional and ``rng_seed`` is unused for every arm except
    ``plan_source in {"destructive", "random"}``, which ignore ``oracle`` and
    synthesize a permuted, structurally-matched wrong value from ``baseline``
    itself (see ``PlanOracleSubstitutor.apply``).

    Pure: never mutates ``baseline``/``oracle`` (``PlanOracleSubstitutor.apply``
    already returns a new plan via ``model_copy``), never touches
    compiler/verifier legality, and never authorizes production use of the
    resulting plan by itself.
    """
    mutated = substitutor.apply(baseline, oracle, rng_seed=rng_seed)
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


def _compatibility_key(plan: SemanticPlanV1) -> tuple[str, int, int, int]:
    """Content-derived compatibility key for pairing a shuffled oracle.

    Deliberately reads only plan *content* (archetype id, structure sizes)
    -- never ``identity.source_program_fingerprint``/``prompt_context_hash``
    -- so a shuffled control cannot access the target record's identity
    through those hidden fields.
    """
    return (
        plan.archetype.id or "",
        len(plan.role_slots),
        len(plan.symbols),
        len(plan.bindings),
    )


def select_shuffled_oracle(
    baseline: SemanticPlanV1,
    candidates: Sequence[SemanticPlanV1],
    *,
    rng_seed: int = 0,
) -> SemanticPlanV1 | None:
    """Pick a different, content-compatible plan from ``candidates`` to use as a shuffled oracle.

    Compatibility is judged only by ``_compatibility_key`` (content shape),
    never by any ``identity`` field: a shuffled arm proves whether downstream
    decisions respond to *content*, not whether it can see which record a
    candidate came from. Returns ``None`` when no compatible, distinct
    candidate exists -- fails closed rather than falling back to an
    incompatible or self pairing.
    """
    key = _compatibility_key(baseline)
    baseline_dict = baseline.to_dict()
    pool = [
        candidate
        for candidate in candidates
        if _compatibility_key(candidate) == key and candidate.to_dict() != baseline_dict
    ]
    if not pool:
        return None
    return pool[rng_seed % len(pool)]


def build_baseline_intervention(
    baseline: SemanticPlanV1,
    *,
    identity: InterventionIdentityV1 | None = None,
    downstream_decision: dict[str, Any] | None = None,
    quality_metrics: dict[str, Any] | None = None,
    compute: dict[str, Any] | None = None,
) -> PlanInterventionRecordV1:
    """Explicit, first-class no-intervention control arm.

    Pays the same ``apply_plan_intervention`` bookkeeping (hashing, identity
    freezing, record construction) as every other arm, so its
    observation/telemetry overhead is genuinely matched to a real
    intervention arm -- the only difference is that
    ``PlanOracleSubstitutor(plan_source="none")`` returns ``baseline`` itself,
    so ``is_no_op()`` is always true for the returned record.
    """
    substitutor = PlanOracleSubstitutor(plan_source="none")
    return apply_plan_intervention(
        substitutor,
        baseline,
        baseline,
        identity=identity,
        downstream_decision=downstream_decision,
        quality_metrics=quality_metrics,
        compute=compute,
    )
