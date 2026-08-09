"""Tests for oracle substitution fail-closed behavior."""

from __future__ import annotations

import dataclasses

import pytest

from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanBinding,
    PlanIdentity,
    PlanSymbol,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.data.semantic_plan.oracle import (
    InterventionIdentityV1,
    PlanInterventionRecordV1,
    PlanOracleSubstitutor,
    apply_plan_intervention,
    build_baseline_intervention,
    filter_manifest_safe,
    intervention_record_integrity_ok,
    select_shuffled_oracle,
)


def _baseline_plan() -> SemanticPlanV1:
    return SemanticPlanV1(
        identity=PlanIdentity(pack_id="openui", provenance="predicted"),
        archetype=PlanArchetype(id="stack", confidence=0.6),
        role_slots=(RoleSlot(role_id="r1", component_family="Stack"),),
        topology=PlanTopology(),
    )


def _oracle_plan() -> SemanticPlanV1:
    return SemanticPlanV1(
        identity=PlanIdentity(pack_id="openui", provenance="gold"),
        archetype=PlanArchetype(id="card", confidence=1.0),
        role_slots=(
            RoleSlot(role_id="r1", component_family="Card"),
            RoleSlot(role_id="r2", component_family="TextContent"),
        ),
        topology=PlanTopology(
            parent_relation_candidates=(
                {"parent_role_id": "r1", "child_role_id": "r2", "relation": "contains"},
            )
        ),
        symbols=(PlanSymbol(symbol_id="s1", semantic_role="text"),),
        bindings=(PlanBinding(role_slot_id="r2", candidate_symbols=("s1",)),),
    )


def _rich_plan(*, fingerprint: str | None = None, prompt_context_hash: str | None = None) -> SemanticPlanV1:
    """A plan with >=2 elements per factor, so destructive/random permutation
    (which needs something to reorder) and shuffled-compatibility tests have
    real structure to work with, unlike ``_baseline_plan``'s single role."""
    return SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui",
            provenance="predicted",
            source_program_fingerprint=fingerprint,
            prompt_context_hash=prompt_context_hash,
        ),
        archetype=PlanArchetype(id="stack_column", confidence=0.8),
        role_slots=(
            RoleSlot(role_id="root", component_family="Stack"),
            RoleSlot(role_id="r1", component_family="TextContent"),
            RoleSlot(role_id="r2", component_family="Button"),
        ),
        topology=PlanTopology(
            parent_relation_candidates=(
                # Distinct parents (root, then r1) -- not two edges under the
                # same parent -- so permuting parent_role_id is a genuine,
                # observable change rather than a no-op on identical values.
                {"parent_role_id": "root", "child_role_id": "r1", "relation": "contains"},
                {"parent_role_id": "r1", "child_role_id": "r2", "relation": "contains"},
            )
        ),
        symbols=(
            PlanSymbol(symbol_id="s1", semantic_role="text"),
            PlanSymbol(symbol_id="s2", semantic_role="label"),
        ),
        bindings=(
            PlanBinding(role_slot_id="r1", candidate_symbols=("s1",)),
            PlanBinding(role_slot_id="r2", candidate_symbols=("s2",)),
        ),
    )


def test_none_source_returns_baseline() -> None:
    baseline = _baseline_plan()
    oracle = _oracle_plan()
    subst = PlanOracleSubstitutor(plan_source="none", oracle_factors=("archetype",))
    result = subst.apply(baseline, oracle)
    assert result == baseline


def test_gold_source_rejected_in_production() -> None:
    subst = PlanOracleSubstitutor(plan_source="gold", oracle_factors=("archetype",))
    with pytest.raises(ValueError, match="gold plan source requires"):
        subst.apply(_baseline_plan(), _oracle_plan())


def test_gold_source_allowed_in_diagnostic_mode() -> None:
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype",),
        honesty_mode="oracle_diagnostic",
    )
    result = subst.apply(_baseline_plan(), _oracle_plan())
    assert result.archetype.id == "card"


def test_predicted_source_cannot_consume_gold_oracle() -> None:
    subst = PlanOracleSubstitutor(plan_source="predicted", oracle_factors=("archetype",))
    with pytest.raises(ValueError, match="cannot consume a gold oracle"):
        subst.apply(_baseline_plan(), _oracle_plan())


def test_factor_wise_substitution() -> None:
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype", "topology"),
        honesty_mode="oracle_diagnostic",
    )
    result = subst.apply(_baseline_plan(), _oracle_plan())
    assert result.archetype.id == "card"
    assert result.topology.parent_relation_candidates
    assert len(result.role_slots) == 1  # roles not substituted
    assert not result.bindings  # bindings not substituted


def test_contamination_banner_for_gold() -> None:
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype",),
        use_mode="features",
        honesty_mode="oracle_diagnostic",
    )
    banner = subst.contamination_banner()
    assert banner is not None
    assert "ORACLE_DIAGNOSTIC" in banner
    assert "features" in banner


def test_no_banner_for_predicted() -> None:
    subst = PlanOracleSubstitutor(plan_source="predicted", oracle_factors=())
    assert subst.contamination_banner() is None


def test_intervention_touches_only_declared_factors() -> None:
    """The oracle plan differs on every substitutable factor, but only one
    factor is declared -- the intervention must not leak the others."""
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype",),
        honesty_mode="oracle_diagnostic",
    )
    baseline = _baseline_plan()
    oracle = _oracle_plan()
    record = apply_plan_intervention(subst, baseline, oracle)

    assert record.changed_factors == ("archetype",)
    assert record.declared_factors == ("archetype",)
    assert record.touches_only_declared_factors()
    assert not record.is_no_op()


def test_full_factor_intervention_changes_exactly_declared_set() -> None:
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype", "roles", "topology", "bindings"),
        honesty_mode="oracle_diagnostic",
    )
    baseline = _baseline_plan()
    oracle = _oracle_plan()
    record = apply_plan_intervention(subst, baseline, oracle)

    assert set(record.changed_factors) == set(record.declared_factors)
    assert record.touches_only_declared_factors()


def test_declaring_a_factor_that_is_already_equal_is_not_a_fabricated_change() -> None:
    """Declaring a factor whose oracle value equals baseline must not report
    a spurious change -- changed_factors always stays a true diff."""
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("roles",),
        honesty_mode="oracle_diagnostic",
    )
    baseline = _baseline_plan()
    oracle_same_roles = _oracle_plan().model_copy(update={"role_slots": baseline.role_slots})
    record = apply_plan_intervention(subst, baseline, oracle_same_roles)

    assert record.changed_factors == ()
    assert record.touches_only_declared_factors()
    assert record.is_no_op()


def test_noop_intervention_is_hash_equivalent_to_baseline() -> None:
    baseline = _baseline_plan()
    oracle = _oracle_plan()
    predicted_oracle = oracle.model_copy(
        update={"identity": PlanIdentity(pack_id="openui", provenance="predicted")}
    )

    none_source = apply_plan_intervention(
        PlanOracleSubstitutor(plan_source="none", oracle_factors=("archetype",)),
        baseline,
        oracle,
    )
    assert none_source.is_no_op()
    assert none_source.baseline_plan_hash == none_source.intervened_plan_hash
    assert none_source.changed_factors == ()

    empty_factors = apply_plan_intervention(
        PlanOracleSubstitutor(plan_source="predicted", oracle_factors=()),
        baseline,
        predicted_oracle,
    )
    assert empty_factors.is_no_op()
    assert empty_factors.changed_factors == ()


def test_intervention_identity_never_fabricates_unset_fields() -> None:
    identity = InterventionIdentityV1(model_id="m1", seed=7)
    subst = PlanOracleSubstitutor(plan_source="predicted", oracle_factors=("archetype",))
    predicted_oracle = _oracle_plan().model_copy(
        update={"identity": PlanIdentity(pack_id="openui", provenance="predicted")}
    )
    record = apply_plan_intervention(
        subst, _baseline_plan(), predicted_oracle, identity=identity
    )
    assert record.identity == {
        "model_id": "m1",
        "request_fingerprint": None,
        "candidate_budget": None,
        "search_budget": None,
        "seed": 7,
        "verifier_version": None,
        "hardware_id": None,
    }

    default_record = apply_plan_intervention(subst, _baseline_plan(), predicted_oracle)
    assert all(value is None for value in default_record.identity.values())


def test_intervention_record_is_frozen_and_deterministic_on_replay() -> None:
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype", "topology"),
        honesty_mode="oracle_diagnostic",
    )
    baseline = _baseline_plan()
    oracle = _oracle_plan()

    first = apply_plan_intervention(subst, baseline, oracle)
    second = apply_plan_intervention(subst, baseline, oracle)
    assert first == second
    assert first.record_hash == second.record_hash
    assert intervention_record_integrity_ok(first)

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.changed_factors = ()  # type: ignore[misc]

    # Golden round trip.
    assert PlanInterventionRecordV1.from_dict(first.to_dict()) == first


def test_intervention_record_tamper_and_schema_rejection() -> None:
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype",),
        honesty_mode="oracle_diagnostic",
    )
    record = apply_plan_intervention(subst, _baseline_plan(), _oracle_plan())

    tampered = record.to_dict()
    tampered["changed_factors"] = []
    with pytest.raises(ValueError, match="hash mismatch"):
        PlanInterventionRecordV1.from_dict(tampered)

    wrong_schema = record.to_dict()
    wrong_schema["schema_version"] = "plan_intervention_record/v0"
    with pytest.raises(ValueError, match="unsupported plan intervention record schema"):
        PlanInterventionRecordV1.from_dict(wrong_schema)


def test_manifest_safety_excludes_oracle_contaminated_records() -> None:
    baseline = _baseline_plan()
    oracle = _oracle_plan()
    predicted_oracle = oracle.model_copy(
        update={"identity": PlanIdentity(pack_id="openui", provenance="predicted")}
    )

    contaminated = apply_plan_intervention(
        PlanOracleSubstitutor(
            plan_source="gold",
            oracle_factors=("archetype",),
            honesty_mode="oracle_diagnostic",
        ),
        baseline,
        oracle,
    )
    clean = apply_plan_intervention(
        PlanOracleSubstitutor(plan_source="predicted", oracle_factors=("archetype",)),
        baseline,
        predicted_oracle,
    )

    assert contaminated.is_contaminated
    assert not clean.is_contaminated

    safe = filter_manifest_safe([contaminated, clean])
    assert safe == (clean,)


def test_intervened_plan_identity_does_not_self_report_contamination() -> None:
    """Guards the documented gap: apply() never overwrites identity, so a
    mutated plan's own is_oracle_only cannot be trusted -- only the
    intervention record's contamination_banner can gate a manifest."""
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype",),
        honesty_mode="oracle_diagnostic",
    )
    baseline = _baseline_plan()
    oracle = _oracle_plan()
    mutated = subst.apply(baseline, oracle)
    record = apply_plan_intervention(subst, baseline, oracle)

    assert mutated.identity.provenance == "predicted"
    assert not mutated.is_oracle_only
    assert record.is_contaminated


# --------------------------------------------------------------------------
# VCE-005: destructive, random, shuffled, and explicit-baseline arms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factor", ["archetype", "roles", "topology", "bindings"])
def test_destructive_arm_changes_exactly_the_declared_factor_with_matched_shape(
    factor: str,
) -> None:
    """Destructive mutation is a permutation of baseline's own content: the
    declared factor changes, nothing else does, and structural size (role
    count / edge count / binding count) is unchanged -- proving the control
    pays comparable serialization/work cost to a real substitution."""
    baseline = _rich_plan()
    subst = PlanOracleSubstitutor(plan_source="destructive", oracle_factors=(factor,))
    record = apply_plan_intervention(subst, baseline, None)

    assert record.changed_factors == (factor,)
    assert record.declared_factors == (factor,)
    assert record.touches_only_declared_factors()
    assert not record.is_no_op()
    assert not record.is_contaminated
    assert record.contamination_banner is None

    mutated = subst.apply(baseline, None)
    assert len(mutated.role_slots) == len(baseline.role_slots)
    assert len(mutated.topology.parent_relation_candidates or ()) == len(
        baseline.topology.parent_relation_candidates or ()
    )
    assert len(mutated.bindings) == len(baseline.bindings)


def test_destructive_arm_is_deterministic() -> None:
    baseline = _rich_plan()
    subst = PlanOracleSubstitutor(plan_source="destructive", oracle_factors=("roles",))
    first = apply_plan_intervention(subst, baseline, None)
    second = apply_plan_intervention(subst, baseline, None)
    assert first.intervened_plan_hash == second.intervened_plan_hash


def test_random_arm_is_seed_sensitive_across_enough_seeds() -> None:
    """A single pair of seeds can coincide (e.g. a 2-element permutation
    space has only one non-identity ordering), so this asserts sensitivity
    across a spread of seeds rather than any two arbitrary ones."""
    baseline = _rich_plan()
    subst = PlanOracleSubstitutor(plan_source="random", oracle_factors=("archetype",))
    hashes = {
        apply_plan_intervention(subst, baseline, None, rng_seed=seed).intervened_plan_hash
        for seed in range(10)
    }
    assert len(hashes) > 1


@pytest.mark.parametrize("factor", ["archetype", "roles", "topology", "bindings"])
def test_random_arm_matches_destructive_arms_shape_and_banner(factor: str) -> None:
    baseline = _rich_plan()
    subst = PlanOracleSubstitutor(plan_source="random", oracle_factors=(factor,))
    record = apply_plan_intervention(subst, baseline, None, rng_seed=3)
    assert record.declared_factors == (factor,)
    assert record.touches_only_declared_factors()
    assert not record.is_contaminated


def test_synthetic_arms_fall_back_to_no_op_when_nothing_to_permute() -> None:
    """`_baseline_plan` has one role slot, no topology edges, no bindings --
    there is nothing to permute, so destructive/random must leave those
    factors unchanged rather than fabricate content, matching the existing
    'unknown factors preserve baseline behavior' convention."""
    baseline = _baseline_plan()
    for factor in ("roles", "topology", "bindings"):
        subst = PlanOracleSubstitutor(plan_source="destructive", oracle_factors=(factor,))
        record = apply_plan_intervention(subst, baseline, None)
        assert record.changed_factors == ()
        assert record.is_no_op()


def test_select_shuffled_oracle_finds_a_content_compatible_distinct_plan() -> None:
    baseline = _rich_plan(fingerprint="fp_baseline")
    compatible = _rich_plan(fingerprint="fp_other").model_copy(
        update={
            "role_slots": (
                RoleSlot(role_id="root", component_family="Stack"),
                RoleSlot(role_id="r1", component_family="Button"),
                RoleSlot(role_id="r2", component_family="TextContent"),
            )
        }
    )
    incompatible = _baseline_plan()  # different structure sizes

    picked = select_shuffled_oracle(baseline, [incompatible, compatible], rng_seed=0)
    assert picked is not None
    assert picked.role_slots == compatible.role_slots

    assert select_shuffled_oracle(baseline, [incompatible]) is None
    assert select_shuffled_oracle(baseline, []) is None


def test_select_shuffled_oracle_never_uses_identity_fields_to_choose() -> None:
    """Two structurally-identical plans that differ only in
    source_program_fingerprint/prompt_context_hash must still be treated as
    a valid, distinct shuffle pair -- compatibility is judged purely on
    content, so a shuffled control cannot infer or leak target identity
    through those hidden fields."""
    baseline = _rich_plan(fingerprint="fp_a", prompt_context_hash="ctx_a")
    same_content_different_identity = _rich_plan(
        fingerprint="fp_b", prompt_context_hash="ctx_b"
    )

    picked = select_shuffled_oracle(
        baseline, [same_content_different_identity], rng_seed=0
    )
    assert picked is not None
    assert picked.identity.source_program_fingerprint == "fp_b"


def test_shuffled_oracle_intervention_is_gated_by_existing_contamination_rules() -> None:
    baseline = _rich_plan(fingerprint="fp_a")
    shuffled = _rich_plan(fingerprint="fp_b").model_copy(
        update={
            "identity": PlanIdentity(pack_id="openui", provenance="gold"),
            "role_slots": (
                RoleSlot(role_id="root", component_family="Stack"),
                RoleSlot(role_id="r1", component_family="Button"),
                RoleSlot(role_id="r2", component_family="TextContent"),
            ),
        }
    )
    subst = PlanOracleSubstitutor(
        plan_source="gold", oracle_factors=("roles",), honesty_mode="oracle_diagnostic"
    )
    record = apply_plan_intervention(subst, baseline, shuffled)
    assert record.changed_factors == ("roles",)
    assert record.is_contaminated
    assert filter_manifest_safe([record]) == ()


def test_build_baseline_intervention_is_explicit_matched_telemetry_control() -> None:
    baseline = _rich_plan()
    identity = InterventionIdentityV1(model_id="m1", seed=5)
    record = build_baseline_intervention(
        baseline, identity=identity, compute={"forwards": 0, "verifier_calls": 1}
    )
    assert record.declared_factors == ()
    assert record.changed_factors == ()
    assert record.is_no_op()
    assert not record.is_contaminated
    assert record.identity["model_id"] == "m1"
    assert record.compute == {"forwards": 0, "verifier_calls": 1}


def test_new_plan_sources_round_trip_and_replay_deterministically() -> None:
    baseline = _rich_plan()
    for plan_source in ("destructive", "random"):
        subst = PlanOracleSubstitutor(plan_source=plan_source, oracle_factors=("archetype",))
        record = apply_plan_intervention(subst, baseline, None, rng_seed=1)
        assert intervention_record_integrity_ok(record)
        assert PlanInterventionRecordV1.from_dict(record.to_dict()) == record
