"""Tests for SPV2-05 verifier-guided minimal semantic repair fixtures."""

from __future__ import annotations

import dataclasses
import math

import pytest

from slm_training.data.corrupt import CorruptionOperator, build_corruption
from slm_training.dsl.lang_core import validate
from slm_training.evals.semantic_failure import VerifierWitnessV1
from slm_training.harnesses.distill.semantic_repair import (
    ConflictSlice,
    LegalEdit,
    RepairEvidence,
    RepairFeatureExtractor,
    RepairResidualV1,
    SemanticRepairRecordV1,
    SemanticRepairScorer,
    apply_repair_policy,
    build_repair_records_from_corruption,
    project_repair_residual,
    residual_integrity_ok,
    train_repair_policy_fixture,
)


SIMPLE = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'


def test_build_records_from_simple_program() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    assert len(records) >= 35
    families = {r.metadata["family"] for r in records}
    assert families

    for record in records:
        assert record.schema_version == "semantic_repair/v1"
        assert record.source_fingerprint
        assert record.broken_openui != SIMPLE
        assert record.failure_evidence
        assert record.conflict_slice.stage
        assert record.legal_edits
        assert record.accepted_edit_ids
        assert record.oracle_edit_id in record.accepted_edit_ids
        assert record.lineage["operator"]
        assert record.version_stamp is not None
        assert record.version_stamp.get("stamp_schema") == "version_stamp/v1"


def test_record_roundtrip() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    record = records[0]
    restored = SemanticRepairRecordV1.from_dict(record.to_dict())
    assert restored == record


def test_oracle_policy_selects_accepted_repair() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    for record in records:
        chosen, meta = apply_repair_policy(record, "oracle")
        assert chosen.edit_id in record.accepted_edit_ids
        assert meta["policy"] == "oracle"


def test_edit_distance_policy_minimizes_cost() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    for record in records:
        chosen, _ = apply_repair_policy(record, "edit_distance")
        assert chosen.cost == min(e.cost for e in record.legal_edits)


def test_random_policy_is_deterministic_with_fixed_rng() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    record = records[0]
    import random

    rng = random.Random(7)
    chosen1, _ = apply_repair_policy(record, "random", rng=rng)
    rng = random.Random(7)
    chosen2, _ = apply_repair_policy(record, "random", rng=rng)
    assert chosen1.edit_id == chosen2.edit_id


def test_apply_oracle_repair_preserves_hard_validity() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    for record in records:
        chosen, _ = apply_repair_policy(record, "oracle")
        repaired = chosen.after
        program = validate(repaired)
        assert program.serialized or repaired


def test_feature_extractor_has_accepted_label() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    extractor = RepairFeatureExtractor()
    for record in records:
        for edit in record.legal_edits:
            features = extractor.extract(record, edit)
            assert "accepted" in features
            assert features["accepted"] in (0.0, 1.0)


def test_scorer_runs_on_real_records() -> None:
    pytest.importorskip("torch")
    records = build_repair_records_from_corruption(SIMPLE)
    extractor = RepairFeatureExtractor()
    scorer = SemanticRepairScorer()
    for record in records[:5]:
        for edit in record.legal_edits:
            score = scorer.score(record, edit, extractor)
            assert math.isfinite(score)


def _augment_record_with_negatives(
    record: SemanticRepairRecordV1,
) -> SemanticRepairRecordV1:
    """Add two non-accepted legal edits so the scorer has something to rank down."""
    extra_edits: list[LegalEdit] = list(record.legal_edits)
    base_id = record.record_id
    for i, suffix in enumerate(("_bad1", "_bad2")):
        bad_after = record.broken_openui.replace(")", f"{suffix})", 1)
        extra_edits.append(
            LegalEdit(
                edit_id=f"{base_id}-neg{i}",
                kind="replace_program",
                before=record.broken_openui,
                after=bad_after,
                cost=10 + i,
                source="synthetic_negative",
            )
        )
    return SemanticRepairRecordV1(
        record_id=record.record_id,
        source_fingerprint=record.source_fingerprint,
        broken_openui=record.broken_openui,
        failure_evidence=record.failure_evidence,
        conflict_slice=record.conflict_slice,
        legal_edits=tuple(extra_edits),
        accepted_edit_ids=record.accepted_edit_ids,
        oracle_edit_id=record.oracle_edit_id,
        lineage=record.lineage,
        metadata=record.metadata,
        schema_version=record.schema_version,
        version_stamp=record.version_stamp,
    )


def test_train_fixture_reduces_repair_ranking_loss() -> None:
    pytest.importorskip("torch")
    records = build_repair_records_from_corruption(SIMPLE)[:8]
    augmented = [_augment_record_with_negatives(r) for r in records]

    extractor = RepairFeatureExtractor()
    scorer = SemanticRepairScorer()
    result = train_repair_policy_fixture(
        augmented, scorer, extractor, steps=40, lr=0.05, seed=0
    )
    assert result["n_decisions"] == sum(len(r.legal_edits) for r in augmented)
    assert result["history"][0]["loss"] > result["history"][-1]["loss"]

    # After training, accepted edits should outrank non-accepted edits.
    correct = 0
    total = 0
    for record in augmented:
        accepted_scores = []
        rejected_scores = []
        for edit in record.legal_edits:
            score = scorer.score(record, edit, extractor)
            if edit.edit_id in record.accepted_edit_ids:
                accepted_scores.append(score)
            else:
                rejected_scores.append(score)
        if accepted_scores and rejected_scores:
            total += 1
            if min(accepted_scores) > max(rejected_scores):
                correct += 1
    if total:
        assert correct / total >= 0.75


def test_multiple_accepted_repairs_are_not_exact() -> None:
    alternative = SIMPLE.replace('"column"', '"row"')
    build_corruption(
        SIMPLE,
        CorruptionOperator.MISSING_QUOTE,
        acceptable_repairs=(SIMPLE, alternative),
    )
    records = build_repair_records_from_corruption(SIMPLE)
    record = next(
        (r for r in records if r.metadata["operator"] == "missing_quote"), None
    )
    assert record is not None
    assert len(record.accepted_edit_ids) == 1
    assert record.metadata["exact_repair"]


def test_unknown_accepted_set_falls_back_to_random() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    record = records[0]
    unknown = SemanticRepairRecordV1(
        record_id=record.record_id,
        source_fingerprint=record.source_fingerprint,
        broken_openui=record.broken_openui,
        failure_evidence=record.failure_evidence,
        conflict_slice=record.conflict_slice,
        legal_edits=record.legal_edits,
        accepted_edit_ids=(),
        oracle_edit_id=None,
        lineage=record.lineage,
        metadata=record.metadata,
    )
    chosen, meta = apply_repair_policy(unknown, "oracle")
    assert chosen in unknown.legal_edits
    assert meta.get("unknown")


def test_conflict_slice_authorization() -> None:
    exact = ConflictSlice(
        stage="grammar",
        failing_node_ids=(0,),
        dependency_frontier=(0,),
        protected_node_ids=(1,),
        completeness_class="EXACT",
    )
    heuristic = ConflictSlice(
        stage="grammar",
        failing_node_ids=(0,),
        dependency_frontier=(0,),
        protected_node_ids=(1,),
        completeness_class="HEURISTIC",
    )
    assert exact.can_authorize_repair()
    assert not heuristic.can_authorize_repair()


def test_legal_edit_and_evidence_roundtrip() -> None:
    edit = LegalEdit(
        edit_id="e1",
        kind="replace_program",
        before="broken",
        after="fixed",
        cost=3,
        source="test",
    )
    evidence = RepairEvidence(
        reason_code="grammar", analyzer="lang-core", detail="missing quote"
    )
    assert LegalEdit.from_dict(edit.to_dict()) == edit
    assert RepairEvidence.from_dict(evidence.to_dict()) == evidence


def _fake_witness(*, unresolved: tuple[str, ...] = ()) -> VerifierWitnessV1:
    """Directly construct a VerifierWitnessV1 for reconciliation tests."""
    return VerifierWitnessV1(
        schema_version="verifier_witness/v1",
        source_trace_fingerprint="trace-fp-1",
        taxonomy_version="semantic_failure_taxonomy/v1",
        evaluator_version="v1",
        evaluator_hash="hash1",
        first_failed_gate=None,
        localizations=(),
        unresolved_localizations=unresolved,
        witness_digest="unused-in-direct-construction",
    )


def test_project_residual_is_lossless_and_golden() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    for record in records:
        residual = project_repair_residual(record)
        assert residual.schema_version == "repair_residual/v1"
        assert residual.source_record_id == record.record_id
        assert residual.source_fingerprint == record.source_fingerprint
        assert residual.source_witness_fingerprint is None
        assert residual.completeness_class == record.conflict_slice.completeness_class
        assert residual.conflict_stage == record.conflict_slice.stage
        assert residual.failing_node_ids == record.conflict_slice.failing_node_ids
        assert residual.dependency_frontier == record.conflict_slice.dependency_frontier
        assert residual.protected_node_ids == record.conflict_slice.protected_node_ids
        assert residual.advisory_only is True
        assert residual.residual_hash
        assert residual_integrity_ok(residual)
        # Golden round trip: to_dict/from_dict must reconstruct an equal object.
        assert RepairResidualV1.from_dict(residual.to_dict()) == residual


def test_heuristic_conflict_slice_yields_empty_action_domain() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    record = records[0]
    heuristic_record = dataclasses.replace(
        record,
        conflict_slice=dataclasses.replace(
            record.conflict_slice, completeness_class="HEURISTIC"
        ),
    )
    residual = project_repair_residual(heuristic_record)
    assert residual.completeness_class == "HEURISTIC"
    assert not residual.can_authorize_repair()
    assert residual.legal_repair_action_domain == ()

    # An EXACT/SOUND_OVERAPPROX slice with legal edits does expose its domain.
    exact_record = dataclasses.replace(
        record,
        conflict_slice=dataclasses.replace(
            record.conflict_slice, completeness_class="EXACT"
        ),
    )
    exact_residual = project_repair_residual(exact_record)
    assert exact_residual.can_authorize_repair()
    assert exact_residual.legal_repair_action_domain == tuple(
        e.edit_id for e in exact_record.legal_edits
    )


def test_witness_reconciliation_never_fabricates_or_converts_unknown() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    record = records[0]
    witness = _fake_witness(unresolved=("gate:HUMAN_AUDIT", "semantic_check:foo"))

    baseline = project_repair_residual(record)
    reconciled = project_repair_residual(record, witness=witness)

    assert reconciled.source_witness_fingerprint == witness.source_trace_fingerprint
    for item in witness.unresolved_localizations:
        assert f"witness:{item}" in reconciled.unsatisfied_obligations

    # The witness only ever adds obligations, never legal edits, and never
    # flips completeness/authorization -- both stay identical to the
    # witness-free projection.
    assert reconciled.legal_repair_action_domain == baseline.legal_repair_action_domain
    assert reconciled.completeness_class == baseline.completeness_class
    known_edit_ids = {e.edit_id for e in record.legal_edits}
    assert set(reconciled.legal_repair_action_domain) <= known_edit_ids


def test_residual_is_frozen_and_protected_regions_are_not_mutated() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    record = records[0]
    residual = project_repair_residual(record)
    original_protected = record.conflict_slice.protected_node_ids

    with pytest.raises(dataclasses.FrozenInstanceError):
        residual.protected_node_ids = ()  # type: ignore[misc]

    assert record.conflict_slice.protected_node_ids == original_protected
    assert residual.protected_node_ids == original_protected


def test_residual_tamper_and_schema_rejection() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    residual = project_repair_residual(records[0])

    tampered = residual.to_dict()
    tampered["protected_node_ids"] = list(tampered["protected_node_ids"]) + ["injected"]
    with pytest.raises(ValueError, match="hash mismatch"):
        RepairResidualV1.from_dict(tampered)

    wrong_schema = residual.to_dict()
    wrong_schema["schema_version"] = "repair_residual/v0"
    with pytest.raises(ValueError, match="unsupported repair residual schema"):
        RepairResidualV1.from_dict(wrong_schema)


def test_residual_handles_partial_evidence() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    record = dataclasses.replace(records[0], failure_evidence=())
    residual = project_repair_residual(record)
    assert residual.unsatisfied_obligations == ()
    assert residual_integrity_ok(residual)


def test_residual_projection_is_deterministic_and_replayable() -> None:
    records = build_repair_records_from_corruption(SIMPLE)
    record = records[0]
    first = project_repair_residual(record)
    second = project_repair_residual(record)
    assert first == second
    assert first.residual_hash == second.residual_hash
