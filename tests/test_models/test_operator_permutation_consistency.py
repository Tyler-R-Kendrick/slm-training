"""Regression coverage for evaluator-only operator permutation alignment."""

from __future__ import annotations

import hashlib

import pytest

from slm_training.dsl.operators import (
    CompilerFact,
    RefKind,
    ReferenceDescriptorV1,
    build_reference_table,
)
from slm_training.dsl.operators.legal_set import LegalSetCoverage
from slm_training.models.operator_permutation_consistency import (
    OperatorPermutationAlignmentV1,
    OperatorPermutationCacheV1,
    OperatorPermutationReportV1,
    compare_permuted_scores,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _alignment(
    *,
    coverage: LegalSetCoverage = LegalSetCoverage.COMPLETE,
    actions: tuple[str, ...] | None = None,
) -> OperatorPermutationAlignmentV1:
    return OperatorPermutationAlignmentV1(
        request_id="request-1",
        state_digest=_sha("state"),
        branch_digest=_sha("branch"),
        registry_fingerprint=_sha("registry"),
        coverage=coverage,
        semantic_action_ids=actions or (_sha("action-a"), _sha("action-b")),
        descriptor_fingerprints=(_sha("descriptor-a"), _sha("descriptor-b")),
    )


def test_complete_permutation_aligns_scores_by_semantic_action_not_position() -> None:
    original = _alignment()
    permuted = _alignment(actions=tuple(reversed(original.semantic_action_ids)))
    result = compare_permuted_scores(
        original,
        permuted,
        original_scores=(2.0, -1.0),
        permuted_scores=(-1.0, 2.0),
        accepted_semantic_action_ids=(original.semantic_action_ids[0],),
    )
    assert result.comparable is True
    assert result.top1_flip is False
    assert result.accepted_set_mass_drift == pytest.approx(0.0)
    assert result.score_disagreement == pytest.approx(0.0)
    assert result.js_divergence == pytest.approx(0.0)


def test_partial_requires_exactly_the_same_witnessed_membership() -> None:
    original = _alignment(coverage=LegalSetCoverage.PARTIAL)
    equal = _alignment(
        coverage=LegalSetCoverage.PARTIAL,
        actions=tuple(reversed(original.semantic_action_ids)),
    )
    matched = compare_permuted_scores(
        original,
        equal,
        original_scores=(1.0, 0.0),
        permuted_scores=(0.0, 1.0),
        accepted_semantic_action_ids=(original.semantic_action_ids[0],),
    )
    assert matched.comparable is True

    unequal = _alignment(
        coverage=LegalSetCoverage.PARTIAL,
        actions=(original.semantic_action_ids[0], _sha("action-c")),
    )
    rejected = compare_permuted_scores(
        original,
        unequal,
        original_scores=(1.0, 0.0),
        permuted_scores=(1.0, 0.0),
        accepted_semantic_action_ids=(original.semantic_action_ids[0],),
    )
    assert rejected.comparable is False
    assert rejected.reason == "unequal_witnessed_membership"
    assert rejected.js_divergence is None


def test_cross_request_is_never_compared_or_cached() -> None:
    original = _alignment()
    other = OperatorPermutationAlignmentV1(
        request_id="request-2",
        state_digest=original.state_digest,
        branch_digest=original.branch_digest,
        registry_fingerprint=original.registry_fingerprint,
        coverage=original.coverage,
        semantic_action_ids=original.semantic_action_ids,
        descriptor_fingerprints=original.descriptor_fingerprints,
    )
    comparison = compare_permuted_scores(
        original,
        other,
        original_scores=(1.0, 0.0),
        permuted_scores=(1.0, 0.0),
        accepted_semantic_action_ids=(original.semantic_action_ids[0],),
    )
    assert comparison.comparable is False
    assert comparison.reason == "cross_boundary"

    table = build_reference_table(
        request_id="request-1",
        state_digest=original.state_digest,
        branch_digest=original.branch_digest,
        descriptors=(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha("semantic"),
                value_type="openui.string",
                compiler_facts=(CompilerFact.VALUE_VISIBLE,),
            ),
        ),
        seed=1,
    )
    with pytest.raises(ValueError, match="permutation.cross_boundary_cache"):
        OperatorPermutationCacheV1().get_or_build(table, other, seed=7)


def test_cache_is_deterministic_and_report_keeps_negative_result_compact() -> None:
    alignment = _alignment()
    table = build_reference_table(
        request_id=alignment.request_id,
        state_digest=alignment.state_digest,
        branch_digest=alignment.branch_digest,
        descriptors=(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha("semantic-a"),
                value_type="openui.string",
            ),
        ),
        seed=1,
    )
    cache = OperatorPermutationCacheV1()
    assert cache.get_or_build(table, alignment, seed=7) is cache.get_or_build(
        table, alignment, seed=7
    )
    comparison = compare_permuted_scores(
        alignment,
        alignment,
        original_scores=(1.0, 0.0),
        permuted_scores=(1.0, 0.0),
        accepted_semantic_action_ids=(alignment.semantic_action_ids[0],),
    )
    report = OperatorPermutationReportV1((comparison,), cache.evidence()).to_dict()
    assert report["top1_flips"] == 0
    assert report["cache"] == {"hits": 1, "misses": 1, "entries": 1}
