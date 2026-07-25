"""SLM-317 (LAR2-06): metamorphic invariance, commit truth table, visibility.

Runs before any model evaluation: the commit/score pipeline must be invariant
to alpha-renaming, semantics-preserving reordering, formatting normalization,
and equivalent AST serializations — and the do-no-harm rule must cover its
full truth table deterministically.
"""

from __future__ import annotations

import json

import pytest

from slm_training.harnesses.experiments.slm155_factorization_comparison import (
    paired_validity_table,
)
from slm_training.harnesses.experiments.slm317_repair_hybrid import (
    ACTION_ABSTAIN,
    ACTION_COMMIT,
    ACTION_RETAIN,
    REASON_CANDIDATE_UNAVAILABLE,
    REASON_HARD_IMPROVEMENT,
    REASON_HARD_REGRESSION,
    REASON_NO_IMPROVEMENT,
    REASON_ORACLE_NO_GAIN,
    REASON_SOFT_IMPROVEMENT,
    alpha_rename,
    ast_roundtrip,
    commit,
    hard_evidence,
    normalize_formatting,
    oracle_commit,
    reorder_statements,
)
from scripts.run_slm317_repair_hybrid import ArmOutcome

VALID = 'root = Stack([n0], "column")\nn0 = TextContent(":content")'
VALID_TWO = (
    'root = Stack([n0, n1], "column")\n'
    'n0 = TextContent(":content")\n'
    'n1 = Button(":cta")'
)
INVALID_UNPARSEABLE = "root = Stack(([n0],"
VALID_PARSE_ONLY = 'root = Stack([], "column")'  # parses; v1 invalid (empty stack)

TRANSFORMS = (alpha_rename, reorder_statements, normalize_formatting, ast_roundtrip)


# --------------------------------------------------------------------------- #
# Metamorphic invariance (all four transforms)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", [VALID, VALID_TWO, VALID_PARSE_ONLY])
def test_hard_evidence_invariant_across_transforms(source: str) -> None:
    base = hard_evidence(source)
    for transform in TRANSFORMS:
        variant = transform(source)
        ev = hard_evidence(variant)
        assert ev.rank == base.rank, transform.__name__
        assert ev.parse_ok == base.parse_ok
        assert ev.contract_ok == base.contract_ok
        assert ev.v1_verdict == base.v1_verdict


def test_commit_decision_invariant_across_transforms() -> None:
    evidence = {"learned_score_source": 0.2, "learned_score_candidate": 0.9}
    base = commit(VALID, VALID_TWO, evidence=evidence)
    for transform in TRANSFORMS:
        variant = commit(transform(VALID), transform(VALID_TWO), evidence=evidence)
        assert variant.action == base.action, transform.__name__
        assert variant.reason == base.reason, transform.__name__
        assert variant.source_evidence.rank == base.source_evidence.rank
        assert variant.candidate_evidence.rank == base.candidate_evidence.rank


def test_transforms_preserve_validity() -> None:
    for transform in TRANSFORMS:
        assert hard_evidence(transform(VALID)).v1_verdict, transform.__name__


# --------------------------------------------------------------------------- #
# Commit-rule truth table (all branches + determinism)
# --------------------------------------------------------------------------- #


def test_truth_table_hard_improvement() -> None:
    decision = commit(VALID_PARSE_ONLY, VALID)
    assert decision.action == ACTION_COMMIT
    assert decision.reason == REASON_HARD_IMPROVEMENT
    assert decision.final == VALID


def test_truth_table_hard_regression_retains_source() -> None:
    decision = commit(VALID, INVALID_UNPARSEABLE)
    assert decision.action == ACTION_RETAIN
    assert decision.reason == REASON_HARD_REGRESSION
    assert decision.final == VALID


def test_truth_table_soft_improvement_no_hard_regression() -> None:
    candidate = alpha_rename(VALID)  # same hard rank, different source
    decision = commit(
        VALID,
        candidate,
        evidence={"learned_score_source": 0.1, "learned_score_candidate": 0.9},
    )
    assert decision.action == ACTION_COMMIT
    assert decision.reason == REASON_SOFT_IMPROVEMENT
    assert decision.final == candidate


def test_soft_commit_forbidden_when_hard_regresses() -> None:
    decision = commit(
        VALID,
        INVALID_UNPARSEABLE,
        evidence={"learned_score_source": 0.1, "learned_score_candidate": 0.9},
    )
    assert decision.action == ACTION_RETAIN
    assert decision.reason == REASON_HARD_REGRESSION


def test_truth_table_no_improvement() -> None:
    candidate = alpha_rename(VALID)
    decision = commit(VALID, candidate)  # no learned scores, same rank
    assert decision.action == ACTION_RETAIN
    assert decision.reason == REASON_NO_IMPROVEMENT
    assert decision.final == VALID


def test_soft_margin_not_exceeded_retains() -> None:
    candidate = alpha_rename(VALID)
    decision = commit(
        VALID,
        candidate,
        evidence={
            "learned_score_source": 0.5,
            "learned_score_candidate": 0.5,
            "learned_margin": 0.0,
        },
    )
    assert decision.action == ACTION_RETAIN
    assert decision.reason == REASON_NO_IMPROVEMENT


def test_truth_table_abstain_on_missing_candidate() -> None:
    for candidate in (None, "", "   "):
        decision = commit(VALID, candidate)
        assert decision.action == ACTION_ABSTAIN
        assert decision.reason == REASON_CANDIDATE_UNAVAILABLE
        assert decision.final == VALID


def test_commit_deterministic() -> None:
    evidence = {"learned_score_source": 0.1, "learned_score_candidate": 0.9}
    first = commit(VALID, VALID_TWO, evidence=evidence).to_dict()
    second = commit(VALID, VALID_TWO, evidence=evidence).to_dict()
    assert first == second


def test_decision_preserves_source_candidate_reason() -> None:
    decision = commit(VALID_PARSE_ONLY, VALID, record_id="r1", arm_id="a", seed=3)
    row = decision.to_dict()
    assert row["source"] == VALID_PARSE_ONLY
    assert row["candidate"] == VALID
    assert row["reason"] == REASON_HARD_IMPROVEMENT
    assert row["record_id"] == "r1"
    assert row["schema"] == "slm317_commit_decision/v1"
    json.dumps(row)  # durable JSONL-serializable


# --------------------------------------------------------------------------- #
# Oracle selector upper bound
# --------------------------------------------------------------------------- #


def test_oracle_commits_only_on_strict_hard_improvement() -> None:
    better = oracle_commit(VALID_PARSE_ONLY, VALID)
    assert better.action == ACTION_COMMIT
    worse = oracle_commit(VALID, VALID_PARSE_ONLY)
    assert worse.action == ACTION_RETAIN
    assert worse.reason == REASON_ORACLE_NO_GAIN
    equal = oracle_commit(VALID, alpha_rename(VALID))
    assert equal.action == ACTION_RETAIN  # equal rank is not an improvement


def test_oracle_upper_bound_dominates_rule() -> None:
    # Whatever the deterministic rule commits, the oracle commits a superset
    # (it commits on every strict hard improvement; the rule adds soft commits
    # only when hard evidence is unchanged, which can never beat the oracle's
    # final rank).
    cases = [
        (VALID_PARSE_ONLY, VALID),
        (VALID, VALID_PARSE_ONLY),
        (VALID, INVALID_UNPARSEABLE),
        (VALID, alpha_rename(VALID)),
    ]
    for source, candidate in cases:
        rule = commit(
            source,
            candidate,
            evidence={"learned_score_source": 0.0, "learned_score_candidate": 1.0},
        )
        oracle = oracle_commit(source, candidate)
        assert oracle.source_evidence.rank <= oracle.candidate_evidence.rank or (
            oracle.action == ACTION_RETAIN
        )
        assert hard_evidence(oracle.final).rank >= hard_evidence(rule.final).rank


# --------------------------------------------------------------------------- #
# Invalid-over-valid visibility + arm isolation
# --------------------------------------------------------------------------- #


def _outcome(record_id: str, arm: str, seed: int, verdict: bool) -> ArmOutcome:
    return ArmOutcome(
        record_id=record_id,
        arm_id=arm,
        seed=seed,
        final_source=VALID if verdict else INVALID_UNPARSEABLE,
        v1_verdict=verdict,
        hard_rank=3 if verdict else 0,
    )


def test_invalid_over_valid_damage_surfaces() -> None:
    """AR final valid + repaired final invalid must be visible, never hidden."""
    ar = [_outcome("r0", "ar_only", 0, True), _outcome("r1", "ar_only", 0, True)]
    repaired = [
        _outcome("r0", "ar_repair_improved", 0, True),
        _outcome("r1", "ar_repair_improved", 0, False),  # damaged
    ]
    counts = paired_validity_table(ar, repaired)
    assert counts["a_valid_b_invalid"] == 1  # the damage count is explicit
    assert counts["both_valid"] == 1


def test_commit_rule_makes_damage_unreachable() -> None:
    """With the commit rule in the loop, a damaging candidate is retained away."""
    decision = commit(VALID, INVALID_UNPARSEABLE)
    final_ev = hard_evidence(decision.final)
    assert final_ev.v1_verdict  # final is the retained valid AR source


def test_arm_isolation() -> None:
    """Outcomes from one arm never leak into another arm's pairing."""
    ar = [_outcome("r0", "ar_only", 0, True)]
    other = [_outcome("r0", "repair_only", 0, False)]
    counts = paired_validity_table(ar, other)
    assert counts["a_valid_b_invalid"] == 1
    # A third arm with no matching record contributes only unpaired rows.
    third = [_outcome("rX", "ar_repair_improved", 0, True)]
    counts2 = paired_validity_table(ar, third)
    assert counts2["unpaired"] == 2
    assert counts2["a_valid_b_invalid"] == 0


def test_paired_table_deterministic() -> None:
    ar = [_outcome(f"r{i}", "ar_only", 0, i % 2 == 0) for i in range(4)]
    arm = [_outcome(f"r{i}", "ar_repair_improved", 0, True) for i in range(4)]
    assert paired_validity_table(ar, arm) == paired_validity_table(ar, arm)
