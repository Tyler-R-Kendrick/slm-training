"""Regression tests for solver-state supervision mixing (EFS3-01)."""

from __future__ import annotations

import pytest

from slm_training.evals.solver_state_supervision import (
    SupervisionSource,
    SolverStateMixSpec,
    SolverStateTrainingExampleV1,
    build_solver_state_mix,
    compare_solver_state_supervision,
)


def _example(
    state_fingerprint: str,
    source: SupervisionSource,
    split: str = "train",
    group: str | None = None,
    acceptable: tuple[str, ...] = ("a",),
    verdict: str = "SUPPORTED",
    certified: bool = True,
) -> SolverStateTrainingExampleV1:
    return SolverStateTrainingExampleV1(
        problem_id="p1",
        state_fingerprint=state_fingerprint,
        supervision_source=source,
        legal_actions=tuple({"value": v} for v in ("a", "b", "c")),
        acceptable_actions=tuple({"value": v} for v in acceptable),
        support_verdict=verdict,
        cost_to_go=1.0 if verdict == "SUPPORTED" else None,
        cost_observed=verdict == "SUPPORTED",
        split_group_id=group or state_fingerprint,
        split=split,
        lineage_id="lineage-" + state_fingerprint,
        program_family_id="family-1",
        replay_certified=certified,
    )


def test_pure_gold_selects_only_gold() -> None:
    rows = [
        _example("s1", SupervisionSource.GOLD),
        _example("s2", SupervisionSource.ON_POLICY),
    ]
    spec = SolverStateMixSpec(
        mix_id="gold", source_weights={SupervisionSource.GOLD: 1.0}
    )
    result = build_solver_state_mix(rows, spec)
    assert len(result.rows) == 1
    assert result.rows[0].supervision_source is SupervisionSource.GOLD
    assert result.source_counts["gold"] == 1


def test_pure_on_policy_selects_only_on_policy() -> None:
    rows = [
        _example("s1", SupervisionSource.GOLD),
        _example("s2", SupervisionSource.ON_POLICY),
        _example("s3", SupervisionSource.ON_POLICY),
    ]
    spec = SolverStateMixSpec(
        mix_id="on_policy", source_weights={SupervisionSource.ON_POLICY: 1.0}
    )
    result = build_solver_state_mix(rows, spec)
    assert len(result.rows) == 2
    assert all(r.supervision_source is SupervisionSource.ON_POLICY for r in result.rows)


def test_mixed_50_50_respects_fractions() -> None:
    rows = [
        _example(f"g{i}", SupervisionSource.GOLD) for i in range(10)
    ] + [
        _example(f"o{i}", SupervisionSource.ON_POLICY) for i in range(10)
    ]
    spec = SolverStateMixSpec(
        mix_id="mixed",
        source_weights={SupervisionSource.GOLD: 0.5, SupervisionSource.ON_POLICY: 0.5},
        seed=7,
        max_rows_per_source=8,
    )
    result = build_solver_state_mix(rows, spec)
    assert all(r.supervision_source is SupervisionSource.MIXED for r in result.rows)
    assert result.source_counts["mixed"] == len(result.rows)
    assert len(result.rows) == 16


def test_compare_creates_three_mixes() -> None:
    rows = [
        _example(f"g{i}", SupervisionSource.GOLD) for i in range(4)
    ] + [
        _example(f"o{i}", SupervisionSource.ON_POLICY) for i in range(4)
    ]
    comp = compare_solver_state_supervision(rows, seed=1, max_rows_per_source=3)
    assert len(comp.gold.rows) == 3
    assert len(comp.on_policy.rows) == 3
    assert len(comp.mixed.rows) == 6
    assert comp.gold.rows[0].supervision_source is SupervisionSource.GOLD
    assert comp.on_policy.rows[0].supervision_source is SupervisionSource.ON_POLICY
    assert comp.mixed.rows[0].supervision_source is SupervisionSource.MIXED


def test_cross_split_leak_is_rejected() -> None:
    rows = [
        _example("s1", SupervisionSource.GOLD, split="train", group="g1"),
        _example("s1_val", SupervisionSource.GOLD, split="val", group="g1"),
        _example("s2", SupervisionSource.GOLD, split="train", group="g2"),
    ]
    spec = SolverStateMixSpec(
        mix_id="gold", source_weights={SupervisionSource.GOLD: 1.0}
    )
    result = build_solver_state_mix(rows, spec)
    leaked = {r["state_fingerprint"] for r in result.rejected_rows}
    assert "s1" in leaked
    assert "s2" not in leaked
    assert len(result.rows) == 1
    assert result.rows[0].state_fingerprint == "s2"


def test_unknown_verdict_preserved() -> None:
    row = _example("s1", SupervisionSource.GOLD, verdict="UNKNOWN")
    assert row.support_verdict == "UNKNOWN"
    spec = SolverStateMixSpec(
        mix_id="gold", source_weights={SupervisionSource.GOLD: 1.0}
    )
    result = build_solver_state_mix([row], spec)
    assert len(result.rows) == 1
    assert result.rows[0].support_verdict == "UNKNOWN"


def test_round_trip_dict() -> None:
    row = _example("s1", SupervisionSource.GOLD)
    data = row.to_dict()
    restored = SolverStateTrainingExampleV1.from_dict(data)
    assert restored == row


def test_spec_normalizes_weights() -> None:
    spec = SolverStateMixSpec(
        mix_id="mixed",
        source_weights={SupervisionSource.GOLD: 2.0, SupervisionSource.ON_POLICY: 2.0},
    )
    weights = spec.normalized_weights()
    assert weights[SupervisionSource.GOLD] == pytest.approx(0.5)
    assert weights[SupervisionSource.ON_POLICY] == pytest.approx(0.5)
