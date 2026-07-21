"""Tests for SLM-209 (SDE5-02) debt-targeted curriculum fixture harness."""

from __future__ import annotations

from collections import Counter

from slm_training.harnesses.experiments.slm209_debt_targeted_curriculum import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    POLICY_NAMES,
    DebtCurriculumCellV1,
    DebtCurriculumManifestV1,
    build_cells,
    build_debt_curriculum_manifest,
    build_synthetic_debt_and_events,
    compute_selection_score,
    run_fixture_campaign,
    select_states_for_cell,
    validate_manifest,
)


def test_build_synthetic_debt_and_events_produces_valid_rows() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=50, seed=0)
    assert len(debts) == 50
    assert len(events) == 50
    for debt, event in zip(debts, events):
        assert debt.state_id == event.event_id
        assert debt.group_id == event.group_id
        assert debt.decision_kind == event.decision_kind
        assert debt.split == event.split
        assert debt.legal_mass > 0.0
        assert debt.good_debt is not None or debt.legal_debt is not None


def test_build_synthetic_deterministic() -> None:
    debts_a, events_a = build_synthetic_debt_and_events(n_states=50, seed=1)
    debts_b, events_b = build_synthetic_debt_and_events(n_states=50, seed=1)
    assert [d.state_id for d in debts_a] == [d.state_id for d in debts_b]
    assert [e.event_id for e in events_a] == [e.event_id for e in events_b]


def test_compute_selection_score_policies_return_components() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=20, seed=0)
    rarity = Counter(e.decision_kind for e in events)
    for policy in POLICY_NAMES:
        result = compute_selection_score(debts[0], events[0], policy, rarity_counter=rarity)
        assert "score" in result
        assert isinstance(result["score"], float)
        assert len(result) > 1


def test_uniform_policy_score_is_one() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=10, seed=0)
    result = compute_selection_score(debts[0], events[0], "uniform")
    assert result["score"] == 1.0


def test_high_debt_policy_uses_effective_debt() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=10, seed=0)
    result = compute_selection_score(debts[0], events[0], "high_debt")
    expected = debts[0].good_debt if debts[0].good_debt is not None else debts[0].legal_debt
    assert result["score"] == expected


def test_select_states_for_cell_preserves_budget() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=100, seed=0)
    selections, audit = select_states_for_cell(
        debts, events, "high_debt", total_decision_budget=40, per_group_cap=4, seed=0
    )
    assert len(selections) == 40
    assert audit["max_group_count"] <= 4


def test_select_states_for_cell_enforces_per_group_cap() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=100, seed=0)
    selections, audit = select_states_for_cell(
        debts, events, "uniform", total_decision_budget=80, per_group_cap=2, seed=0
    )
    group_counts = Counter(s.group_id for s in selections)
    assert max(group_counts.values()) <= 2
    assert audit["max_group_count"] <= 2


def test_no_train_held_out_group_leakage() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=100, seed=0)
    selections, _audit = select_states_for_cell(
        debts, events, "preregistered_composite", total_decision_budget=80, per_group_cap=4, seed=0
    )
    group_splits: dict[str, set[str]] = {}
    for sel in selections:
        group_splits.setdefault(sel.group_id, set()).add(sel.split)
    for group_id, splits in group_splits.items():
        assert len(splits) == 1, f"group {group_id} appears in multiple splits"


def test_score_components_persisted_for_each_selection() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=50, seed=0)
    selections, _audit = select_states_for_cell(
        debts, events, "debt_plus_rarity", total_decision_budget=30, per_group_cap=4, seed=0
    )
    for sel in selections:
        assert "score" in sel.score_components
        assert "effective_debt" in sel.score_components


def test_permutation_control_preserves_weight_histogram() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=60, seed=0)
    selections_a, _audit_a = select_states_for_cell(
        debts, events, "high_debt", total_decision_budget=30, per_group_cap=3, seed=7
    )
    shuffled_debts = list(debts)
    shuffled_events = list(events)
    # Reversing is a deterministic permutation.
    shuffled_debts.reverse()
    shuffled_events.reverse()
    selections_b, _audit_b = select_states_for_cell(
        shuffled_debts,
        shuffled_events,
        "high_debt",
        total_decision_budget=30,
        per_group_cap=3,
        seed=7,
    )
    ids_a = {s.state_id for s in selections_a}
    ids_b = {s.state_id for s in selections_b}
    assert ids_a == ids_b
    assert _audit_a["by_decision_kind"] == _audit_b["by_decision_kind"]


def test_manifest_round_trip() -> None:
    manifest = run_fixture_campaign(seeds=(0,))
    reconstructed = DebtCurriculumManifestV1.from_dict(manifest.to_dict())
    assert reconstructed.matrix_set == MATRIX_SET
    assert reconstructed.matrix_version == MATRIX_VERSION
    assert reconstructed.experiment_id == EXPERIMENT_ID
    assert reconstructed.status == "fixture"
    assert reconstructed.claim_class == "wiring"
    assert len(reconstructed.cells) == len(POLICY_NAMES)


def test_build_cells_produces_all_policies_per_seed() -> None:
    cells = build_cells(seeds=(0, 1, 2))
    assert len(cells) == len(POLICY_NAMES) * 3
    per_seed = {}
    for cell in cells:
        per_seed.setdefault(cell.seed, set()).add(cell.policy_name)
    assert len(per_seed) == 3
    for seed, names in per_seed.items():
        assert names == set(POLICY_NAMES), f"seed {seed} missing policies"


def test_validate_manifest_accepts_valid_cells() -> None:
    cells = build_cells(seeds=(0,))
    assert validate_manifest(cells) == []


def test_validate_manifest_rejects_duplicate_cell() -> None:
    cells = build_cells(seeds=(0,))
    duplicated = cells + (cells[0],)
    errors = validate_manifest(duplicated)
    assert any("duplicate" in e for e in errors)


def test_validate_manifest_rejects_invalid_policy() -> None:
    cells = build_cells(seeds=(0,))
    bad = DebtCurriculumCellV1(
        policy_name="unknown",
        weight_config={},
        selections=(),
        exposure_audit={},
        decision_budget=120,
        per_group_cap=6,
        seed=0,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("invalid policy" in e for e in errors)


def test_validate_manifest_rejects_non_positive_budget() -> None:
    cells = build_cells(seeds=(0,))
    bad = DebtCurriculumCellV1(
        policy_name="uniform",
        weight_config={},
        selections=(),
        exposure_audit={},
        decision_budget=0,
        per_group_cap=6,
        seed=0,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("decision_budget" in e for e in errors)


def test_fixture_campaign_produces_manifest() -> None:
    manifest = run_fixture_campaign(seeds=(0,))
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert manifest.experiment_id == EXPERIMENT_ID
    assert manifest.status == "fixture"
    assert manifest.claim_class == "wiring"
    assert len(manifest.cells) == len(POLICY_NAMES)
    assert manifest.lineage["synthetic_state_count"] == 200
    assert manifest.lineage["source_event_digest"]
    assert manifest.lineage["debt_artifact_digest"]


def test_version_stamp_includes_components() -> None:
    manifest = run_fixture_campaign(seeds=(0,))
    components = manifest.version_stamp.get("components", {})
    assert "harness.experiments" in components
    assert "harness.experiments.slm209_debt_targeted_curriculum" in components
    assert "harness.preference.constraint_debt" in components
    assert "harness.train_data" in components


def test_debt_targeted_policies_select_higher_debt_than_uniform() -> None:
    manifest = run_fixture_campaign(seeds=(0, 1))
    by_policy: dict[str, list[float]] = {}
    for cell in manifest.cells:
        by_policy.setdefault(cell.policy_name, []).append(
            cell.exposure_audit.get("mean_effective_debt", 0.0)
        )
    means = {p: sum(v) / len(v) for p, v in by_policy.items()}
    assert means["high_debt"] > means["uniform"]


def test_build_debt_curriculum_manifest_uses_provided_cells() -> None:
    debts, events = build_synthetic_debt_and_events(n_states=50, seed=0)
    cells = build_cells(seeds=(0,), total_decision_budget=25, per_group_cap=3)
    manifest = build_debt_curriculum_manifest(
        debts, events, cells=cells, total_decision_budget=25, per_group_cap=3, seed=0
    )
    assert manifest.total_decision_budget == 25
    assert manifest.per_group_cap == 3
    for cell in manifest.cells:
        assert cell.decision_budget == 25
        assert cell.per_group_cap == 3
