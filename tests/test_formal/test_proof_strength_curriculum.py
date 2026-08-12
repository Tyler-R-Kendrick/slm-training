"""RESEARCH-10 / SLM-566: proof-strength curriculum backend tests."""

from __future__ import annotations

from pathlib import Path

from slm_training.formal import proof_strength_curriculum as psc

ROOT = Path(__file__).resolve().parents[2]


def test_corpus_loads_with_obligation_kinds() -> None:
    groups = psc.default_groups(root=ROOT)
    assert len(groups) >= 4
    kinds = {o.kind for g in groups for o in g.obligations}
    assert "original_proof" in kinds
    assert "assumption_ablation" in kinds
    assert "reversal" in kinds


def test_eval_train_roots_disjoint() -> None:
    groups = psc.default_groups(root=ROOT)
    train_roots = {g.root_family_id for g in groups if g.split == "train"}
    eval_roots = {g.root_family_id for g in groups if g.split == "eval"}
    assert not train_roots & eval_roots


def test_paired_report_accepts_fixture_corpus() -> None:
    report = psc.paired_curriculum_report(root=ROOT)
    assert report["decision"] == "accept"
    assert report["train_eval_root_leakage"] is False
    assert report["sample_efficiency_to_target_obligation_pass_rate"] > 1.0


def test_unordered_shuffle_is_deterministic() -> None:
    groups = [g for g in psc.default_groups(root=ROOT) if g.split == "eval"]
    a = psc.schedule_obligations(groups[0], ordering="unordered")
    b = psc.schedule_obligations(groups[0], ordering="unordered")
    assert [o.obligation_id for o in a] == [o.obligation_id for o in b]
