"""SLM-362 (DSH1-10): matched CAP0 curriculum experiment + CERT_CAP0 decision."""

from __future__ import annotations

import pytest

from slm_training.dsl.grammar_capabilities import GrammarCapabilityAdapterV1
from slm_training.dsl.pack import get_pack
from slm_training.harnesses.experiments.slm362_cap0_experiment import (
    ARMS,
    ARM_LEVERS,
    EVAL_ANSWERS,
    PREREGISTERED_MAX_P_VALUE,
    TRAIN_ANSWERS,
    StopRule,
    SuiteScoreV1,
    build_arm_corpus,
    build_cap0_eval_suite,
    build_preregistration,
    decide_certificate,
    exposure_accounting,
    paired_evidence,
    train_arm,
)
from slm_training.harnesses.train_data.cap0_eval_freeze import (
    Cap0GateV1,
    Cap0MetricsV1,
    evaluate_gate,
)


@pytest.fixture(scope="module")
def adapter() -> GrammarCapabilityAdapterV1:
    return GrammarCapabilityAdapterV1(get_pack("openui"))


@pytest.fixture(scope="module")
def corpora(adapter):
    return {arm: build_arm_corpus(arm, adapter=adapter) for arm in ARMS}


def _metrics(**overrides) -> Cap0MetricsV1:
    base = {
        "canonical_equivalence": 1.0,
        "accepted_set_mass": 1.0,
        "production_coverage": 1.0,
        "structure": 1.0,
        "binding_aware": 1.0,
        "empty_trivial_rate": 0.0,
        "diversity": 1.0,
        "identity_baseline_score": 0.0,
        "n_by_subsuite": dict(Cap0GateV1().preregistered_n),
    }
    base.update(overrides)
    return Cap0MetricsV1(**base)


def _score(metrics: Cap0MetricsV1 | None = None, **overrides) -> SuiteScoreV1:
    base = {
        "metrics": metrics or _metrics(),
        "parse_valid_rate": 1.0,
        "marker_fidelity": 1.0,
        "compositional_success": (1, 1, 1, 1),
        "prefix_success": (1, 1, 1, 1),
        "copy_hits": 0,
    }
    base.update(overrides)
    return SuiteScoreV1(**base)


def _winning_evidence() -> list[dict]:
    return [
        {
            "label": "compositional_held_out:STAGED_FULL_vs_BASE",
            "n": 4,
            "control_rate": 0.0,
            "candidate_rate": 1.0,
            "mcnemar": {
                "effect": 1.0,
                "control_only": 0,
                "candidate_only": 4,
                "p_value": 0.05,
            },
            "meets_preregistered_effect": True,
        }
    ]


def _decide(**overrides):
    staged = _score()
    base = _score(
        metrics=_metrics(canonical_equivalence=0.5, accepted_set_mass=0.5),
        compositional_success=(0, 0, 0, 0),
        prefix_success=(0, 0, 0, 0),
    )
    atomic = _score(metrics=_metrics(canonical_equivalence=0.5))
    gate = evaluate_gate(staged.metrics, gate=Cap0GateV1())
    assert gate.status == "certified"
    kwargs = {
        "staged_full": staged,
        "base": base,
        "atomic": atomic,
        "staged_gate": gate,
        "staged_predictions": {"r1": "out-a", "r2": "out-b"},
        "control_predictions": {
            "BASE": {"r1": "base-a", "r2": "base-b"},
            "ATOMIC_ONLY": {"r1": "atom-a", "r2": "atom-b"},
        },
        "composition_evidence": _winning_evidence(),
        "prefix_evidence": [],
        "contamination_findings": [],
        "binding": {
            "checkpoint_sha256": "a" * 64,
            "corpus_sha256": "b" * 64,
            "suite_sha256": "c" * 64,
            "gate_version": "cap0_gate/v1",
            "config_sha256": "d" * 64,
            "code_commit": "deadbeef",
            "evidence_json_uri": "docs/design/x.json",
        },
        "plan_sha256": "e" * 64,
    }
    kwargs.update(overrides)
    return decide_certificate(**kwargs)


# --------------------------------------------------------------------------- #
# Exposure equality + arm isolation
# --------------------------------------------------------------------------- #


def test_exposure_equality(corpora) -> None:
    accounting = exposure_accounting(corpora)
    assert accounting["equal_record_count"] is True
    assert accounting["equal_source_family_exposure"] is True
    counts = {arm: corpora[arm].accounting["records_total"] for arm in ARMS}
    assert len(set(counts.values())) == 1
    families = {arm: tuple(corpora[arm].accounting["source_families"]) for arm in ARMS}
    assert len(set(families.values())) == 1


def test_only_the_data_lever_varies(corpora) -> None:
    # Arms differ only in which lever-tagged records they carry.
    for arm, corpus in corpora.items():
        levers_present = {record.meta.get("lever") for record in corpus.records}
        declared = {lever for lever, on in ARM_LEVERS[arm].items() if on}
        assert levers_present <= declared
    assert len({arm for arm in ARMS}) == 4
    # Identical preregistered model/training config for every arm.
    prereg = build_preregistration()
    assert prereg.arm_levers == ARM_LEVERS
    assert set(prereg.arms) == set(ARMS)


def test_fixture_roots_split_disjoint(adapter) -> None:
    train_families = {family for _, family in TRAIN_ANSWERS}
    suite = build_cap0_eval_suite(
        list(EVAL_ANSWERS), adapter=adapter, seed=7, dsl="openui"
    )
    eval_families = {row.root_family_id for row in suite.rows}
    assert train_families.isdisjoint(eval_families)
    assert suite.manifest["rows_total"] > 0
    assert all(
        count > 0 for count in suite.manifest["rows_by_subsuite"].values()
    )


# --------------------------------------------------------------------------- #
# Certificate paths
# --------------------------------------------------------------------------- #


def test_certificate_issued_on_synthetic_win() -> None:
    decision = _decide()
    assert decision.status == "issued"
    assert decision.certificate == "CERT_CAP0"
    assert decision.rejection_codes == ()
    artifact = decision.artifact
    assert artifact["passed"] is True
    assert artifact["capability"] == "CAP0_GRAMMAR"
    binding = artifact["binding"]
    for key in (
        "checkpoint_sha256",
        "corpus_sha256",
        "suite_sha256",
        "gate_version",
        "config_sha256",
        "code_commit",
        "evidence_json_uri",
    ):
        assert binding[key]


def test_reject_prediction_identical() -> None:
    decision = _decide(
        control_predictions={"BASE": {"r1": "out-a", "r2": "out-b"}}
    )
    assert decision.status == "rejected"
    assert StopRule.PREDICTION_IDENTICAL.value in decision.rejection_codes


def test_reject_copy_dominated() -> None:
    # No paired win -> copy/atomic controls unexplained.
    decision = _decide(composition_evidence=[], prefix_evidence=[])
    assert decision.status == "rejected"
    assert StopRule.COPY_DOMINATED.value in decision.rejection_codes
    # Copy shortcut above the gate ceiling -> copy-dominated.
    decision = _decide(
        staged_full=_score(metrics=_metrics(identity_baseline_score=0.9))
    )
    assert StopRule.COPY_DOMINATED.value in decision.rejection_codes


def test_reject_structurally_regressive() -> None:
    decision = _decide(staged_full=_score(parse_valid_rate=0.5))
    assert decision.status == "rejected"
    assert StopRule.STRUCTURALLY_REGRESSIVE.value in decision.rejection_codes
    decision = _decide(
        staged_full=_score(metrics=_metrics(empty_trivial_rate=0.5))
    )
    assert StopRule.STRUCTURALLY_REGRESSIVE.value in decision.rejection_codes


def test_reject_contaminated() -> None:
    decision = _decide(contamination_findings=["split_leakage:row-1"])
    assert decision.status == "rejected"
    assert StopRule.CONTAMINATED.value in decision.rejection_codes


def test_reject_underpowered_fixture_n() -> None:
    fixture_metrics = _metrics(
        n_by_subsuite={name: 2 for name in Cap0GateV1().preregistered_n}
    )
    gate = evaluate_gate(fixture_metrics, gate=Cap0GateV1())
    assert gate.status == "insufficient_n"
    decision = _decide(staged_gate=gate)
    assert decision.status == "rejected"
    assert StopRule.UNDERPOWERED.value in decision.rejection_codes
    assert decision.certificate is None
    assert gate.insufficient_n  # underpowered detail is reported, not hidden


def test_failed_gate_thresholds_reject() -> None:
    gate = evaluate_gate(
        _metrics(canonical_equivalence=0.1), gate=Cap0GateV1()
    )
    assert gate.status == "failed"
    decision = _decide(staged_gate=gate)
    assert decision.status == "rejected"
    assert StopRule.UNDERPOWERED.value in decision.rejection_codes


# --------------------------------------------------------------------------- #
# Paired evidence + determinism
# --------------------------------------------------------------------------- #


def test_paired_evidence_mcnemar_and_wilson() -> None:
    evidence = paired_evidence([0, 0, 0, 0, 1], [1, 1, 1, 1, 0], label="t")
    assert evidence["n"] == 5
    assert evidence["mcnemar"]["candidate_only"] == 4
    assert evidence["mcnemar"]["control_only"] == 1
    assert evidence["candidate_wilson"]["n"] == 5
    assert 0.0 <= evidence["candidate_wilson"]["low"] <= 1.0
    assert PREREGISTERED_MAX_P_VALUE >= 0.0
    # Prediction-identical arms can never meet the preregistered effect.
    same = paired_evidence([1, 0, 1], [1, 0, 1], label="same")
    assert same["meets_preregistered_effect"] is False


def test_determinism(adapter) -> None:
    first = {arm: build_arm_corpus(arm, adapter=adapter) for arm in ARMS}
    second = {arm: build_arm_corpus(arm, adapter=adapter) for arm in ARMS}
    assert all(first[arm].sha256 == second[arm].sha256 for arm in ARMS)
    assert build_preregistration().sha256 == build_preregistration().sha256
    d1, d2 = _decide(), _decide()
    assert d1.to_dict() == d2.to_dict()


def test_train_arm_deterministic_small(corpora) -> None:
    prereg = build_preregistration()
    m1 = train_arm(
        corpora["BASE"].records,
        steps=1,
        batch_size=2,
        lr=prereg.lr,
        seed=0,
        model_config=prereg.model_config,
    )
    m2 = train_arm(
        corpora["BASE"].records,
        steps=1,
        batch_size=2,
        lr=prereg.lr,
        seed=0,
        model_config=prereg.model_config,
    )
    s1 = {k: v.sum().item() for k, v in m1.state_dict().items()}
    s2 = {k: v.sum().item() for k, v in m2.state_dict().items()}
    assert s1 == s2
