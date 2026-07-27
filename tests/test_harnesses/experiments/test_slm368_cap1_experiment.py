"""SLM-368 (DSH2-08): matched CAP1 grounding experiment + CERT_CAP1 decision."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.slm368_cap1_experiment import (
    ARMS,
    ARM_LEVERS,
    CERT_CAP1,
    CONTROL_ARMS,
    FILTERED_ARMS,
    PREREGISTERED_MAX_P_VALUE,
    SCHEMA_ARMS,
    StopRule,
    SuiteScoreV1,
    build_arm_corpus,
    build_cap1_two_pack_suite,
    build_preregistration,
    build_train_cases,
    decide_certificate,
    exposure_accounting,
    paired_evidence,
    prepare_context,
    symbol_only_report,
    train_arm,
    wrap_prompt,
)


@pytest.fixture(scope="module")
def cases():
    return build_train_cases()


@pytest.fixture(scope="module")
def corpora(cases):
    return {arm: build_arm_corpus(arm, cases) for arm in ARMS}


def _score(**overrides) -> SuiteScoreV1:
    base = {
        "canonical_accuracy": 1.0,
        "facts_accuracy": 1.0,
        "prompt_invariance_rate": 1.0,
        "relevant_sensitivity": 1.0,
        "invariant_rate": 1.0,
        "schema_sensitivity_passed": True,
        "determinacy_calibrated": True,
        "cap0_retention_accuracy": 1.0,
        "gate_passed": True,
        "gate_failures": (),
        "decisions": {},
    }
    base.update(overrides)
    return SuiteScoreV1(**base)


def _winning_paired(n: int = 64) -> list[dict]:
    return [
        {
            "label": f"{candidate}:vs_{control}",
            "n": n,
            "control_rate": 0.25,
            "candidate_rate": 0.75,
            "mcnemar": {
                "effect": 0.5,
                "control_only": 4,
                "candidate_only": 36,
                "p_value": 0.001,
            },
            "meets_preregistered_effect": True,
        }
        for candidate in FILTERED_ARMS
        for control in CONTROL_ARMS
    ]


def _decide(**overrides):
    kwargs = {
        "filtered_scores": {arm: _score() for arm in FILTERED_ARMS},
        "control_predictions": {
            arm: {"r1": f"{arm}-a", "r2": f"{arm}-b"} for arm in CONTROL_ARMS
        },
        "filtered_predictions": {
            arm: {"r1": f"{arm}-a", "r2": f"{arm}-b"} for arm in FILTERED_ARMS
        },
        "paired": _winning_paired(),
        "symbol_only": {"passed": True, "violations": ()},
        "contamination_findings": [],
        "binding": {
            "checkpoint_sha256": "a" * 64,
            "corpus_sha256": "b" * 64,
            "suite_sha256": "c" * 64,
            "suite_schema_version": "cap1_two_pack_freeze/v1",
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
    # Same root families and same targets across every arm.
    targets = {
        arm: sorted({record.openui for record in corpus.records})
        for arm, corpus in corpora.items()
    }
    assert len({tuple(value) for value in targets.values()}) == 1
    # Schema side-channel only where declared.
    for arm, corpus in corpora.items():
        has_schema = any(record.prompt.startswith("schema:") for record in corpus.records)
        assert has_schema == (arm in SCHEMA_ARMS)
        for record in corpus.records:
            assert (
                record.prompt.startswith("schema:")
                == ARM_LEVERS[arm]["schema_side_channel"]
            )
    # Ambiguity filtering removes underdeterminate prompts.
    for arm in ("SCHEMA_FILTERED_SINGLE", "SCHEMA_FILTERED_MULTI"):
        styles = {record.meta.get("style") for record in corpora[arm].records}
        assert "ambiguous" not in styles
    assert "ambiguous" in {
        record.meta.get("style") for record in corpora["SCHEMA_UNFILTERED"].records
    }
    # Single-style arm carries exactly one accepted style per case.
    singles = {
        (record.meta.get("case_id"), record.meta.get("style"))
        for record in corpora["SCHEMA_FILTERED_SINGLE"].records
    }
    case_ids = {case for case, _ in singles}
    assert all(sum(1 for c, _ in singles if c == case) == 1 for case in case_ids)
    # Identical preregistered model/training config for every arm.
    prereg = build_preregistration()
    assert prereg.arm_levers == ARM_LEVERS
    assert set(prereg.arms) == set(ARMS)


def test_wrap_prompt_arm_conditioned() -> None:
    assert wrap_prompt("NL_NO_SCHEMA", "{}", "do x") == "do x"
    wrapped = wrap_prompt("SCHEMA_FILTERED_MULTI", '{"a": 1}', "do x")
    assert wrapped.startswith("schema:") and wrapped.endswith("do x")


def test_train_eval_roots_split_disjoint(cases) -> None:
    suite = build_cap1_two_pack_suite(seed=7)
    from slm_training.harnesses.train_data.semantic_counterfactuals import (
        root_family_for,
    )

    train_families = {root_family_for(c.canonical_target, c.pack_id) for c in cases}
    eval_families = {row.root_family_id for row in suite.rows}
    assert train_families.isdisjoint(eval_families)
    train_targets = {c.canonical_target.strip() for c in cases}
    eval_targets = {t.strip() for row in suite.rows for t in row.targets}
    assert train_targets.isdisjoint(eval_targets)


def test_symbol_only_report(cases, corpora) -> None:
    suite = build_cap1_two_pack_suite(seed=7)
    report = symbol_only_report(cases, corpora, suite)
    assert report["passed"] is True
    assert report["train_targets_checked"] > 0
    assert report["eval_targets_checked"] > 0


def test_prepare_context_clean() -> None:
    context = prepare_context()
    assert context.contamination == ()
    assert context.suite_integrity_ok is True
    assert context.symbol_only["passed"] is True
    assert len(context.suite.rows) > 0


# --------------------------------------------------------------------------- #
# Certificate paths
# --------------------------------------------------------------------------- #


def test_certificate_issued_on_synthetic_win() -> None:
    decision = _decide()
    assert decision.status == "issued"
    assert decision.certificate == CERT_CAP1
    assert decision.rejection_codes == ()
    artifact = decision.artifact
    assert artifact["passed"] is True
    assert artifact["capability"] == "CAP1_SEMANTICS"
    binding = artifact["binding"]
    for key in (
        "checkpoint_sha256",
        "corpus_sha256",
        "suite_sha256",
        "suite_schema_version",
        "config_sha256",
        "code_commit",
        "evidence_json_uri",
    ):
        assert binding[key]


def test_reject_ignores_schema() -> None:
    # Relevant schema perturbations no longer change the filtered decisions.
    scores = {arm: _score() for arm in FILTERED_ARMS}
    scores["SCHEMA_FILTERED_MULTI"] = _score(relevant_sensitivity=0.5)
    decision = _decide(filtered_scores=scores)
    assert decision.status == "rejected"
    assert StopRule.IGNORES_SCHEMA.value in decision.rejection_codes


def test_reject_fails_hard_contrasts_no_paired_win() -> None:
    decision = _decide(paired=[{"label": "x", "n": 64, "meets_preregistered_effect": False}])
    assert decision.status == "rejected"
    assert StopRule.FAILS_HARD_CONTRASTS.value in decision.rejection_codes


def test_reject_fails_hard_contrasts_invariant_flip() -> None:
    # Irrelevant schema replays must stay invariant; a flip fails the contrast.
    scores = {arm: _score() for arm in FILTERED_ARMS}
    scores["SCHEMA_FILTERED_SINGLE"] = _score(invariant_rate=0.5)
    decision = _decide(filtered_scores=scores)
    assert decision.status == "rejected"
    assert StopRule.FAILS_HARD_CONTRASTS.value in decision.rejection_codes


def test_reject_only_one_control_beaten() -> None:
    paired = [
        row for row in _winning_paired() if "vs_NL_NO_SCHEMA" in row["label"]
    ] + [
        {
            "label": f"{candidate}:vs_SCHEMA_UNFILTERED",
            "n": 64,
            "meets_preregistered_effect": False,
        }
        for candidate in FILTERED_ARMS
    ]
    decision = _decide(paired=paired)
    assert StopRule.FAILS_HARD_CONTRASTS.value in decision.rejection_codes


def test_reject_cap0_regression() -> None:
    scores = {arm: _score() for arm in FILTERED_ARMS}
    scores["SCHEMA_FILTERED_MULTI"] = _score(
        cap0_retention_accuracy=0.5,
        gate_passed=False,
        gate_failures=("cap0_retention_perfect",),
    )
    decision = _decide(filtered_scores=scores)
    assert decision.status == "rejected"
    assert StopRule.CAP0_REGRESSION.value in decision.rejection_codes


def test_reject_underpowered_fixture_n() -> None:
    paired = [
        {**row, "n": 16} for row in _winning_paired(n=16)
    ]
    decision = _decide(paired=paired)
    assert decision.status == "rejected"
    assert StopRule.UNDERPOWERED.value in decision.rejection_codes
    assert decision.certificate is None
    assert decision.artifact["paired_n"] == 16  # reported, not hidden


def test_reject_contaminated() -> None:
    decision = _decide(contamination_findings=["split_leakage:row-1"])
    assert decision.status == "rejected"
    assert StopRule.CONTAMINATED.value in decision.rejection_codes
    # Symbol-only violations contaminate the run as well.
    decision = _decide(symbol_only={"passed": False, "violations": ("r1: bad marker",)})
    assert StopRule.CONTAMINATED.value in decision.rejection_codes


def test_reject_prediction_identical() -> None:
    decision = _decide(
        filtered_predictions={
            "SCHEMA_FILTERED_SINGLE": {"r1": "NL_NO_SCHEMA-a", "r2": "NL_NO_SCHEMA-b"},
            "SCHEMA_FILTERED_MULTI": {"r1": "x", "r2": "y"},
        }
    )
    assert decision.status == "rejected"
    assert StopRule.PREDICTION_IDENTICAL.value in decision.rejection_codes


def test_cap2_closed_on_rejection() -> None:
    assert _decide().artifact["cap2_natural_language"] == "open"
    rejected = _decide(contamination_findings=["x"])
    assert rejected.artifact["cap2_natural_language"] == "CLOSED (stop rule)"


# --------------------------------------------------------------------------- #
# Replay probe + paired evidence + determinism
# --------------------------------------------------------------------------- #


def test_replay_probe_contradiction_changes_irrelevant_invariant() -> None:
    # Frozen-suite scorer semantics: relevant replays flip, irrelevant hold.
    suite = build_cap1_two_pack_suite(seed=7)
    from slm_training.harnesses.experiments.slm368_cap1_experiment import (
        score_predictions,
    )
    from slm_training.harnesses.train_data.cap1_two_pack_freeze import (
        STRATUM_SCHEMA_CONTRADICTORY,
        STRATUM_SCHEMA_EMPTY,
    )

    # Oracle decisions: canonical target on original/invariant strata; an
    # alternative candidate on schema_empty/contradictory; the counterfactual
    # target on the counterfactual stratum (already a changed decision).
    predictions: dict[str, str] = {}
    for row in suite.rows:
        if row.ambiguity_disposition == "excluded":
            continue
        if row.stratum in (STRATUM_SCHEMA_EMPTY, STRATUM_SCHEMA_CONTRADICTORY):
            predictions[row.row_id] = row.candidates[1]
        elif row.targets:
            predictions[row.row_id] = row.targets[0]
        else:
            predictions[row.row_id] = ""
    score, gate = score_predictions(suite, predictions)
    assert score.relevant_sensitivity == 1.0  # contradiction affects decisions
    assert score.invariant_rate == 1.0  # irrelevant stays invariant
    assert all(stratum not in score.decisions for stratum in ())
    assert score.cap0_retention_accuracy == 1.0
    assert gate.passed is True


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


def test_determinism(cases) -> None:
    first = {arm: build_arm_corpus(arm, cases) for arm in ARMS}
    second = {arm: build_arm_corpus(arm, cases) for arm in ARMS}
    assert all(first[arm].sha256 == second[arm].sha256 for arm in ARMS)
    assert build_preregistration().sha256 == build_preregistration().sha256
    assert build_train_cases() == cases
    d1, d2 = _decide(), _decide()
    assert d1.to_dict() == d2.to_dict()


def test_train_arm_deterministic_small(corpora) -> None:
    prereg = build_preregistration()
    m1 = train_arm(
        corpora["NL_NO_SCHEMA"].records,
        steps=1,
        batch_size=2,
        lr=prereg.lr,
        seed=0,
        model_config=prereg.model_config,
    )
    m2 = train_arm(
        corpora["NL_NO_SCHEMA"].records,
        steps=1,
        batch_size=2,
        lr=prereg.lr,
        seed=0,
        model_config=prereg.model_config,
    )
    s1 = {k: v.sum().item() for k, v in m1.state_dict().items()}
    s2 = {k: v.sum().item() for k, v in m2.state_dict().items()}
    assert s1 == s2
