from __future__ import annotations

import pytest

from slm_training.evals.judge_independence import (
    ExternalJudgeAdapter,
    ExternalJudgeConfig,
    JudgeEvidenceV1,
    analyze_triple_judges,
    binary_metrics,
    bootstrap_pairwise_ci,
    calibration_error,
    cohen_kappa,
    fleiss_kappa,
    kendall_tau_b,
    judge_evidence_agentv_cases,
    set_jaccard,
    spearman,
)

SHA = "a" * 64


def evidence(**overrides: object) -> JudgeEvidenceV1:
    values = {
        "evidence_id": "ev_01",
        "audit_id": "efs0-04",
        "record_id": "record_01",
        "generation_id": "generation_01",
        "pair_id": "pair_01",
        "judge_id": "external_01",
        "provenance": "external_model",
        "prompt_sha256": SHA,
        "left_output_sha256": SHA,
        "right_output_sha256": SHA,
        "left_checkpoint_sha256": "b" * 64,
        "right_checkpoint_sha256": "c" * 64,
        "candidate_model_families": ("twotower", "choice"),
        "prior_judge_providers": ("repository",),
        "prior_judge_model_families": ("binding-aware-v2",),
        "provider": "independent-provider",
        "provider_version": "api-2026-07",
        "model_family": "independent-family",
        "model_version": "independent-model-v1",
        "rubric_id": "semantic-pair",
        "rubric_version": "1",
        "rubric_sha256": SHA,
        "participated_in_creation": False,
        "participated_in_admission": False,
        "participated_in_training": False,
        "participated_in_preference": False,
        "participated_in_evaluation": False,
        "rubric_used_for_training_admission": False,
        "verdict": "left",
        "left_acceptable": True,
        "right_acceptable": False,
        "score": 0.8,
        "reason_codes": ("role_match",),
        "confidence": 0.7,
        "created_at": "2026-07-17T12:00:00Z",
        "blinded": True,
        "order_seed": 7,
        "order_sha256": SHA,
        "saw_candidate_identity": False,
        "saw_automatic_judgments": False,
    }
    values.update(overrides)
    return JudgeEvidenceV1(**values)  # type: ignore[arg-type]


def test_judge_evidence_round_trip_and_external_blinding() -> None:
    row = evidence()
    assert JudgeEvidenceV1.from_dict(row.to_dict()) == row
    assert row.independent is True

    overlapping = evidence(model_family="twotower")
    assert overlapping.independent is False
    with pytest.raises(ValueError, match="independence"):
        overlapping.require_independent()

    participated = evidence(participated_in_admission=True)
    assert participated.independent is False
    reused_provider = evidence(provider="repository")
    assert reused_provider.independent is False


def test_human_evidence_requires_blind_role_and_consistent_refusal() -> None:
    row = evidence(
        provenance="human",
        judge_id="ann_12345678",
        provider="human",
        provider_version="study-v1",
        model_family="human",
        model_version="opaque-rater",
        annotator_role="rater",
    )
    assert row.provenance == "human"

    with pytest.raises(ValueError, match="annotator role"):
        evidence(provenance="human")
    with pytest.raises(ValueError, match="refused"):
        evidence(refused=True)


def test_agreement_statistics_handle_ties_and_degenerate_inputs() -> None:
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert cohen_kappa(["a"], ["a"]) is None
    assert fleiss_kappa([["a", "a"], ["b", "b"]]) == pytest.approx(1.0)
    assert set_jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert binary_metrics([True, True, False], [True, False, True]) == {
        "n": 3,
        "precision": 0.5,
        "recall": 0.5,
    }
    assert calibration_error([0.9, 0.1], [True, False], bins=2) == pytest.approx(0.1)
    interval = bootstrap_pairwise_ci(
        [1, 2, 3, 4], [1, 2, 3, 4], spearman, seed=3, resamples=50
    )
    assert interval["estimate"] == pytest.approx(1.0)
    assert interval["low"] == pytest.approx(1.0)
    assert interval["high"] == pytest.approx(1.0)


def test_triple_judge_analysis_reports_disagreement_and_ambiguity_sensitivity() -> None:
    rows = []
    values = [
        ("p1", 1.0, 1.0, 1.0, "left", "left", "left", True, True, True, False),
        ("p2", 0.0, 0.0, 0.0, "right", "right", "right", False, False, False, False),
        ("p3", 1.0, 0.0, 1.0, "left", "right", "left", True, False, True, False),
        ("p4", 0.0, 0.0, 0.0, "right", "right", "right", False, False, False, True),
    ]
    for value in values:
        (
            pair_id,
            deterministic_score,
            external_score,
            human_score,
            deterministic_verdict,
            external_verdict,
            human_verdict,
            deterministic_pass,
            external_pass,
            human_pass,
            ambiguous,
        ) = value
        rows.append(
            {
                "pair_id": pair_id,
                "deterministic_score": deterministic_score,
                "external_score": external_score,
                "human_score": human_score,
                "deterministic_verdict": deterministic_verdict,
                "external_verdict": external_verdict,
                "human_verdict": human_verdict,
                "deterministic_pass": deterministic_pass,
                "external_pass": external_pass,
                "human_pass": human_pass,
                "human_ambiguous": ambiguous,
                "external_confidence": 0.8,
                "external_cost_usd": 0.01,
                "external_latency_ms": 100,
                "reason_codes": ["binding_failure"] if pair_id == "p3" else [],
                "checkpoint_family": "twotower",
                "suite": "held_out",
            }
        )
    report = analyze_triple_judges(rows)
    assert report["pair_n"] == 4
    assert report["disagreement"]["pair_n"] == 1
    assert report["disagreement"]["reason_clusters"] == {"binding_failure": 1}
    assert report["full"]["external_vs_human"]["precision"] == pytest.approx(1.0)
    assert report["full"]["external_vs_human"]["recall"] == pytest.approx(0.5)
    assert report["excluding_ambiguous_human_pairs"]["pair_n"] == 3
    assert report["external_operations"]["total_cost_usd"] == pytest.approx(0.04)


def test_agentv_cases_check_evidence_integrity_not_semantic_quality() -> None:
    case = judge_evidence_agentv_cases([evidence()])[0]
    assert case["pass"] is True
    assert case["metadata"]["semantic_judge"] is False

    failed = judge_evidence_agentv_cases([evidence(participated_in_training=True)])[0]
    assert failed["pass"] is False
    assert "independence" in failed["failures"][0]


def test_external_adapter_retries_schema_failure_and_never_sends_identity() -> None:
    requests: list[dict[str, object]] = []

    def transport(request: dict[str, object]) -> dict[str, object]:
        requests.append(request)
        if len(requests) == 1:
            return {}
        return {
            "verdict": "left",
            "left_acceptable": True,
            "right_acceptable": False,
            "score": 0.75,
            "confidence": 0.8,
            "reason_codes": ["role_match"],
            "cost_usd": 0.002,
        }

    adapter = ExternalJudgeAdapter(
        ExternalJudgeConfig(
            provider="other-provider",
            provider_version="api-2026-07",
            model_family="other-family",
            model_version="other-family-v1",
            rubric_id="semantic-pair",
            rubric_version="1",
            rubric="Judge prompt-role and binding fidelity.",
            temperature=0,
            seed=7,
            max_attempts=2,
            max_tokens=512,
            max_cost_usd=0.01,
        ),
        transport,
    )
    result = adapter.judge(
        audit_id="efs0-04",
        record_id="record_01",
        generation_id="generation_01",
        pair_id="pair_01",
        prompt="Profile card",
        left_openui='Card(Text("A"))',
        right_openui='Card(Text("B"))',
        left_checkpoint_sha256="b" * 64,
        right_checkpoint_sha256="c" * 64,
        candidate_model_families=("twotower", "choice"),
        prior_judge_providers=("repository",),
        prior_judge_model_families=("binding-aware-v2",),
        order_seed=7,
    )

    assert result.retry_count == 1
    assert result.independent is True
    assert result.cost_usd == pytest.approx(0.002)
    assert all("model_family" not in str(request) for request in requests)
    assert all("deterministic" not in str(request) for request in requests)


def test_external_adapter_records_exhausted_errors() -> None:
    adapter = ExternalJudgeAdapter(
        ExternalJudgeConfig(
            provider="other-provider",
            provider_version="api-2026-07",
            model_family="other-family",
            model_version="other-family-v1",
            rubric_id="semantic-pair",
            rubric_version="1",
            rubric="Judge semantic fidelity.",
            temperature=0,
            seed=None,
            max_attempts=2,
            max_tokens=256,
            max_cost_usd=0.01,
        ),
        lambda _request: {},
    )
    result = adapter.judge(
        audit_id="efs0-04",
        record_id="record_01",
        generation_id="generation_01",
        pair_id="pair_01",
        prompt="Profile card",
        left_openui='Card(Text("A"))',
        right_openui='Card(Text("B"))',
        left_checkpoint_sha256="b" * 64,
        right_checkpoint_sha256="c" * 64,
        candidate_model_families=("twotower", "choice"),
        prior_judge_providers=("repository",),
        prior_judge_model_families=("binding-aware-v2",),
        order_seed=7,
    )
    assert result.verdict == "error"
    assert result.error == "KeyError"
    assert result.retry_count == 1


def test_external_adapter_records_refusal_without_acceptability_labels() -> None:
    config = ExternalJudgeConfig(
        provider="other-provider",
        provider_version="api-2026-07",
        model_family="other-family",
        model_version="other-family-v1",
        rubric_id="semantic-pair",
        rubric_version="1",
        rubric="Judge semantic fidelity.",
        temperature=0,
        seed=None,
        max_attempts=1,
        max_tokens=256,
        max_cost_usd=0.01,
    )
    adapter = ExternalJudgeAdapter(
        config, lambda _request: {"verdict": "refusal", "reason_codes": []}
    )
    result = adapter.judge(
        audit_id="efs0-04",
        record_id="record_01",
        generation_id="generation_01",
        pair_id="pair_01",
        prompt="Profile card",
        left_openui='Card(Text("A"))',
        right_openui='Card(Text("B"))',
        left_checkpoint_sha256="b" * 64,
        right_checkpoint_sha256="c" * 64,
        candidate_model_families=("twotower", "choice"),
        prior_judge_providers=("repository",),
        prior_judge_model_families=("binding-aware-v2",),
        order_seed=7,
    )
    assert result.verdict == "refusal"
    assert result.refused is True
    assert result.left_acceptable is None
