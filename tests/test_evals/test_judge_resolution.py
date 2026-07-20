from __future__ import annotations

import pytest

from slm_training.evals.judge_resolution import (
    JudgeInvocationEnvelopeV1,
    JudgeResolutionItemV1,
    SemanticResolutionEndpointV1,
    SemanticResolutionManifestV1,
    abstention_aware_scores,
    apply_resolution_manifest,
    brier_score,
    build_fixture_corpus,
    build_resolution_manifest,
    classify_delta,
    equivalence_invariance_error_rate,
    expected_calibration_error,
    flip_rate,
    krippendorff_alpha,
    pairwise_ordering_consistency,
    perturbation_detection_rate,
    run_resolution_fixture,
    test_retest_reliability,
)

SHA = "a" * 64


def envelope(**overrides: object) -> JudgeInvocationEnvelopeV1:
    values = {
        "provider": "fixture",
        "model": "test",
        "revision": "1",
        "system_sha256": SHA,
        "rubric_sha256": SHA,
        "prompt_sha256": SHA,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "retry_policy": {},
        "response_digest": "test",
        "parser_version": "v1",
        "prior_judge_families": (),
        "independent": True,
        "participated_in_creation": False,
        "participated_in_admission": False,
        "participated_in_training": False,
    }
    values.update(overrides)
    return JudgeInvocationEnvelopeV1(**values)  # type: ignore[arg-type]


def item(
    expected_class: str,
    verdicts: dict[str, tuple[str | None, ...]],
    scores: dict[str, tuple[float | None, ...]] | None = None,
    abstentions: dict[str, tuple[bool, ...]] | None = None,
) -> JudgeResolutionItemV1:
    return JudgeResolutionItemV1(
        item_id="i1",
        pair_group="test",
        source_a="a",
        source_b="b",
        transformation_provenance="test",
        expected_class=expected_class,  # type: ignore[arg-type]
        canonical_fingerprint_a=SHA,
        canonical_fingerprint_b=SHA,
        human_label=None,
        repeated_scores_per_endpoint=scores or {},
        verdicts_per_endpoint=verdicts,
        abstentions_per_endpoint=abstentions or {},
    )


def test_flip_rate_computes_adjacent_disagreement() -> None:
    # 3 adjacent pairs: (a,a)=same, (a,b)=diff, (b,b)=same -> 1/3.
    assert flip_rate(["a", "a", "b", "b"]) == pytest.approx(1 / 3)
    assert flip_rate(["a", "a", "a"]) == pytest.approx(0.0)
    assert flip_rate([["a", "a"], ["a", "b"]]) == pytest.approx(0.5)
    # 2 adjacent pairs: (0.2,0.8) crosses 0.5, (0.8,0.9) stays above -> 1/2.
    assert flip_rate([0.2, 0.8, 0.9], threshold=0.5) == pytest.approx(1 / 2)
    assert flip_rate([]) is None


def test_krippendorff_alpha_degrades_gracefully() -> None:
    assert krippendorff_alpha([["a", "a"], ["b", "b"]]) == pytest.approx(1.0)
    assert krippendorff_alpha([["a", "a"], ["a", "a"]]) is None
    assert krippendorff_alpha([]) is None
    assert krippendorff_alpha([["a", "b", "a"], ["b", "a", "b"]]) is not None


def test_pairwise_ordering_consistency_handles_ties_and_small_inputs() -> None:
    result = pairwise_ordering_consistency([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], repeats=5)
    assert result is not None
    assert result["spearman"] == pytest.approx(1.0)
    assert result["kendall_tau_b"] == pytest.approx(1.0)
    assert pairwise_ordering_consistency([1.0], [1.0], repeats=1) is None


def test_brier_score_and_calibration_basic() -> None:
    assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert brier_score([0.8, 0.2], [True, False]) == pytest.approx(0.02 + 0.02)
    assert expected_calibration_error([0.9, 0.1], [True, False], bins=2) == pytest.approx(0.1)


def test_retest_reliability_returns_icc_components() -> None:
    result = test_retest_reliability([1.0, 1.0, 1.0, 1.0])
    assert "icc" in result
    assert result["n_clusters"] == 4


def test_abstention_aware_scores_excludes_abstentions() -> None:
    result = abstention_aware_scores(
        ["left", "right", "equivalent", "left"], [False, True, False, False]
    )
    assert result is not None
    assert result["abstention_rate"] == pytest.approx(0.25)
    # Abstained "right" is excluded; remaining three are all passes.
    assert result["scored_n"] == 3
    assert result["pass_rate"] == pytest.approx(1.0)



def test_equivalence_invariance_and_perturbation_detection() -> None:
    items = [
        item("canonical_equivalent", {"e1": ("equivalent", "equivalent")}),
        item("canonical_equivalent", {"e1": ("different", "equivalent")}),
        item("semantic_error", {"e1": ("different", "different")}),
    ]
    assert equivalence_invariance_error_rate(items, "e1") == pytest.approx(0.5)
    assert perturbation_detection_rate(items, "e1") == pytest.approx(1.0)
    assert equivalence_invariance_error_rate([], "e1") is None
    assert perturbation_detection_rate([], "e1") is None


def test_envelope_round_trip_and_validation() -> None:
    env = envelope()
    assert JudgeInvocationEnvelopeV1.from_dict(env.to_dict()) == env
    with pytest.raises(ValueError, match="sha256"):
        envelope(system_sha256="not-a-hash")
    with pytest.raises(ValueError, match="temperature"):
        envelope(temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        envelope(top_p=1.5)


def test_endpoint_and_manifest_round_trip() -> None:
    ep = SemanticResolutionEndpointV1(
        endpoint_label="e1",
        provider="fixture",
        model="m",
        revision="1",
        metric_family="f",
        measured_flip_rate=0.0,
        cohen_kappa=1.0,
        fleiss_kappa=1.0,
        krippendorff_alpha=1.0,
        icc_1_1={"icc": 0.9},
        pairwise_ordering_consistency=None,
        equivalence_invariance_error_rate=0.0,
        perturbation_detection_rate=1.0,
        brier_score=0.1,
        ece=0.05,
        abstention_rate=0.0,
        required_repeats=5,
        majority_rule=False,
        minimum_resolvable_delta=0.05,
        equivalence_margin=0.01,
        claim_language_permit_set=("directional", "resolved", "below_noise_floor"),
        requires_independent_confirmation=True,
    )
    assert SemanticResolutionEndpointV1.from_dict(ep.to_dict()) == ep

    manifest = build_resolution_manifest(
        [item("canonical_equivalent", {"e1": ("equivalent",)})],
        [ep],
        corpus_path="/tmp/corpus.jsonl",
        envelopes=[envelope()],
        run_id="test",
    )
    assert SemanticResolutionManifestV1.from_dict(manifest.to_dict()).schema == "SemanticResolutionManifestV1"


def test_classify_delta_respects_floors() -> None:
    ep = SemanticResolutionEndpointV1(
        endpoint_label="e1",
        provider="fixture",
        model="m",
        revision="1",
        metric_family="f",
        measured_flip_rate=0.0,
        cohen_kappa=1.0,
        fleiss_kappa=1.0,
        krippendorff_alpha=1.0,
        icc_1_1={"icc": 0.9},
        pairwise_ordering_consistency=None,
        equivalence_invariance_error_rate=0.0,
        perturbation_detection_rate=1.0,
        brier_score=0.1,
        ece=0.05,
        abstention_rate=0.0,
        required_repeats=5,
        majority_rule=False,
        minimum_resolvable_delta=0.05,
        equivalence_margin=0.01,
        claim_language_permit_set=("directional", "resolved", "below_noise_floor"),
        requires_independent_confirmation=True,
    )
    manifest = build_resolution_manifest([], [ep], run_id="test")
    assert classify_delta(0.005, "e1", manifest) == "below_noise_floor"
    assert classify_delta(0.02, "e1", manifest) == "resolved"
    assert classify_delta(0.1, "e1", manifest) == "directional"
    assert classify_delta(-0.1, "e1", manifest) == "directional"
    assert classify_delta(0.1, "missing", manifest) == "below_noise_floor"


def test_apply_resolution_manifest_annotates_without_mutating_raw() -> None:
    ep = SemanticResolutionEndpointV1(
        endpoint_label="fixture_binding_aware_v2",
        provider="fixture",
        model="m",
        revision="1",
        metric_family="f",
        measured_flip_rate=0.0,
        cohen_kappa=1.0,
        fleiss_kappa=1.0,
        krippendorff_alpha=1.0,
        icc_1_1={"icc": 0.9},
        pairwise_ordering_consistency=None,
        equivalence_invariance_error_rate=0.0,
        perturbation_detection_rate=1.0,
        brier_score=0.1,
        ece=0.05,
        abstention_rate=0.0,
        required_repeats=5,
        majority_rule=False,
        minimum_resolvable_delta=0.05,
        equivalence_margin=0.01,
        claim_language_permit_set=("directional", "resolved", "below_noise_floor"),
        requires_independent_confirmation=True,
    )
    manifest = build_resolution_manifest([], [ep], run_id="test")
    scoreboard = {
        "suites": {
            "smoke": {"fixture_binding_aware_v2": 0.8},
            "held_out": {"fixture_binding_aware_v2": 0.02},
        }
    }
    annotated = apply_resolution_manifest(scoreboard, manifest)
    assert annotated["suites"]["smoke"]["fixture_binding_aware_v2"] == 0.8
    assert "semantic_resolution" in annotated
    assert annotated["semantic_resolution"]["fixture_binding_aware_v2"]["deltas"]["smoke"] == "directional"
    assert annotated["semantic_resolution"]["fixture_binding_aware_v2"]["deltas"]["held_out"] == "resolved"


def test_fixture_corpus_has_expected_groups_and_canonical_equivalence() -> None:
    items, envelopes = build_fixture_corpus(repeats=3)
    assert len(envelopes) == 2
    groups = {item.pair_group for item in items}
    assert groups == {"canonical_equivalent", "semantic_error", "historical_delta"}

    ce = [item for item in items if item.pair_group == "canonical_equivalent"]
    assert len(ce) == 6
    for item in ce:
        assert item.canonical_fingerprint_a == item.canonical_fingerprint_b
        assert "fixture_binding_aware_v2" in item.verdicts_per_endpoint
        assert len(item.verdicts_per_endpoint["fixture_binding_aware_v2"]) == 3

    se = [item for item in items if item.pair_group == "semantic_error"]
    assert len(se) == 6
    for item in se:
        assert item.canonical_fingerprint_a != item.canonical_fingerprint_b

    hd = [item for item in items if item.pair_group == "historical_delta"]
    assert len(hd) == 3


def test_run_resolution_fixture_produces_manifest_with_metrics() -> None:
    manifest, items = run_resolution_fixture(repeats=3, run_id="test-run")
    assert len(items) == 15
    assert len(manifest.endpoints) == 3
    assert manifest.status == "fixture"
    assert manifest.claim_class == "wiring"
    for ep in manifest.endpoints:
        assert ep.required_repeats == 3
        assert ep.icc_1_1 is not None

    binding_ep = next(
        ep for ep in manifest.endpoints if ep.endpoint_label == "fixture_binding_aware_v2"
    )
    assert binding_ep.equivalence_invariance_error_rate == pytest.approx(0.0)
    assert binding_ep.perturbation_detection_rate == pytest.approx(1.0)
