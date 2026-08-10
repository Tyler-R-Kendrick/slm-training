"""SIE-003 (SLM-477) regression tests: prompt-to-partial-requirements
prediction behind the advisory-only seam."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tests.casefiles import case_values

from slm_training.data.progspec.prompt_requirements import PromptSemanticRequirementsV1
from slm_training.data.semantic_plan.compiler import EvidenceKind
from slm_training.data.semantic_plan.requirements_integrate import requirements_hard_prune_allowed
from slm_training.models.prompt_factor_predictor import (
    DEFAULT_AUTHORIZED_FACTOR_FAMILIES,
    PREDICTOR_SCHEMA_VERSION,
    PromptFactorExampleV1,
    PromptFactorPredictionV1,
    TinyBagOfWordsClassifier,
    calibrate_abstention_threshold,
    deterministic_keyword_predict,
    frequency_baseline_predict,
    predict_factor,
    prediction_to_requirement_fact,
    resolve_confidence_threshold,
)

_CORPUS_PATH = Path("src/slm_training/resources/data/eval/openui_hard_valid_v1/records.jsonl")


def _synthetic_examples() -> list[PromptFactorExampleV1]:
    # Two trivially separable vocabularies -- proves the classifier/
    # calibration machinery responds to real signal, independent of
    # whether this particular fixture corpus's signal clears the bar.
    column_prompts = [
        "stack these fields in a vertical column layout",
        "arrange the items in a column, top to bottom",
        "make a vertical stack of the components",
        "column layout with items stacked vertically",
    ]
    row_prompts = [
        "arrange the buttons in a horizontal row toolbar",
        "lay the items out in a row side by side",
        "make a horizontal row of the components",
        "row layout with items placed side by side",
    ]
    examples = []
    for index, prompt in enumerate(column_prompts):
        examples.append(PromptFactorExampleV1(source_program_id=f"col_{index}", prompt=prompt, label="stack_column"))
    for index, prompt in enumerate(row_prompts):
        examples.append(PromptFactorExampleV1(source_program_id=f"row_{index}", prompt=prompt, label="stack_row"))
    return examples


# ---------------------------------------------------------------------------
# TinyBagOfWordsClassifier
# ---------------------------------------------------------------------------


def test_classifier_rejects_empty_training_set():
    with pytest.raises(ValueError, match="at least one example"):
        TinyBagOfWordsClassifier.fit([])


def test_classifier_separates_two_easily_distinguishable_vocabularies():
    examples = _synthetic_examples()
    classifier = TinyBagOfWordsClassifier.fit(examples)
    for example in examples:
        proba = classifier.predict_proba(example.prompt)
        top = max(proba, key=proba.get)
        assert top == example.label, (example.prompt, proba)


def test_classifier_predict_proba_sums_to_one():
    classifier = TinyBagOfWordsClassifier.fit(_synthetic_examples())
    proba = classifier.predict_proba("stack these vertically in a column")
    assert proba.keys() == {"stack_column", "stack_row"}
    assert sum(proba.values()) == pytest.approx(1.0)


def test_classifier_round_trips_through_dict():
    classifier = TinyBagOfWordsClassifier.fit(_synthetic_examples())
    restored = TinyBagOfWordsClassifier.from_dict(classifier.to_dict())
    assert restored == classifier


def test_classifier_rejects_malformed_shapes():
    with pytest.raises(ValueError, match="one entry per class"):
        TinyBagOfWordsClassifier(
            vocab=("a",), class_labels=("x", "y"), log_prior=(0.0,), log_likelihood=((0.0,), (0.0,))
        )
    with pytest.raises(ValueError, match="one entry per vocab word"):
        TinyBagOfWordsClassifier(
            vocab=("a", "b"), class_labels=("x",), log_prior=(0.0,), log_likelihood=((0.0,),)
        )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def test_frequency_baseline_reports_the_true_class_distribution():
    examples = _synthetic_examples()  # 4 column, 4 row
    freq = frequency_baseline_predict(examples)
    assert freq == {"stack_column": pytest.approx(0.5), "stack_row": pytest.approx(0.5)}


def test_deterministic_keyword_baseline_matches_and_abstains_honestly():
    assert deterministic_keyword_predict("please stack this in a column") == "stack_column"
    assert deterministic_keyword_predict("lay the buttons out in a row") == "stack_row"
    assert deterministic_keyword_predict("show a summary of recent orders") is None


# ---------------------------------------------------------------------------
# Calibration / abstention (VCE-010's own risk-coverage machinery)
# ---------------------------------------------------------------------------


def test_calibration_report_has_the_expected_shape():
    examples = _synthetic_examples()
    classifier = TinyBagOfWordsClassifier.fit(examples)
    report = calibrate_abstention_threshold(classifier, examples)
    assert report.keys() >= {"n", "brier", "ece", "operating_point", "abstain_required", "risk_coverage"}
    assert report["n"] == len(examples)


def test_resolve_confidence_threshold_falls_back_to_always_abstain_when_uncalibratable():
    # An empty/degenerate operating point (no viable threshold) must yield a
    # threshold no real probability in [0, 1] can ever clear.
    threshold = resolve_confidence_threshold({"operating_point": None})
    assert threshold > 1.0


def test_resolve_confidence_threshold_uses_the_real_operating_point_when_present():
    threshold = resolve_confidence_threshold({"operating_point": {"threshold": 0.6, "coverage": 0.8}})
    assert threshold == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# predict_factor / PromptFactorPredictionV1 invariants
# ---------------------------------------------------------------------------


def test_predict_factor_returns_confident_non_abstained_prediction_when_above_threshold():
    classifier = TinyBagOfWordsClassifier.fit(_synthetic_examples())
    prediction = predict_factor(
        classifier, "stack these vertically in a column", factor_family="style_layout", confidence_threshold=0.5
    )
    assert prediction.abstained is False
    assert prediction.predicted_value == "stack_column"
    assert prediction.abstention_reason is None
    assert prediction.authority == "advisory-learned"
    assert prediction.provenance == "predicted"


def test_predict_factor_abstains_below_threshold():
    classifier = TinyBagOfWordsClassifier.fit(_synthetic_examples())
    prediction = predict_factor(
        classifier, "stack these vertically in a column", factor_family="style_layout", confidence_threshold=0.999999
    )
    assert prediction.abstained is True
    assert prediction.predicted_value is None
    assert prediction.abstention_reason is not None


def test_prediction_invariants_reject_gold_and_non_advisory_authority():
    base = dict(
        schema_version=PREDICTOR_SCHEMA_VERSION,
        predictor_version="v",
        factor_family="style_layout",
        predicted_value="stack_column",
        confidence=0.9,
        abstained=False,
        abstention_reason=None,
        authority="advisory-learned",
        provenance="predicted",
    )
    PromptFactorPredictionV1(**base)  # sanity: valid construction succeeds
    with pytest.raises(ValueError, match="advisory-learned"):
        PromptFactorPredictionV1(**{**base, "authority": "compiler-hard"})
    with pytest.raises(ValueError, match="never gold"):
        PromptFactorPredictionV1(**{**base, "provenance": "oracle_override"})


def test_prediction_invariants_reject_inconsistent_abstention_state():
    base = dict(
        schema_version=PREDICTOR_SCHEMA_VERSION,
        predictor_version="v",
        factor_family="style_layout",
        confidence=0.9,
        authority="advisory-learned",
        provenance="predicted",
    )
    with pytest.raises(ValueError, match="must not carry a predicted_value"):
        PromptFactorPredictionV1(
            **base, predicted_value="stack_column", abstained=True, abstention_reason="low confidence"
        )
    with pytest.raises(ValueError, match="must carry a predicted_value"):
        PromptFactorPredictionV1(**base, predicted_value=None, abstained=False, abstention_reason=None)


# ---------------------------------------------------------------------------
# Bridging into RequirementFact -- authorization gate and gold-safety
# ---------------------------------------------------------------------------


def _confident_prediction(value: str = "stack_column") -> PromptFactorPredictionV1:
    return PromptFactorPredictionV1(
        schema_version=PREDICTOR_SCHEMA_VERSION,
        predictor_version="v",
        factor_family="style_layout",
        predicted_value=value,
        confidence=0.9,
        abstained=False,
        abstention_reason=None,
        authority="advisory-learned",
        provenance="predicted",
    )


def _abstained_prediction() -> PromptFactorPredictionV1:
    return PromptFactorPredictionV1(
        schema_version=PREDICTOR_SCHEMA_VERSION,
        predictor_version="v",
        factor_family="style_layout",
        predicted_value=None,
        confidence=0.2,
        abstained=True,
        abstention_reason="too uncertain",
        authority="advisory-learned",
        provenance="predicted",
    )


def test_prediction_to_requirement_fact_emits_a_real_fact_when_authorized_and_confident():
    prediction = _confident_prediction()
    fact = prediction_to_requirement_fact(prediction, fact_id="fact-1")
    assert fact is not None
    assert fact.factor_family == "style_layout"
    assert fact.provenance == "predicted"
    assert fact.authority == "advisory-learned"
    assert fact.disposition == "preferred"
    assert fact.confidence == pytest.approx(0.9)


def test_prediction_to_requirement_fact_fails_closed_when_not_authorized():
    prediction = _confident_prediction()
    fact = prediction_to_requirement_fact(prediction, fact_id="fact-1", authorized_factor_families=("topology",))
    assert fact is None


def test_prediction_to_requirement_fact_returns_none_on_abstention():
    fact = prediction_to_requirement_fact(_abstained_prediction(), fact_id="fact-1")
    assert fact is None


def test_default_authorized_factor_families_is_an_honest_placeholder():
    # AC: "No factor without an oracle ceiling is promoted into the model
    # merely for convenience" -- EXP-SR-1 has not run yet, so the default
    # must be a small, explicit, documented set, not "everything".
    from slm_training.models.semantic_plan_factors import PROGRAM_FACTOR_FAMILIES

    assert set(DEFAULT_AUTHORIZED_FACTOR_FAMILIES) < set(PROGRAM_FACTOR_FAMILIES)


# ---------------------------------------------------------------------------
# Legal-support parity: predictions can never hard-prune (AC: "Exact legal
# support is byte/set identical with predictor on/off").
# ---------------------------------------------------------------------------


def test_purely_predicted_facts_can_never_hard_prune():
    fact = prediction_to_requirement_fact(_confident_prediction(), fact_id="fact-1")
    assert fact is not None
    requirements = PromptSemanticRequirementsV1(facts=(fact,))
    # No evidence_kinds supplied at all -- the realistic production case.
    assert requirements_hard_prune_allowed(requirements) is False
    # Even with PREDICTION_ONLY evidence explicitly present, still no prune.
    assert requirements_hard_prune_allowed(requirements, evidence_kinds=(EvidenceKind.PREDICTION_ONLY,)) is False
    # Only independently certified compiler evidence can ever unlock pruning
    # -- and this predictor never produces that evidence kind itself.
    assert (
        requirements_hard_prune_allowed(requirements, evidence_kinds=(EvidenceKind.COMPILER_AUTHORED_CERTIFIED,))
        is True
    )


# ---------------------------------------------------------------------------
# Gold safety: predict_factor's signature has no path to a plan/oracle.
# ---------------------------------------------------------------------------


def test_predict_factor_signature_has_no_plan_or_oracle_parameter():
    import inspect

    parameters = set(inspect.signature(predict_factor).parameters)
    assert parameters == {"classifier", "text", "factor_family", "confidence_threshold", "predictor_version"}
    assert "plan" not in parameters and "oracle" not in parameters and "gold" not in parameters


# ---------------------------------------------------------------------------
# Metamorphic invariance: cosmetic (case/whitespace) changes to the prompt
# must never change the prediction -- the tokenizer already normalizes both,
# so this is a genuine, provable invariance, not just an assertion of luck.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    case_values(__file__, "test_prediction_is_invariant_to_case_and_whitespace_cosmetic_changes"),
)
def test_prediction_is_invariant_to_case_and_whitespace_cosmetic_changes(variant):
    classifier = TinyBagOfWordsClassifier.fit(_synthetic_examples())
    canonical = classifier.predict_proba("stack these vertically in a column")
    variant_proba = classifier.predict_proba(variant)
    assert canonical == variant_proba


# ---------------------------------------------------------------------------
# Real-corpus integration: a genuine "tiny trainable baseline" trained on
# committed, real (prompt, archetype) pairs -- role="positive" only (never
# the corpus's deliberately-corrupted contrast negatives), split at the
# source-program level before train/held-out (EXP-SR-2's own declared
# leakage control).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CORPUS_PATH.is_file(), reason="openui_hard_valid_v1 corpus not present")
def test_real_corpus_tiny_baseline_trains_and_calibrates_without_crashing():
    examples: list[PromptFactorExampleV1] = []
    with _CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if len(examples) >= 300:
                break
            payload = json.loads(line)
            if payload["role"] != "positive":
                continue  # never train on the corpus's deliberately-corrupted negatives
            examples.append(
                PromptFactorExampleV1(
                    source_program_id=payload["source_program_id"],
                    prompt=payload["record"]["prompt"],
                    label=payload["source_plan"]["archetype"]["id"],
                )
            )
    assert len(examples) > 50

    programs = sorted({example.source_program_id for example in examples})
    rng = random.Random(0)
    rng.shuffle(programs)
    split_at = int(len(programs) * 0.7)
    train_programs = set(programs[:split_at])
    train = [example for example in examples if example.source_program_id in train_programs]
    held_out = [example for example in examples if example.source_program_id not in train_programs]
    assert train and held_out
    # Grouped split: no source program appears on both sides.
    assert {e.source_program_id for e in train} & {e.source_program_id for e in held_out} == set()

    classifier = TinyBagOfWordsClassifier.fit(train)
    report = calibrate_abstention_threshold(classifier, held_out)
    threshold = resolve_confidence_threshold(report)

    predictions = [
        predict_factor(classifier, example.prompt, factor_family="style_layout", confidence_threshold=threshold)
        for example in held_out
    ]
    # Whatever the calibrated threshold turns out to be on this real,
    # small, templated fixture corpus, every non-abstained prediction must
    # still be a real, valid predictor version/authority/provenance record
    # -- this is a smoke test of the mechanism, not a claim about accuracy
    # (accuracy evidence-gathering is SIE-004/SLM-482's job, not this one's).
    for prediction in predictions:
        assert prediction.schema_version == PREDICTOR_SCHEMA_VERSION
        assert prediction.authority == "advisory-learned"
        assert prediction.provenance == "predicted"
        if not prediction.abstained:
            assert prediction.predicted_value in classifier.class_labels
