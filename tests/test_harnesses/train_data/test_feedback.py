"""Synthesis-feedback loop tests (per-family yields → recommendations)."""

from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.feedback import build_synthesis_feedback


def _admitted(record_id: str, family: str, synth: str | None = None) -> ExampleRecord:
    meta: dict = {"source_family": family}
    if synth:
        meta["synth"] = synth
    return ExampleRecord(
        id=record_id,
        prompt=f"prompt {record_id}",
        openui="root = Stack([x])",
        placeholders=[],
        split="train",
        meta=meta,
    )


def _reject(stage: str, reason: str, family: str, synth: str | None = None) -> dict:
    detail: dict = {"source_family": family}
    if synth:
        detail["synth"] = synth
    return {"id": f"{family}-{reason}", "stage": stage, "reason": reason, "detail": detail}


def test_feedback_groups_yields_and_recommends() -> None:
    admitted = [
        _admitted("a1", "gold_family", synth="template"),
        _admitted("a2", "gold_family"),
        *[_admitted(f"c{i}", "weak_family") for i in range(3)],
    ]
    rejections = [
        *[_reject("dedup", "fuzzy_minhash", "dup_family", synth="layout") for _ in range(12)],
        *[_reject("quality", "quality_gate_failed", "weak_family") for _ in range(7)],
        _reject("decontamination", "ngram_overlap", "gold_family"),
    ]
    feedback = build_synthesis_feedback(
        version="vtest",
        profile="strict",
        built_at="2026-07-18T00:00:00Z",
        admitted=admitted,
        rejections=rejections,
        quality_report={"warnings": [{"code": "high_rejection_rate"}]},
    )

    families = feedback["families"]
    assert families["dup_family"]["candidates"] == 12
    assert families["dup_family"]["yield"] == 0.0
    assert families["weak_family"]["yield"] == 0.3
    assert families["gold_family"]["admitted"] == 2
    assert feedback["synthesizers"]["layout"]["rejected"] == 12
    assert feedback["warnings"] == [{"code": "high_rejection_rate"}]

    codes = {(r["code"], r["target"]) for r in feedback["recommendations"]}
    assert ("redundant_expansion", "dup_family") in codes
    assert ("low_yield", "weak_family") in codes
    # gold_family only has 3 candidates — below the min group size for
    # yield/dup recommendations, but leakage is always flagged.
    assert ("eval_leakage_source", "gold_family") in codes

    hypotheses = [e["hypothesis"] for e in feedback["experiment_candidates"]]
    assert any("dup_family" in h for h in hypotheses)
    assert all(
        {"hypothesis", "rationale", "expected_effect", "falsification_criteria", "knobs"}
        <= set(e)
        for e in feedback["experiment_candidates"]
    )


def test_small_groups_do_not_trigger_noise() -> None:
    feedback = build_synthesis_feedback(
        version="v",
        profile="strict",
        built_at="t",
        admitted=[_admitted("a", "tiny")],
        rejections=[_reject("dedup", "exact_pair_duplicate", "tiny")],
        quality_report={},
    )
    assert feedback["families"]["tiny"]["candidates"] == 2
    assert feedback["recommendations"] == []
    assert feedback["experiment_candidates"] == []
