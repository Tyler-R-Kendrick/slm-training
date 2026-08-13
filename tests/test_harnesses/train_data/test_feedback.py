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


def test_auditor_cue_only_lift_maps_to_shortcut() -> None:
    feedback = build_synthesis_feedback(
        version="v",
        profile="strict",
        built_at="t",
        admitted=[_admitted("a", "synthetic")],
        rejections=[],
        quality_report={
            "pair_quality": {
                "passed": False,
                "blocking_reasons": ["cue_only_lift", "not_a_real_reason"],
            },
        },
    )
    codes = {item["code"] for item in feedback["findings"]}
    assert "cue_only_shortcut" in codes
    assert "target_surface_leakage" not in codes
    assert "unknown_blocking_reason" in codes
    cue = next(
        item for item in feedback["findings"] if item["code"] == "cue_only_shortcut"
    )
    assert cue["evidence"]["reason"] == "cue_only_lift"


def test_synthetic_anchor_deficit_uses_admission_severity() -> None:
    feedback = build_synthesis_feedback(
        version="v",
        profile="strict",
        built_at="t",
        admitted=[_admitted("a", "synthetic")],
        rejections=[],
        quality_report={"source_anchors": {"deficit": True, "trusted_roots": 0}},
    )
    item = next(
        finding
        for finding in feedback["findings"]
        if finding["code"] == "synthetic_anchor_deficit"
    )
    assert item["severity"] == "hard_admission_failure"
    assert item["authority"] == "blocks_capability_claim"
    assert item["waiver"] == "diagnostic_waiver_not_promotion"


def test_blocking_findings_need_a_build_receipt() -> None:
    from slm_training.harnesses.train_data.feedback import action_receipt

    feedback = build_synthesis_feedback(
        version="v",
        profile="strict",
        built_at="t",
        admitted=[_admitted("a", "synthetic")],
        rejections=[],
        quality_report={
            "unique_roots": {"requested": 8, "admitted": 1},
            "pair_quality": {
                "passed": False,
                "blocking_reasons": ["cue_only_shortcut"],
            },
        },
    )
    codes = {item["code"] for item in feedback["findings"]}
    assert "insufficient_unique_roots" in codes
    assert "cue_only_shortcut" in codes
    assert any(item.get("data_only") for item in feedback["experiment_candidates"])
    try:
        action_receipt(
            prior_feedback_hash="a" * 64,
            action_code="insufficient_unique_roots",
            synthesis_plan_hash="b" * 64,
            dataset_manifest_hash="c" * 64,
            before={"admitted": 1},
            after={"admitted": 1},
            disposition="acknowledged",
        )
        raise AssertionError("prose ack must not close a blocking finding")
    except ValueError as exc:
        assert "cannot close" in str(exc)


def test_selection_drops_are_not_reported_as_synthesis_failures() -> None:
    feedback = build_synthesis_feedback(
        version="v",
        profile="strict",
        built_at="t",
        admitted=[_admitted("kept", "prompt_paraphrase", "template")],
        rejections=[
            _reject(
                "selection", "difficulty_easy_tail", "prompt_paraphrase", "template"
            )
            for _ in range(12)
        ],
        quality_report={},
    )

    assert feedback["families"]["prompt_paraphrase"]["candidates"] == 1
    assert feedback["synthesizers"]["template"]["candidates"] == 1
    assert feedback["recommendations"] == []
    assert feedback["experiment_candidates"] == []
