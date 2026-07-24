"""SLM-288: typed v1 reason codes + validation harness invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_slm288_meaningful_validation import (
    analyze_labels,
    build_histograms,
    export_blind,
    freeze_packet,
    load_labels,
    score_bundle,
    _load_bundle,
)
from slm_training.harnesses.model_build.eval_runner import (
    _is_meaningful_program,
    meaningful_program_v1_report,
)


def _gold():
    from slm_training.dsl.schema import ExampleRecord

    return ExampleRecord(
        id="g1",
        prompt="hero with title and button",
        openui='Stack([Card([Text(":title"), Button(":cta")])])',
        placeholders=[":title", ":cta"],
        split="smoke",
    )


def test_report_matches_v1_verdict_and_reason_on_corpus() -> None:
    bundle = _load_bundle()
    for row in score_bundle(bundle):
        pred = row["prediction"]
        v1, reason, serialized = _is_meaningful_program(pred)
        report = meaningful_program_v1_report(pred)
        assert report["verdict"] == v1
        assert report["legacy_reason"] == reason
        assert report["serialized"] == serialized


def test_clause_codes_typed_and_ordered() -> None:
    report = meaningful_program_v1_report("Stack([])")
    codes = [c["code"] for c in report["checks"]]
    assert codes[0] in {
        "parse_failed",
        "free_form_output_string",
        "empty_root_stack",
    }
    assert report["reason_codes"] == [c["code"] for c in report["checks"] if c["status"] != "pass"]


def test_unknown_recall_unobservable_is_not_a_failure() -> None:
    bundle = _load_bundle()
    raw = bundle["records"]["smoke:smoke_hero_01"]
    from slm_training.dsl.schema import ExampleRecord

    gold = ExampleRecord(
        id=raw["id"],
        prompt=raw["prompt"],
        openui=raw["openui"],
        placeholders=raw.get("placeholders"),
        split=raw.get("split", "smoke"),
        source=raw.get("source", "fixture"),
    )
    pred = gold.openui
    report = meaningful_program_v1_report(pred, gold=gold)
    assert report["verdict"] is True
    assert all(c["status"] != "fail" for c in report["checks"])
    stack_only = ExampleRecord(
        id="g2",
        prompt="plain stack",
        openui="root = Stack([])",
        placeholders=[],
        split="smoke",
    )
    report2 = meaningful_program_v1_report(pred, gold=stack_only)
    assert report2["verdict"] is True
    assert any(
        c["code"] == "component_recall_unobservable" and c["status"] == "unknown"
        for c in report2["checks"]
    )
    report3 = meaningful_program_v1_report(pred)  # no gold at all
    assert report3["verdict"] is True
    assert any(c["status"] == "unknown" for c in report3["checks"])


def test_empty_children_code_carries_type() -> None:
    report = meaningful_program_v1_report('Stack([Modal([], ":x")])')
    codes = {c["code"]: c for c in report["checks"]}
    if "empty_children" in codes:
        assert codes["empty_children"]["detail"] == "Modal"
        assert report["legacy_reason"] == "empty_children:Modal"


def test_histograms_shape() -> None:
    bundle = _load_bundle()
    rows = score_bundle(bundle)
    hist = build_histograms(rows)
    assert hist["overall"]["n"] == 33
    assert set(hist["per_suite"]) == {"smoke"}
    assert len(hist["per_set"]) == 11
    total_fail = sum(hist["overall"]["fail_reason_histogram"].values())
    assert total_fail >= hist["overall"]["verdicts"].get("false", 0)


def test_packet_frozen_deterministic_and_blinded(tmp_path: Path) -> None:
    bundle = _load_bundle()
    rows = score_bundle(bundle)
    first = freeze_packet(rows, bundle)
    second = freeze_packet(rows, bundle)
    assert first["packet_sha256"] == second["packet_sha256"]
    assert first["n"] == 33
    assert first["shortfall"]  # honest shortfall vs the ~100 target
    blind = export_blind(first)
    for line in blind.splitlines():
        row = json.loads(line)
        assert set(row) == {"packet_id", "suite", "prompt", "prediction"}
    assert "e228" not in blind and "checkpoint" not in blind


def test_analyze_kappa_pr_and_sweep(tmp_path: Path) -> None:
    bundle = _load_bundle()
    rows = score_bundle(bundle)
    packet = freeze_packet(rows, bundle)
    # Synthetic labels: rater_a agrees with the metric verdict, rater_b
    # disagrees on exactly one item.
    label_files = []
    for rater, flip in (("rater_a", None), ("rater_b", "pkt_001")):
        path = tmp_path / f"{rater}.jsonl"
        with path.open("w") as handle:
            for item in packet["items"]:
                verdict = bool(item["provenance"]["v1_report"]["verdict"])
                if item["packet_id"] == flip:
                    verdict = not verdict
                handle.write(
                    json.dumps(
                        {
                            "schema": "slm288_meaningful_label/v1",
                            "packet_id": item["packet_id"],
                            "rater_id": rater,
                            "label": "meaningful" if verdict else "not_meaningful",
                            "reason_codes": [],
                            "confidence": 0.9,
                        }
                    )
                    + "\n"
                )
        label_files.append(path)
    labels = load_labels(label_files)
    analysis = analyze_labels(packet, labels)
    assert analysis["coverage"]["fully_labeled"] == 33
    assert analysis["coverage"]["needs_adjudication"] == 1
    assert analysis["human_agreement"]["cohen_kappa"] is not None
    assert analysis["metric_vs_labels"]["n"] == 32
    assert analysis["metric_vs_labels"]["precision"] == 1.0
    assert set(analysis["recall_floor_sweep"]["results"]) == {"0.3", "0.5", "0.7"}
    # Determinism
    assert analyze_labels(packet, labels) == analysis


def test_load_labels_rejects_bad_values(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "schema": "slm288_meaningful_label/v1",
                "packet_id": "pkt_001",
                "rater_id": "rater_a",
                "label": "maybe",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="bad label value"):
        load_labels([bad])
