from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.annotations.judge_audit import (
    REQUIRED_STRATA,
    freeze_blinded_pairs,
    freeze_matched_training_rows,
    import_blinded_labels,
    validate_campaign_sample,
)


def source_pair(pair_id: str = "pair_01") -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "source_record_id": f"source_{pair_id}",
        "stratum": "structurally_similar",
        "prompt": "Build a profile card",
        "training_use": False,
        "audit_holdout": True,
        "candidate_a": {
            "candidate_id": "a",
            "model_family": "twotower",
            "run_id": "run_a",
            "openui": 'Card(Text("A"))',
            "checkpoint_sha256": "a" * 64,
        },
        "candidate_b": {
            "candidate_id": "b",
            "model_family": "choice",
            "run_id": "run_b",
            "openui": 'Card(Text("B"))',
            "checkpoint_sha256": "b" * 64,
        },
    }


def label(
    annotator_id: str,
    winner: str,
    *,
    role: str = "rater",
) -> dict[str, object]:
    return {
        "schema": "BlindJudgeLabelV1",
        "audit_id": "efs0-04",
        "pair_id": "pair_01",
        "annotator_id": annotator_id,
        "role": role,
        "winner": winner,
        "acceptable_left": True,
        "acceptable_right": False,
        "reasons": ["prompt_role_match"],
        "confidence": 0.8,
        "duration_ms": 1200,
        "submitted_at": "2026-07-17T12:00:00Z",
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_freeze_blinded_pairs_is_reproducible_and_hides_identity(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = freeze_blinded_pairs(
        [source_pair()],
        audit_id="efs0-04",
        seed=19,
        output_dir=first,
        private_key_path=tmp_path / "first-private.jsonl",
        enforce_campaign_coverage=False,
    )
    freeze_blinded_pairs(
        [source_pair()],
        audit_id="efs0-04",
        seed=19,
        output_dir=second,
        private_key_path=tmp_path / "second-private.jsonl",
        enforce_campaign_coverage=False,
    )

    assert manifest["training_use"] is False
    assert (first / "blind_pairs.jsonl").read_bytes() == (
        second / "blind_pairs.jsonl"
    ).read_bytes()
    public = (first / "blind_pairs.jsonl").read_text(encoding="utf-8")
    html = (first / "index.html").read_text(encoding="utf-8")
    assert "twotower" not in public
    assert "run_a" not in public
    assert "candidate_id" not in public
    assert not (first / "private_unblinding_key.jsonl").exists()
    assert "twotower" not in html
    assert "run_a" not in html


def test_import_requires_two_raters_and_adjudicates_disagreement(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            label("ann_12345678", "left"),
            label("ann_abcdefgh", "right"),
            label("ann_adjudicate", "left", role="adjudicator"),
        ],
    )
    payload = import_blinded_labels(
        [labels],
        audit_id="efs0-04",
        pair_ids={"pair_01"},
        output_path=tmp_path / "aggregate.json",
    )

    assert payload["complete_pair_n"] == 1
    assert payload["pairs"][0]["needs_adjudication"] is True
    assert payload["pairs"][0]["consensus_winner"] == "left"


def test_import_is_idempotent_but_rejects_conflicts_and_identity_fields(
    tmp_path: Path,
) -> None:
    first = label("ann_12345678", "left")
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [first, first])
    payload = import_blinded_labels(
        [labels],
        audit_id="efs0-04",
        pair_ids={"pair_01"},
        output_path=tmp_path / "aggregate.json",
    )
    assert payload["label_n"] == 1

    first["display_name"] = "Alice"
    write_jsonl(labels, [first])
    with pytest.raises(ValueError, match="forbidden identity"):
        import_blinded_labels(
            [labels],
            audit_id="efs0-04",
            pair_ids={"pair_01"},
            output_path=tmp_path / "aggregate.json",
        )


def test_campaign_sample_gate_requires_powered_holdout_and_five_families() -> None:
    families = ["x22", "twotower", "choice", "grammar-diffusion", "baseline"]
    strata = sorted(REQUIRED_STRATA)
    rows = []
    for index in range(100):
        row = source_pair(f"pair_{index:03d}")
        row["stratum"] = strata[index % len(strata)]
        row["candidate_a"]["model_family"] = families[index % len(families)]  # type: ignore[index]
        rows.append(row)
    validate_campaign_sample(rows, training_record_ids=set())

    rows[0]["audit_holdout"] = False
    with pytest.raises(ValueError, match="non-training holdout"):
        validate_campaign_sample(rows, training_record_ids=set())
    rows[0]["audit_holdout"] = True
    with pytest.raises(ValueError, match="overlap"):
        validate_campaign_sample(
            rows, training_record_ids={str(rows[0]["source_record_id"])}
        )


def test_matched_training_rows_fail_closed_on_leakage_and_match_exposure(
    tmp_path: Path,
) -> None:
    rows = [
        {"record_id": f"r{index}", "prompt": f"p{index}", "completion": "Card()"}
        for index in range(6)
    ]
    manifest = freeze_matched_training_rows(
        rows,
        intersection_ids={"r0", "r1"},
        audit_holdout_ids={"audit-only"},
        seeds=(1, 2, 3),
        target_token_exposure=10_000,
        external_evidence_sha256="c" * 64,
        human_aggregate_sha256="d" * 64,
        output_dir=tmp_path / "matched",
    )
    assert len(manifest["arms"]) == 9
    assert {arm["target_token_exposure"] for arm in manifest["arms"]} == {10_000}
    assert {
        arm["row_n"]
        for arm in manifest["arms"]
        if arm["arm"] != "original"
    } == {2}

    rows[0]["judge"] = {"verdict": "accept"}
    with pytest.raises(ValueError, match="cannot enter training"):
        freeze_matched_training_rows(
            rows,
            intersection_ids={"r0"},
            audit_holdout_ids=set(),
            seeds=(1, 2, 3),
            target_token_exposure=1,
            external_evidence_sha256="c" * 64,
            human_aggregate_sha256="d" * 64,
            output_dir=tmp_path / "rejected",
        )
