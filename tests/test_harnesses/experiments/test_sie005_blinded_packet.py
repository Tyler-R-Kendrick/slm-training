"""SIE-005 (SLM-478): blinded construct-validity packet harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.sie005_blinded_packet import (
    AUDIT_ID,
    RELATED_CATALOGUE_ID,
    SIE005_STRATA,
    assert_public_packet_blinded,
    build_fixture_audit_rows,
    mock_external_transport,
    prepare_blinded_packet,
    run_fixture_packet,
)


def test_fixture_rows_cover_required_strata() -> None:
    rows = build_fixture_audit_rows(seed=0)
    strata = {row["stratum"] for row in rows}
    assert strata == set(SIE005_STRATA)
    assert all(str(row["pair_id"]).startswith("rec_") for row in rows)
    assert len(rows) == len(SIE005_STRATA) * 2


def test_prepare_packet_blinds_and_excludes_training(tmp_path: Path) -> None:
    report = prepare_blinded_packet(output_dir=tmp_path, seed=0)
    assert report.status == "fixture"
    assert report.blinding_ok is True
    assert report.training_excluded is True
    assert report.independence_ok is True
    assert report.related_catalogue_id == RELATED_CATALOGUE_ID
    assert set(report.strata) == set(SIE005_STRATA)
    assert report.pair_n == len(SIE005_STRATA) * 2
    assert report.import_complete_pair_n == report.pair_n
    assert report.mock_judge_modes_ok == ["ok", "refusal", "cost_overrun"]

    public = Path(report.public_dir)
    assert_public_packet_blinded(public)
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_use"] is False
    assert manifest["audit_holdout"] is True
    assert manifest["corpus_admission"] is False
    assert manifest["automatic_judgments_distributed"] is False
    assert manifest["identity_fields_distributed"] is False

    order = json.loads((public / "annotation_order.json").read_text(encoding="utf-8"))
    assert order["schema"] == "Sie005AnnotationOrderV1"
    assert len(order["opaque_record_ids"]) == report.pair_n
    assert set(order["opaque_record_ids"]) == {
        json.loads(line)["pair_id"]
        for line in (public / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    label_schema = json.loads((public / "label_schema.json").read_text(encoding="utf-8"))
    assert label_schema["policy"]["no_automatic_scores"] is True
    assert label_schema["policy"]["min_raters"] == 2

    assert (public / "CREDENTIAL_RUNBOOK.md").is_file()
    assert "EXTERNAL_JUDGE_API_KEY" in (public / "CREDENTIAL_RUNBOOK.md").read_text()
    assert (public / "external_judge" / "request_schema.json").is_file()
    assert (public / "external_judge" / "response_schema.json").is_file()
    assert (tmp_path / "harness_fixtures" / "mock_adapter_ok.json").is_file()
    assert Path(report.private_key_path).is_file()
    # Private key must live outside the redacted package.
    assert not Path(report.private_key_path).resolve().is_relative_to(public.resolve())


def test_packet_is_deterministic(tmp_path: Path) -> None:
    a = prepare_blinded_packet(output_dir=tmp_path / "a", seed=3)
    b = prepare_blinded_packet(output_dir=tmp_path / "b", seed=3)
    assert a.annotation_order_sha256 == b.annotation_order_sha256
    assert a.rubric_sha256 == b.rubric_sha256
    assert a.pair_n == b.pair_n


def test_mock_transport_rejects_identity_leak() -> None:
    transport = mock_external_transport(mode="ok")
    with pytest.raises(ValueError, match="leaked"):
        transport(
            {
                "pair": {"prompt": "x", "left_openui": "a", "right_openui": "b"},
                "deterministic_verdict": "left",
            }
        )


def test_run_fixture_packet_smoke(tmp_path: Path) -> None:
    report = run_fixture_packet(output_dir=tmp_path, seed=1)
    assert report.audit_id == AUDIT_ID
    assert report.version_stamp
