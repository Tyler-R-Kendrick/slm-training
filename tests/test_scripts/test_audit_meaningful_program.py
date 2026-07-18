from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.audit_meaningful_program as audit_cli
from slm_training.data.contract import GenerationRequest
from slm_training.dsl.schema import load_jsonl
from scripts.audit_meaningful_program import (
    _confusion,
    _frontier_set,
    _labeled_metrics,
    _matrix,
)


def _write_records(test_dir: Path) -> None:
    path = test_dir / "suites" / "smoke" / "records.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "id": "good",
            "prompt": "Build a Button. Placeholders: :cta.label",
            "openui": 'root = Button(":cta.label")',
            "split": "smoke",
            "source": "audit-test",
        },
        {
            "id": "bad",
            "prompt": "Build a Button. Placeholders: :bad",
            "openui": 'root = Button(":bad")',
            "split": "smoke",
            "source": "audit-test",
        },
        {
            "id": "cut",
            "prompt": "Build a Button. Placeholders: :cut",
            "openui": 'root = Button(":cut")',
            "split": "smoke",
            "source": "audit-test",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _checkpoint_payload(tmp_path: Path, name: str = "checkpoint.pt") -> dict[str, str]:
    path = tmp_path / name
    path.write_bytes(name.encode())
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _envelope(test_dir: Path, case_id: str) -> dict[str, object]:
    records = load_jsonl(test_dir / "suites" / "smoke" / "records.jsonl")
    record = next(row for row in records if row.id == case_id)
    return {
        "generation_request": GenerationRequest.from_record(
            record, include_design_md=False
        ).to_dict(),
        "source_record_sha256": hashlib.sha256(
            json.dumps(
                record.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }


def test_frontier_replay_aggregates_versions_reasons_and_explicit_labels(
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    _write_records(test_dir)
    good = 'root = Button(":cta.label")'
    eval_path = tmp_path / "eval_smoke.json"
    eval_path.write_text(
        json.dumps(
            {
                "suite": "smoke",
                **_checkpoint_payload(tmp_path),
                "n": 2,
                "details": [
                    {
                        "id": "good",
                        **_envelope(test_dir, "good"),
                        "prediction": good,
                        "prediction_sha256": hashlib.sha256(good.encode()).hexdigest(),
                        "semantic_labels": {
                            "agentv": {"verdict": True, "provenance": "agentv:test"}
                        },
                    },
                    {
                        "id": "bad",
                        **_envelope(test_dir, "bad"),
                        "prediction": "root = Stack([])",
                        "prediction_sha256": hashlib.sha256(b"root = Stack([])").hexdigest(),
                        "semantic_labels": {
                            "human": {"verdict": False, "provenance": "human:test"}
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _frontier_set("frontier", eval_path, test_dir)

    assert result["replayable"] is True
    assert result["v2"]["n"] == 2
    assert result["v2"]["strict_rate"] == 0.5
    assert result["v2"]["coverage_conditioned_rate"] == 0.5
    assert result["v1_v2"]["counts"] == {
        "v1_false_v2_false": 1,
        "v1_true_v2_true": 1,
    }
    assert result["reason_prevalence"]["no_nontrivial_content"] == 1
    assert result["external_labels"]["agentv"]["v2_vs_label"]["tp"] == 1
    assert result["external_labels"]["human"]["v2_vs_label"]["tn"] == 1
    assert result["external_labels"]["independent_judge"]["status"] == "UNKNOWN"
    assert result["checkpoint_verifications"][0]["status"] == "PASS"
    assert len(result["cases"]) == 2


def test_frontier_replay_marks_legacy_500_character_prediction_unknown(
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    _write_records(test_dir)
    eval_path = tmp_path / "eval_smoke.json"
    eval_path.write_text(
        json.dumps(
            {
                "suite": "smoke",
                **_checkpoint_payload(tmp_path),
                "n": 1,
                "details": [{"id": "cut", "prediction": "x" * 500}],
            }
        ),
        encoding="utf-8",
    )

    result = _frontier_set("frontier", eval_path, test_dir)

    assert result["replayable"] is False
    assert result["v2"]["n"] == 1
    assert result["v2"]["replayable_n"] == 0
    assert result["v2"]["strict_rate"] == 0.0
    assert result["v2"]["coverage"] == 0.0
    assert result["v2"]["reason_prevalence"] == {
        "stored_prediction_may_be_truncated": 1
    }
    case = result["cases"][0]
    assert case["prediction_status"] == "UNKNOWN"
    assert case["meaningful_program_v1"]["verdict"] is None
    assert case["binding_aware_meaningful_v2"]["verdict"] is False
    assert result["v1_v2"]["n"] == 0


def test_exactly_500_characters_is_replayable_with_matching_digest(
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    _write_records(test_dir)
    prediction = 'root = Button(":cut")' + " " * (500 - len('root = Button(":cut")'))
    eval_path = tmp_path / "eval_smoke.json"
    eval_path.write_text(
        json.dumps(
            {
                "suite": "smoke",
                **_checkpoint_payload(tmp_path),
                "n": 1,
                "details": [
                    {
                        "id": "cut",
                        **_envelope(test_dir, "cut"),
                        "prediction": prediction,
                        "prediction_sha256": hashlib.sha256(
                            prediction.encode()
                        ).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _frontier_set("frontier", eval_path, test_dir)

    assert result["replayable"] is True
    assert result["cases"][0]["prediction_status"] == "COMPLETE"
    assert result["v2"]["replayable_n"] == 1


def test_main_accepts_repeated_explicit_generation_sets(
    tmp_path: Path, monkeypatch
) -> None:
    test_dir = tmp_path / "test"
    _write_records(test_dir)
    prediction = 'root = Button(":cta.label")'
    eval_paths = [tmp_path / "eval_a.json", tmp_path / "eval_b.json"]
    for index, path in enumerate(eval_paths):
        checkpoint = _checkpoint_payload(tmp_path, f"checkpoint-{index}.pt")
        path.write_text(
            json.dumps(
                {
                    "suite": "smoke",
                    **checkpoint,
                    "n": 1,
                    "details": [{
                        "id": "good",
                        **_envelope(test_dir, "good"),
                        "prediction": prediction,
                        "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
                    }],
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        audit_cli,
        "publish_agentv_evaluation",
        lambda *_args, **_kwargs: {"status": "synthetic-test"},
    )
    output = tmp_path / "audit.json"
    bundle = tmp_path / "replay-bundle.json"

    code = audit_cli.main(
        [
            "--generation-set",
            f"a={eval_paths[0]}",
            "--generation-set",
            f"b={eval_paths[1]}",
            "--test-dir",
            str(test_dir),
            "--minimum-frontier-sets",
            "2",
            "--unavailable-set",
            "named=no stored generation envelope",
            "--output",
            str(output),
            "--capture-replay-bundle",
            str(bundle),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["replayable_frontier_sets"] == 2
    assert [row["label"] for row in payload["sets"]] == ["a", "b"]
    assert payload["unavailable_named_sets"] == [
        {
            "label": "named",
            "status": "UNKNOWN",
            "reason": "no stored generation envelope",
        }
    ]
    replay_output = tmp_path / "replayed-audit.json"
    replay_code = audit_cli.main(
        [
            "--replay-bundle",
            str(bundle),
            "--minimum-frontier-sets",
            "2",
            "--output",
            str(replay_output),
            "--run-dir",
            str(tmp_path / "replay-run"),
        ]
    )
    replayed = json.loads(replay_output.read_text(encoding="utf-8"))
    assert replay_code == 0
    assert replayed["replayable_frontier_sets"] == 2
    assert replayed["durable_replay_bundle"]["sha256"] == hashlib.sha256(
        bundle.read_bytes()
    ).hexdigest()


def test_frontier_replay_rejects_checkpoint_digest_mismatch(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    _write_records(test_dir)
    checkpoint = _checkpoint_payload(tmp_path)
    eval_path = tmp_path / "eval_smoke.json"
    eval_path.write_text(
        json.dumps(
            {
                "suite": "smoke",
                **checkpoint,
                "checkpoint_sha256": "0" * 64,
                "n": 1,
                "details": [{
                    "id": "good",
                    **_envelope(test_dir, "good"),
                    "prediction": 'root = Button(":cta.label")',
                    "prediction_sha256": hashlib.sha256(
                        b'root = Button(":cta.label")'
                    ).hexdigest(),
                }],
            }
        ),
        encoding="utf-8",
    )

    result = _frontier_set("frontier", eval_path, test_dir)

    assert result["replayable"] is False
    assert result["checkpoint_verifications"][0]["status"] == "FAIL"
    assert result["failures"] == ["eval_smoke.json:checkpoint_digest_mismatch"]


def test_main_blocks_when_gaming_corpus_fails(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "gaming.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "wrong-expectation",
                "prompt": "Build a Button. Placeholders: :cta.label",
                "prediction": 'root = Button(":cta.label")',
                "expected_verdict": False,
                "expected_reason_codes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        audit_cli,
        "publish_agentv_evaluation",
        lambda *_args, **_kwargs: {"status": "synthetic-test"},
    )
    output = tmp_path / "audit.json"

    code = audit_cli.main(
        [
            "--gaming-corpus",
            str(corpus),
            "--minimum-frontier-sets",
            "0",
            "--output",
            str(output),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["non_replayable_sets=deterministic_gaming_corpus"]


def test_kappa_is_unknown_for_degenerate_constant_raters() -> None:
    pairs = [(False, False), (False, False)]

    assert _matrix(pairs)["kappa"] is None
    assert _confusion(pairs)["cohen_kappa"] is None
    assert _labeled_metrics(pairs)["kappa"] is None


def test_frontier_replay_rejects_incomplete_or_duplicate_envelope(
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    _write_records(test_dir)
    eval_path = tmp_path / "eval_smoke.json"
    eval_path.write_text(
        json.dumps(
            {
                "suite": "smoke",
                **_checkpoint_payload(tmp_path),
                "n": 3,
                "details": [
                    {"id": "good", "prediction": 'root = Button(":cta.label")'},
                    {"id": "good", "prediction": 'root = Button(":cta.label")'},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _frontier_set("frontier", eval_path, test_dir)

    assert result["replayable"] is False
    assert "eval_smoke.json:declared_n_mismatch:3!=2" in result["failures"]
    assert "eval_smoke.json:duplicate_case_id" in result["failures"]
