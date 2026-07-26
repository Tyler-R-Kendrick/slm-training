from __future__ import annotations

import json

from scripts.run_dsh3_31_operator_failure_prediction import main


def test_preflight_rejects_eval_without_temporal_rows(tmp_path) -> None:
    source = tmp_path / "eval.json"
    source.write_text(
        json.dumps(
            {
                "checkpoint_sha256": "fixture-checkpoint",
                "details": [
                    {"id": "r1", "decode_outcome": "model_invalid"}
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report"

    assert main(["--eval-json", str(source), "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["decision"]["verdict"] == "reject"
    assert report["feature_rows"] == []


def test_preflight_records_unavailable_local_source_without_fake_rows(tmp_path) -> None:
    output = tmp_path / "report"

    assert main(
        [
            "--output-dir",
            str(output),
            "--preflight-reason",
            "checkpoint contract was incompatible before generation",
        ]
    ) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["decision"]["verdict"] == "unavailable"
    assert report["feature_rows"] == []
    assert report["preflight_reasons"] == [
        "checkpoint contract was incompatible before generation"
    ]
