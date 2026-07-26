from __future__ import annotations

import json

import scripts.run_dsh3_31_operator_failure_prediction as runner


def _stub_agentv(output_dir, **_kwargs):
    return {"spec": str(output_dir / "agentv" / "fixture.eval.jsonl")}


def test_preflight_rejects_eval_without_temporal_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "publish_agentv_evaluation", _stub_agentv)
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

    assert runner.main(["--eval-json", str(source), "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["decision"]["verdict"] == "reject"
    assert report["feature_rows"] == []


def test_preflight_records_unavailable_local_source_without_fake_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "publish_agentv_evaluation", _stub_agentv)
    output = tmp_path / "report"

    assert runner.main(
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
