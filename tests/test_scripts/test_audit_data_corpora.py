from __future__ import annotations

import json

from scripts.audit_data_corpora import main


def test_report_certification_writes_stamped_durable_result(tmp_path, monkeypatch):
    snapshot = tmp_path / "openui_verified_v1"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "openui_verified_v1",
                "source_fingerprints": {"fixture": "abc"},
                "source_rows": 2,
                "core_records": 1,
                "tiers": {"Gold": 0, "Silver": 1, "Bronze": 0, "Quarantine": 1},
                "version_stamp": {"stamp_schema": "version_stamp/v1"},
            }
        )
    )
    (snapshot / "quality_report.json").write_text(json.dumps({"quarantine_rate": 0.5}))
    monkeypatch.chdir(tmp_path)

    assert main(["--mode", "report-certification", "--output-dir", str(snapshot)]) == 0
    report = json.loads(
        (tmp_path / "docs/design/iter-slm289-corpus-certification-20260725.json").read_text()
    )
    assert report["source_rows"] == 2
    assert report["version_stamp"]["stamp_schema"] == "version_stamp/v1"
