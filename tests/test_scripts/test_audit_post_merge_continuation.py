from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_post_merge_continuation as runner


def test_audit_writes_versioned_portable_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "publish_agentv_evaluation",
        lambda run_dir, **_: {"spec": str(Path(run_dir) / "agentv/spec.eval.jsonl")},
    )
    assert runner.main(
        [
            "--output-dir",
            str(tmp_path),
            "--max-combinations",
            "32",
            "--candidate-limit",
            "6",
        ]
    ) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "PostMergeContinuationReportV1"
    assert report["audit"]["trace_turns"] == 1
    assert report["audit"]["old_table_refusal_codes"] == [
        "ref.stale_state",
        "ref.cross_branch",
    ]
    assert report["agentv"]["spec"] == "output-dir://agentv/spec.eval.jsonl"
    assert (tmp_path / "summary.md").is_file()
