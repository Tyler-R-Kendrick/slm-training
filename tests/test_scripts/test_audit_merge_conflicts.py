from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_merge_conflicts as runner


def test_audit_writes_versioned_portable_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "publish_agentv_evaluation",
        lambda run_dir, **_: {"spec": str(Path(run_dir) / "agentv/spec.eval.jsonl")},
    )
    assert runner.main(["--output-dir", str(tmp_path), "--max-combinations", "64"]) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "MergeConflictAuditV1"
    assert report["audit"]["pair_denominator"] >= 0
    assert report["agentv"]["spec"] == "output-dir://agentv/spec.eval.jsonl"
    assert (tmp_path / "summary.md").is_file()
