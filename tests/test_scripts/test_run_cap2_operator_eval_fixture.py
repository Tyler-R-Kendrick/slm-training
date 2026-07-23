from __future__ import annotations

import json
from pathlib import Path

from scripts import run_cap2_operator_eval_fixture as runner


def test_runner_writes_portable_agentv_report(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def publish(run_dir, *, name, claim, cases):
        captured.update(
            {
                "run_dir": Path(run_dir),
                "name": name,
                "claim": claim,
                "cases": cases,
            }
        )
        return {
            "format": "AgentEvals JSONL",
            "sdk": "@agentv/core",
            "spec": str(Path(run_dir).resolve() / "agentv/spec.eval.jsonl"),
            "summary": {
                "total": len(cases),
                "passed": len(cases),
                "failed": 0,
                "meanScore": 1.0,
                "executionErrors": 0,
            },
        }

    monkeypatch.setattr(runner, "publish_agentv_evaluation", publish)
    monkeypatch.setattr(runner.tracemalloc, "start", lambda: None)
    monkeypatch.setattr(
        runner.tracemalloc, "get_traced_memory", lambda: (0, 1)
    )
    monkeypatch.setattr(runner.tracemalloc, "stop", lambda: None)
    assert runner.main(["--output-dir", str(tmp_path)]) == 0
    report = json.loads(
        (tmp_path / "report.json").read_text(encoding="utf-8")
    )
    assert report["suite"]["suite_hash"] == (
        "16f210786bac7fd5f5edb64d13888c3cc7d634330a81b5065150e7a41fcb1d4d"
    )
    assert report["policy_scores"]["oracle"]["gate_pass"] is True
    assert report["policy_scores"]["constant_operator"]["gate_pass"] is False
    assert report["agentv"]["spec"] == "output-dir://agentv/spec.eval.jsonl"
    assert captured["name"] == "cap2-operator-fixture-v1"
    assert len(captured["cases"]) == 6
