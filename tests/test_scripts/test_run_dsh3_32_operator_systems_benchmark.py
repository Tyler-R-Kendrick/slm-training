"""CLI proof for the SLM-407 local no-claim preflight."""

from __future__ import annotations

import json
from pathlib import Path


def test_preflight_writes_versioned_no_claim_evidence(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_dsh3_32_operator_systems_benchmark as script

    monkeypatch.setattr(
        script,
        "publish_agentv_evaluation",
        lambda *_args, **_kwargs: {"criteria": {"pass": True}},
    )
    typed_report = tmp_path / "typed.json"
    typed_report.write_text(
        json.dumps({"decision": {"verdict": "reject", "reason": "negative"}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "evidence"

    assert script.main(["--typed-policy-report", str(typed_report), "--output-dir", str(output_dir)]) == 0
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["benchmark"]["verdict"] == "unavailable"
    assert payload["benchmark"]["missing_arms"]
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"


def test_portable_rewrites_ephemeral_paths(tmp_path: Path) -> None:
    from scripts.run_dsh3_32_operator_systems_benchmark import _portable

    output_dir = tmp_path / "evidence"
    value = str(output_dir / "agentv" / "report.json")

    assert _portable(value, output_dir=output_dir) == "agentv-dir://agentv/report.json"
