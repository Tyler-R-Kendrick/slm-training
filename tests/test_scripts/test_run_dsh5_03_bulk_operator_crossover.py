"""CLI proof for the SLM-411 local no-claim crossover preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ``run_local_preflight`` measures ~250s on an idle machine in a SINGLE test
# node, so no shard packing can fit it under the canonical per-job wall
# (MAX_RUN_MINUTES = 3). Before this marker it was cancelled on every CI run,
# which produced no evidence at all while reddening a shard; deselecting it by
# default turns a silent timeout into a documented deferral. Run explicitly:
#   python -m pytest tests/test_scripts/test_run_dsh5_03_bulk_operator_crossover.py -m slow
@pytest.mark.slow
def test_preflight_writes_versioned_unavailable_evidence(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_dsh5_03_bulk_operator_crossover as script

    monkeypatch.setattr(
        script,
        "publish_agentv_evaluation",
        lambda *_args, **_kwargs: {"criteria": {"pass": True}},
    )
    typed = tmp_path / "typed.json"
    typed.write_text(json.dumps({"decision": {"verdict": "reject"}}), encoding="utf-8")
    output_dir = tmp_path / "evidence"

    assert script.main(["--typed-policy-report", str(typed), "--output-dir", str(output_dir)]) == 0
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["crossover"]["verdict"] == "unavailable"
    assert payload["crossover"]["fixture_contract_satisfied"] is True
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
