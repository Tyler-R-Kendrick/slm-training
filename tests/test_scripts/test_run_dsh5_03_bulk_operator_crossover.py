"""CLI proof for the SLM-411 local no-claim crossover preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.evals.bulk_operator_matrix import build_bulk_operator_crossover_report


def _cheap_ok_probes() -> tuple[dict, ...]:
    """Synthetic probes that satisfy the report contract without the 250s live stack."""

    return tuple(
        {
            "fanout": fanout,
            "coverage": "complete",
            "legal_actions": 1,
            "bulk_selector_candidate_rows": [],
            "replay_exact": True,
            "primitive_equivalent": True,
            "final_state_equal": True,
            "unintended_edits": 0,
            "action_accuracy": 1.0,
            "ambiguous_rounds": 0,
            "model_forwards": 0,
            "claim_class": "compiler_fixture_wiring_only",
        }
        for fanout in (1, 2, 4, 8)
    )


def test_preflight_writes_versioned_unavailable_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    """Default CI path: report builder + CLI + version stamp (no live probes)."""

    from scripts import run_dsh5_03_bulk_operator_crossover as script

    monkeypatch.setattr(
        script,
        "publish_agentv_evaluation",
        lambda *_args, **_kwargs: {"criteria": {"pass": True}},
    )
    monkeypatch.setattr(
        script,
        "run_local_preflight",
        lambda decision: build_bulk_operator_crossover_report(
            typed_policy_decision=decision,
            probes=_cheap_ok_probes(),
        ),
    )
    typed = tmp_path / "typed.json"
    typed.write_text(json.dumps({"decision": {"verdict": "reject"}}), encoding="utf-8")
    output_dir = tmp_path / "evidence"

    assert script.main(["--typed-policy-report", str(typed), "--output-dir", str(output_dir)]) == 0
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["crossover"]["verdict"] == "unavailable"
    assert payload["crossover"]["fixture_contract_satisfied"] is True
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"


# Live ``run_local_preflight`` measures ~250s on an idle machine in a SINGLE
# test node, so no shard packing can fit it under MAX_RUN_MINUTES. Keep the
# real preflight as an explicit slow integration check.
@pytest.mark.slow
def test_live_preflight_writes_versioned_unavailable_evidence(
    tmp_path: Path, monkeypatch
) -> None:
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
