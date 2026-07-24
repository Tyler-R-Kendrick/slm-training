from __future__ import annotations

import json
from pathlib import Path

import slm_training.evals.agentv as agentv_module

from slm_training.evals.agentv import (
    _agentv_runtime,
    model_ship_gate_cases,
    publish_agentv_evaluation,
    publish_model_evaluation,
)


def test_agentv_runtime_uses_git_common_checkout_for_worktree_sdk(
    tmp_path, monkeypatch
) -> None:
    common_root = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    runner = common_root / "scripts/run_agentv_eval.mjs"
    sdk = common_root / "node_modules/@agentv/core/package.json"
    runner.parent.mkdir(parents=True)
    sdk.parent.mkdir(parents=True)
    runner.write_text("// runner")
    sdk.write_text("{}")
    worktree.mkdir()
    monkeypatch.delenv("AGENTV_RUNNER", raising=False)
    monkeypatch.setattr(
        agentv_module,
        "checkout_roots",
        lambda root: (root, common_root),
    )

    assert _agentv_runtime(worktree) == (runner, common_root)


def test_model_ship_cases_fail_closed_on_missing_suites() -> None:
    cases = model_ship_gate_cases(
        {
            "smoke": {
                "n": 32,
                "parse_rate": 1.0,
                "structural_similarity": 1.0,
                "component_type_recall": 1.0,
                "placeholder_fidelity": 1.0,
                "reward_score": 1.0,
                "fallback_count": 0,
            }
        }
    )
    assert [case["id"] for case in cases] == [
        "smoke",
        "held_out",
        "adversarial",
        "ood",
        "rico_held",
    ]
    assert all(item["actual"] is not None for item in cases[0]["assertions"])
    assert all(
        case["assertions"] == [
            {
                "id": f"{case['id']}:missing_suite",
                "actual": None,
                "operator": "present",
                "expected": True,
                "suite": case["id"],
            }
        ]
        for case in cases[1:]
    )


def test_publish_agentv_evaluation_uses_sdk_and_jsonl(tmp_path) -> None:
    published = publish_agentv_evaluation(
        tmp_path,
        name="sdk-wiring",
        claim="fixture_wiring_not_ship",
        cases=[
            {
                "id": "case-1",
                "criteria": "The fixture wiring case passes.",
                "pass": True,
                "checks": {"value_is_one": True},
                "result": {"value": 1},
            }
        ],
        version_stamp={"stamp_schema": "version_stamp/v1"},
    )
    spec = tmp_path / "agentv" / "sdk-wiring.eval.jsonl"
    row = json.loads(spec.read_text(encoding="utf-8"))
    assert row["assert"][0]["type"] == "code-grader"
    assert row["assert"][0]["config"] == {
        "actual": True,
        "expected": True,
        "id": "case-1:domain_criterion",
        "operator": "eq",
    }
    assert "agentv_pass" not in json.loads(row["input"])
    assert Path(published["spec"]).is_absolute()
    assert published["authority"] == "AgentEvals assertions"
    assert published["criteria"]["pass"] is True
    assert published["summary"]["passed"] == 1
    assert published["summary"]["executionErrors"] == 0
    benchmark = json.loads(
        Path(published["artifacts"]["benchmarkPath"]).read_text(encoding="utf-8")
    )
    assert benchmark["version_stamp"]["stamp_schema"] == "version_stamp/v1"


def test_agentv_contract_checks_fail_even_when_pass_flag_is_true(tmp_path) -> None:
    published = publish_agentv_evaluation(
        tmp_path,
        name="checked-contract",
        claim="fixture_wiring_not_ship",
        cases=[
            {
                "id": "bad-value",
                "criteria": "The named contract value must pass.",
                "pass": True,
                "assertions": [{
                    "id": "bad-value:value_is_one",
                    "actual": False,
                    "operator": "eq",
                    "expected": True,
                }],
                "result": {"value": 0},
            }
        ],
    )
    assert published["criteria"]["pass"] is False
    assert published["criteria"]["failed"] == 1


def test_agentv_model_bundle_cannot_pass_a_smoke_only_run(tmp_path) -> None:
    published = publish_model_evaluation(
        tmp_path,
        {
            "smoke": {
                "n": 32,
                "parse_rate": 1.0,
                "structural_similarity": 1.0,
                "component_type_recall": 1.0,
                "placeholder_fidelity": 1.0,
                "reward_score": 1.0,
                "fallback_count": 0,
                "evaluated_at": "2026-07-14T00:00:00+00:00",
            }
        },
    )
    assert published["criteria"]["passed"] == 7
    assert published["criteria"]["failed"] == 4
    assert published["criteria"]["pass"] is False
