from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
                "meaningful_program_rate": 1.0,
                "structural_similarity": 1.0,
                "component_type_recall": 1.0,
                "ast_beq_rate": 1.0,
                "canonical_beq_rate": 1.0,
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


def test_model_ship_cases_publish_reachability_as_raw_assertion() -> None:
    cases = model_ship_gate_cases(
        {
            "smoke": {
                "n": 32,
                "meaningful_program_rate": 1.0,
                "structural_similarity": 1.0,
                "component_type_recall": 1.0,
                "placeholder_fidelity": 1.0,
                "reward_score": 1.0,
                "fallback_count": 0,
            }
        },
        include_missing_suites=False,
        suite_reachability={"smoke": 0.5},
    )

    assertion = next(
        item
        for item in cases[0]["assertions"]
        if item["id"] == "smoke:reachability_unproven"
    )
    assert assertion == {
        "id": "smoke:reachability_unproven",
        "suite": "smoke",
        "actual": 0.5,
        "operator": "eq",
        "expected": 1.0,
    }


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


def test_agentv_forwards_w3c_trace_id_to_the_node_runner(tmp_path, monkeypatch) -> None:
    runner = tmp_path / "runner.mjs"
    runner.write_text("// fixture")
    sdk_root = tmp_path / "sdk-root"
    captured = {}
    (tmp_path / "trace.json").write_text(
        json.dumps({"trace_id": "0123456789abcdef0123456789abcdef"})
    )
    monkeypatch.setattr(agentv_module, "_agentv_runtime", lambda _: (runner, sdk_root))

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"summary": {}, "artifacts": {}}),
            stderr="",
        )

    monkeypatch.setattr(agentv_module.subprocess, "run", fake_run)
    publish_agentv_evaluation(
        tmp_path,
        name="trace-link",
        claim="fixture_wiring_not_ship",
        cases=[{"id": "case", "criteria": "passes", "pass": True}],
    )
    assert captured["command"][-4:] == [
        "--trace-id",
        "0123456789abcdef0123456789abcdef",
        "--run-id",
        tmp_path.name,
    ]


def test_agentv_node_subprocess_clears_node_options(
    tmp_path, monkeypatch
) -> None:
    runner = tmp_path / "runner.mjs"
    runner.write_text("// fixture")
    sdk_root = tmp_path / "sdk-root"
    captured = {}
    monkeypatch.setattr(agentv_module, "_agentv_runtime", lambda _: (runner, sdk_root))
    monkeypatch.setenv("NODE_OPTIONS", "--import tsx")

    def fake_run(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"summary": {}, "artifacts": {}}),
            stderr="",
        )

    monkeypatch.setattr(agentv_module.subprocess, "run", fake_run)
    publish_agentv_evaluation(
        tmp_path,
        name="node-options",
        claim="fixture_wiring_not_ship",
        cases=[{"id": "case", "criteria": "passes", "pass": True}],
    )
    assert captured["env"] is not None
    assert captured["env"]["NODE_OPTIONS"] == ""


def test_agentv_model_bundle_cannot_pass_a_smoke_only_run(tmp_path) -> None:
    published = publish_model_evaluation(
        tmp_path,
        {
            "smoke": {
                "n": 32,
                "parse_rate": 1.0,
                "meaningful_program_rate": 1.0,
                "structural_similarity": 1.0,
                "component_type_recall": 1.0,
                "ast_beq_rate": 1.0,
                "canonical_beq_rate": 1.0,
                "placeholder_fidelity": 1.0,
                "reward_score": 1.0,
                "fallback_count": 0,
                "evaluated_at": "2026-07-14T00:00:00+00:00",
            }
        },
    )
    assert published["criteria"]["passed"] == 9
    assert published["criteria"]["failed"] == 4
    assert published["criteria"]["pass"] is False
