from __future__ import annotations

import json
from pathlib import Path

import slm_training.evals.agentv as agentv_module

from slm_training.evals.agentv import (
    MODEL_QUALITY_METRICS,
    _agentv_runtime,
    apply_agentv_metric_results,
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


def test_model_cases_publish_every_named_domain_metric() -> None:
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
    assert "pass" not in cases[0]
    assert set(cases[0]["result"]["metrics"]) == set(MODEL_QUALITY_METRICS)
    assert cases[0]["result"]["metrics"]["parse_rate"] == 1.0
    assert cases[1]["result"]["metrics"]["parse_rate"] is None


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
                "result": {"value": 1},
            }
        ],
    )
    spec = tmp_path / "agentv" / "sdk-wiring.eval.jsonl"
    row = json.loads(spec.read_text(encoding="utf-8"))
    assert row["assert"] == [{"required": True, "type": "is-json"}]
    assert "agentv_pass" not in json.loads(row["input"])
    assert Path(published["spec"]).is_absolute()
    assert published["sdk"] == "@agentv/core"
    assert published["summary"]["executionErrors"] == 0


def test_agentv_model_bundle_reports_named_metrics_for_a_smoke_run(tmp_path) -> None:
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
    metrics = published["metric_results"]["smoke"]
    assert set(metrics) == set(MODEL_QUALITY_METRICS)
    assert metrics["parse_rate"] == {"value": 1.0, "defined_n": 32}
    assert metrics["ast_node_f1"] == {"value": None, "defined_n": 0}


def test_apply_agentv_metric_results_rejects_missing_or_mismatched_metrics() -> None:
    local = {
        "n": 2,
        "parse_rate": 0.5,
        "metric_defined_n": {"parse_rate": 2},
        **{metric: None for metric in MODEL_QUALITY_METRICS if metric != "parse_rate"},
    }
    results = {
        metric: {"value": None, "defined_n": 0}
        for metric in MODEL_QUALITY_METRICS
    }
    results["parse_rate"] = {"value": 0.5, "defined_n": 2}
    publication = {
        "format": "AgentEvals JSONL",
        "summary": {"executionErrors": 0},
        "metric_results": {"smoke": results},
    }

    apply_agentv_metric_results(local, publication, "smoke")

    assert local["metric_evaluator"]["sdk"] == "@agentv/core"
    assert "agentv" not in local
