"""Finite measurement CLI compilation and controller contracts, not neural gains."""

import json
import sys
from pathlib import Path

import pytest

from tests.casefiles import case_values

from slm_training.harnesses.experiments.autonomous_learning import (
    measurement_fixture as fixture,
)
from slm_training.harnesses.experiments.autonomous_learning.measurement_fixture_evidence import (
    project_paths,
    verify_agentv,
)


def test_documentation_paths_are_portable_without_altering_evidence():
    absolute = str(Path.cwd() / "outputs/fixture.json")
    original = {absolute: {"digest": "a" * 64, "count": 6}}
    assert project_paths(original) == {
        "outputs/fixture.json": {"digest": "a" * 64, "count": 6}
    }
    assert absolute in original
    with pytest.raises(ValueError, match="collide"):
        project_paths({absolute: 1, "outputs/fixture.json": 2})


pytest_plugins = ("tests.test_harnesses.experiments.measurement_fixture_support",)


def test_compiled_commands_lock_current_endpoint_and_preserve_full_selection(prepared):
    from scripts.autotrain_measurement import locked_primary_failure
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        primary_for_role,
    )

    store, plan = prepared
    primary = primary_for_role(load_climb_policy(), "screening")
    assert (
        locked_primary_failure(
            store.root, "control", "candidate", primary["metric"], primary
        )
        is None
    )
    for arm in plan["arms"].values():
        decode = arm["commands"]["decode"]
        assert decode[:3] == [sys.executable, "-m", "scripts.evaluate_model"]
        assert decode[decode.index("--eval-limit") + 1] == "6"
        assert decode[decode.index("--max-records-this-run") + 1] == "2"
        assert decode[decode.index("--seed") + 1] == "7301"
        assert "--ship-gates" in decode and "--no-unconstrained-fallback" in decode
        assert "--checkpoint" in decode and "--resume-run" in decode
        assert any("scripts.train_model" in command for command in arm["compiled"])
        assert all(
            "scripts.train_model" not in command for command in arm["commands"].values()
        )
    assert plan["maximum_child_attempts"] == 8
    assert not plan["new_training"] and not plan["promotion_allowed"]


@pytest.mark.parametrize(
    "component,version",
    [
        ("harness.model_build.eval", "v108"),
    ],
)
def test_old_version_cannot_start_new_measurement(component, version):
    components = {
        "harness.model_build.eval": "v109",
        "evals.scoring": "v28",
        "harness.experiments": "v169",
    }
    components[component] = version
    with pytest.raises(ValueError, match="release version pending"):
        fixture.require_release_versions({"components": components})


def test_current_measurement_does_not_wait_for_unrelated_release_composition():
    fixture.require_release_versions(
        {
            "components": {
                "harness.model_build.eval": "v109",
                "evals.scoring": "v27",
                "harness.experiments": "v168",
            }
        }
    )


def test_plan_edit_cannot_change_locked_endpoint(prepared):
    store, plan = prepared
    plan["primary"]["metric"] = "binder_reference_f1"
    fixture._write(store.root / "measurement_fixture.json", plan)
    with pytest.raises(ValueError, match="canonical event lock"):
        fixture.load_plan(store)


def test_source_successor_refuses_before_loading_checkpoint(prepared, monkeypatch):
    store, _ = prepared
    monkeypatch.setattr(fixture, "source_identity", lambda: {"source": "changed"})
    with pytest.raises(ValueError, match="source/policy/version changed"):
        fixture.load_plan(store)


def test_repeated_activity_cannot_mint_fresh_resource_budget(prepared, monkeypatch):
    store, plan = prepared
    monkeypatch.setattr(fixture, "load_plan", lambda _: plan)
    receipts = store.root / "measurement_receipts"
    receipts.mkdir()
    (receipts / "control-loss.started.json").write_text("{}")
    monkeypatch.setattr(
        fixture,
        "run_bounded_process",
        lambda *a, **k: pytest.fail("launched duplicate"),
    )
    with pytest.raises(ValueError, match="already attempted"):
        fixture.run_activity(store, "control-loss")


def test_pending_chunk_and_final_gate_rejection_are_distinct(prepared, monkeypatch):
    from slm_training.harness_core.bounded_process import (
        BoundedProcessResult,
        ProcessOutcome,
    )

    store, plan = prepared
    monkeypatch.setattr(fixture, "load_plan", lambda _: plan)
    exits = iter((10, 10, 8))

    def child(command, **limits):
        assert limits["interrupt_after_seconds"] == fixture.INTERRUPT_AFTER_SECONDS
        assert limits["kill_grace_seconds"] == fixture.KILL_GRACE_SECONDS
        return BoundedProcessResult(
            tuple(command), ProcessOutcome.COMPLETED, next(exits), "", "", 0.01
        )

    monkeypatch.setattr(fixture, "run_bounded_process", child)
    for index, expected in enumerate((10, 10, 8), 1):
        result = fixture.run_activity(store, f"control-decode{index}")
        assert result["result"]["returncode"] == expected
    assert len(list((store.root / "measurement_receipts").glob("*.started.json"))) == 3


def test_cli_requires_explicit_fixture_activation(tmp_path):
    with pytest.raises(SystemExit) as exc:
        fixture.main(["prepare", "--run-id", "test", "--output-root", str(tmp_path)])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_agentv_execution_is_not_domain_gate_success(tmp_path):
    run = tmp_path / "sdk"
    run.mkdir()
    spec = run / "spec.jsonl"
    index = run / "index.jsonl"
    benchmark = run / "benchmark.json"
    timing = run / "timing.json"
    spec.write_text(json.dumps({"id": "case"}) + "\n")
    index.write_text(json.dumps({"id": "case", "score": 0}) + "\n")
    benchmark.write_text("{}")
    timing.write_text("{}")
    payload = {
        "sdk": "@agentv/core",
        "runner": {"execution_errors": 0},
        "summary": {"executionErrors": 0, "total": 1},
        "criteria": {"pass": False},
        "spec": str(spec),
        "artifacts": {
            "runDir": str(run),
            "indexPath": str(index),
            "benchmarkPath": str(benchmark),
            "timingPath": str(timing),
        },
    }
    assert len(verify_agentv(payload)) == 4
    index.write_text("")
    with pytest.raises(ValueError, match="artifact missing"):
        verify_agentv(payload)


def test_fixture_module_parser_help_is_runnable():
    with pytest.raises(SystemExit) as exc:
        fixture.main(["--help"])
    assert exc.value.code == 0


@pytest.fixture
def complete_arm(tmp_path, monkeypatch):
    from slm_training.evals import denoising_nll
    from slm_training.evals.measurement_identity import loss_record_rows
    from slm_training.harnesses.experiments.autonomous_learning import (
        measurement_fixture_evidence as evidence,
    )
    from slm_training.harnesses.model_build.eval_measurement import evaluator_identity
    from slm_training.versioning import build_version_stamp

    selection = {
        "selected_record_ids": [f"case-{i}" for i in range(6)],
        "selected_root_ids": [f"root-{i}" for i in range(6)],
        "input_sha256s": [str(i) * 64 for i in range(6)],
        "selection_sha256": "e" * 64,
    }
    stamp = build_version_stamp(
        "harness.model_build.eval", "evals.scoring", "model.twotower"
    )
    evaluator = fixture._sha(denoising_nll.__file__)
    broad = [
        dict(
            id=case,
            case_id=case,
            root_id=selection["selected_root_ids"][i],
            input_sha256=selection["input_sha256s"][i],
            seed=7301,
            estimator_id="test-estimator",
            evaluator_sha256=evaluator,
            selection_sha256="e" * 64,
            mean_nll=1.0,
            masked_tokens=2,
            nll_sum=2.0,
            units="nats_per_masked_token",
        )
        for i, case in enumerate(selection["selected_record_ids"])
    ]
    loss = {
        "selection": selection,
        "checkpoint": str(tmp_path / "last.pt"),
        "categories": {"broad": {"per_record": broad}},
        "estimator_id": "test-estimator",
        "version_stamp": build_version_stamp("evals.loss_suite", "model.twotower"),
        "agentv": {},
    }
    loss["per_record"] = loss_record_rows(loss["categories"])
    metrics = {
        "n": 6,
        "document_n": 6,
        "completed_document_n": 6,
        "incomplete_document_n": 0,
        "decode_timeout_count": 0,
        "measurement_complete": True,
        "checkpoint_sha256": "a" * 64,
        "version_stamp": stamp,
        "evaluator_sha256": evaluator_identity(stamp),
        **selection,
    }
    scoreboard = {
        "suites": {"smoke": metrics},
        "measurement_complete": True,
        "publication_complete": True,
        "evals": {},
    }
    plan = {
        "arms": {"control": {"run_dir": str(tmp_path)}},
        "inputs": {
            "selection": selection,
            "arms": {
                "control": {
                    "checkpoint": str(tmp_path / "last.pt"),
                    "checkpoint_sha256": "a" * 64,
                }
            },
        },
        "identity": {
            "source_files": {"src/slm_training/evals/denoising_nll.py": evaluator}
        },
    }
    # SDK plumbing is independently exercised above and by the actual CLI run.
    monkeypatch.setattr(evidence, "verify_agentv", lambda payload: {})
    fixture._write(tmp_path / "scoreboard.json", scoreboard)
    fixture._write(tmp_path / "loss_suites.json", loss)
    return evidence, plan, scoreboard, loss, tmp_path


def test_collector_accepts_exact_current_complete_producer_shape(complete_arm):
    evidence, plan, _, _, _ = complete_arm
    assert len(evidence.checked_arm(plan, "control")["records"]) == 6


@pytest.mark.parametrize(
    "fault",
    case_values(__file__, "test_collector_rejects_stale_or_incomplete_evidence"),
)
def test_collector_rejects_stale_or_incomplete_evidence(complete_arm, fault):
    evidence, plan, scoreboard, loss, root = complete_arm
    metrics = scoreboard["suites"]["smoke"]
    if fault == "old_version":
        metrics["version_stamp"]["components"]["harness.model_build.eval"] = "v1"
    elif fault == "old_evaluator":
        metrics["evaluator_sha256"] = "0" * 64
    elif fault == "wrong_checkpoint":
        metrics["checkpoint_sha256"] = "b" * 64
    elif fault == "pending":
        metrics["measurement_complete"] = False
    elif fault == "missing_case":
        loss["per_record"].pop()
    elif fault == "wrong_input":
        loss["categories"]["broad"]["per_record"][0]["input_sha256"] = "b" * 64
    fixture._write(root / "scoreboard.json", scoreboard)
    fixture._write(root / "loss_suites.json", loss)
    with pytest.raises(ValueError):
        evidence.checked_arm(plan, "control")
