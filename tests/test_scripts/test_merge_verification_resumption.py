"""Regression boundaries for coverage, cursor replay, and source immutability."""

import json
from pathlib import Path

import pytest

from scripts import merge_verification as gate
from scripts import check_changed
from tests.test_scripts.test_merge_verification_scheduling import state_for


def test_budget_exhaustion_preserves_next_pending_cursor_across_json_restart(monkeypatch):
    nodes = [f"tests/test_case.py::test_{i}" for i in range(512)]
    state = state_for(*nodes)
    state["passed_nodes"] = nodes[:262] + nodes[264:]
    state["shard_cursor"] = 500
    state["shard_budget_seconds"] = state["workload_budget_seconds"] = 30
    calls = []

    def execute(root, targets, **kwargs):
        calls.extend(targets)
        return {"status": "ok", "nodes": targets, "seconds": 1}

    monkeypatch.setattr(gate.check_changed, "_test_file_durations", lambda: {})
    monkeypatch.setattr(gate, "run_workload", execute)
    budget = iter([40, 5])
    gate._run_shards(state, Path("."), Path("."), lambda: next(budget), lambda: None)
    assert calls == [nodes[262]]
    assert state["shard_cursor"] == 263
    # A restart restores the signed-state representation, without any manual
    # cursor adjustment or deletion of already-completed shard obligations.
    state = json.loads(json.dumps(state))
    gate._run_shards(state, Path("."), Path("."), lambda: 40, lambda: None)
    assert calls == nodes[262:264]
    assert len(state["shards"]) == 512
    assert sorted(state["passed_nodes"]) == sorted(nodes)


def test_split_shards_wrap_and_resume_without_repeating_completed_nodes(monkeypatch):
    nodes = [f"tests/test_case.py::test_{i}" for i in range(4)]
    state = state_for(*nodes)
    state["shards"] = [nodes[:2], nodes[2:]]
    calls = []

    def execute(root, targets, **kwargs):
        calls.append(targets)
        return {"status": "timeout" if len(calls) == 1 else "ok", "nodes": targets, "seconds": 1}

    monkeypatch.setattr(gate.check_changed, "_test_file_durations", lambda: {})
    monkeypatch.setattr(gate, "run_workload", execute)
    gate._run_shards(state, Path("."), Path("."), lambda: 40, lambda: None)
    assert state["passed_nodes"] == nodes[2:]
    state = json.loads(json.dumps(state))
    gate._run_shards(state, Path("."), Path("."), lambda: 40, lambda: None)
    assert sorted(state["passed_nodes"]) == nodes
    assert calls.count(nodes[2:]) == 1
    child_state = {
        "binding": {"max_attempts_per_obligation": 2},
        "attempts": [{"kind": "shard", "nodes": nodes[:2], "status": "timeout"}],
    }
    assert not gate._attempts_exhausted(child_state, "shard", nodes[:1])
    child_state["attempts"].append(
        {"kind": "shard", "nodes": nodes[:1], "status": "timeout"}
    )
    assert not gate._attempts_exhausted(child_state, "shard", nodes[:1])
    child_state["attempts"].append(
        {"kind": "shard", "nodes": nodes[:1], "status": "timeout"}
    )
    assert gate._attempts_exhausted(child_state, "shard", nodes[:1])


def test_source_drift_cannot_be_returned_as_completed_verification(tmp_path, monkeypatch):
    state = state_for("tests/test_case.py::test_case")
    state["passed_nodes"] = state["nodes"].copy()
    saved = []
    cache = type("Cache", (), {"directory": tmp_path, "save": lambda self, s: saved.append(dict(s))})()
    monkeypatch.setattr(gate, "_identity_mismatch", lambda *args: "source_changed_during_verification")
    with pytest.raises(ValueError, match="source_changed_during_verification"):
        gate._execute_pending(state, root=tmp_path, cache=cache, steps=(), run_step=None,
                              deadline=gate.time.monotonic() + 60, step_seconds=30)
    assert saved[-1]["invalidated"] == "source_changed_during_verification"


@pytest.mark.parametrize("unknown", ["ops/new.sh", "docs/tool.py", "new.wasm", "uv.lock"])
def test_unknown_executable_alongside_source_and_test_keeps_full_scope(tmp_path, unknown):
    tests = tmp_path / "tests/test_web"
    tests.mkdir(parents=True)
    (tests / "test_one.py").write_text("def test_one(): pass\n")
    changed = ["src/slm_training/web/routes.py", "tests/test_web/test_one.py"]
    assert check_changed.select_tests(changed, root=tmp_path) == ["tests/test_web"]
    assert check_changed.select_tests([*changed, unknown], root=tmp_path) == ["tests"]


def test_deleted_and_renamed_test_keeps_surviving_owner(tmp_path):
    tests = tmp_path / "tests/test_web"
    tests.mkdir(parents=True)
    (tests / "test_new.py").write_text("def test_new(): pass\n")
    assert check_changed.select_tests(
        ["tests/test_web/test_old.py", "tests/test_web/test_new.py"], root=tmp_path
    ) == ["tests/test_web"]


def test_controller_dependency_never_downgrades_isolation(tmp_path, monkeypatch):
    from scripts import autotrain_verification as dependency_owner
    from slm_training.autoresearch.heal import isolation
    from tests.test_scripts.test_autotrain_verification import fixture_dependency

    dependency = fixture_dependency(tmp_path, monkeypatch)
    original = dependency_owner.verification_binding
    observed = []

    def binding(*args, **kwargs):
        observed.append(kwargs["isolated"])
        return original(*args, **kwargs)

    monkeypatch.setattr(dependency_owner, "verification_binding", binding)
    monkeypatch.setattr(isolation, "probe_isolation", lambda: type("Probe", (), {"available": False})())
    plan = dependency_owner.dependency_plan(dependency)
    assert observed == [True]
    assert plan["local_feedback"] is False
    dependency["grant"]["interrupt_seconds"] = 55
    with pytest.raises(dependency_owner.VerificationCapabilityUnavailable, match="startup_reserve"):
        dependency_owner.dependency_plan(dependency)


def test_split_collection_inherits_retry_charge_and_reports_exact_wait():
    state = state_for()
    state.pop("nodes")
    state["binding"]["targets"] = ["tests"]
    batch = ["tests/test_one.py", "tests/test_two.py"]
    state["collection_batches"] = [[batch[0]], [batch[1]]]
    state["collection_batch_index"] = 0
    state["attempts"] = [
        {"kind": "collection", "nodes": batch, "status": "timeout", "seconds": 1}
        for _ in range(state["binding"]["max_attempts_per_obligation"])
    ]
    assert gate._allowance(state, "collection", [batch[0]], 40) == 0
    assert gate._summary(state)["status"] == "waiting_repair"


def test_directory_collection_includes_both_pytest_default_filename_patterns(tmp_path):
    from scripts.merge_verification_collection import collect

    tests = tmp_path / "tests"
    tests.mkdir()
    expected = ["tests/example_test.py", "tests/test_example.py"]
    for path in expected:
        (tmp_path / path).write_text("def test_one(): pass\n")
    state = state_for()
    state.pop("nodes")
    state["binding"]["targets"] = ["tests"]
    state["shard_budget_seconds"] = 30
    calls = []

    def execute(root, targets, **kwargs):
        calls.extend(targets)
        return {"status": "ok", "nodes": [p + "::test_one" for p in targets]}

    assert collect(state, tmp_path, tmp_path, lambda: 40, lambda: None,
                   allowance=lambda *args: 40, run_workload=execute,
                   plan_shards=lambda nodes, _: [nodes])
    assert calls == expected


@pytest.mark.parametrize("passes", [True, False])
def test_canonical_full_entrypoint_runs_actual_requested_tree(tmp_path, monkeypatch, capsys, passes):
    import subprocess
    import sys

    from scripts import verify_merge_ready as entrypoint

    root = tmp_path / "candidate with spaces"
    (root / "tests").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True, timeout=10)
    (root / "tests/test_case.py").write_text(f"def test_case(): assert {passes}\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(gate, "changed_paths", lambda *_: ("fixture-base", ["tests/test_case.py"]))
    step = entrypoint.Step("location", (sys.executable, "-c", "import pathlib; assert pathlib.Path('tests/test_case.py').exists()"))
    monkeypatch.setattr(entrypoint, "merge_gate_steps", lambda: (step,))
    result = entrypoint.main([
        "--source", str(root), "--base-ref", "fixture-base", "--state-dir", str(tmp_path / "receipts"),
        "--runtime-root", str(runtime), "--max-step-seconds", "30", "--local-feedback", "--json",
    ])
    summary = json.loads(capsys.readouterr().out)
    assert summary["verification_complete"] is passes
    assert summary["release_authorized"] is False
    assert summary["node_counts"] == {"required": 1, "passed": int(passes), "pending": int(not passes)}
    assert result == (0 if passes else 10)
