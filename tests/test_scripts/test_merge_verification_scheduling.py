"""Scheduling counterexamples exercise canonical owners, not copied algorithms."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts import merge_verification as owner
from scripts.merge_verification_evidence import digest, validate_cached_state
from scripts.verify_merge_ready import Step


def state_for(*nodes):
    return {
        "identity": "fixture",
        "binding": {
            "static_commands": [],
            "max_attempts_per_obligation": 3,
            "isolation_enforced": False,
            "runtime_roots": [],
        },
        "static": {},
        "attempts": [],
        "passed_nodes": [],
        "nodes": list(nodes),
        "shards": [[node] for node in nodes],
    }


def test_small_static_budget_runs_and_missing_obligations_are_not_vacuously_done():
    state = state_for()
    step = Step("probe", ("python", "-c", "pass"))
    calls = []

    def execute(step, **kwargs):
        calls.append(kwargs["budget_seconds"])
        return {"name": step.name, "status": "ok", "exit_code": 0, "seconds": 0.1}

    assert owner._run_statics(
        state, (step,), execute, Path("."), lambda: 40, lambda: None
    )
    assert calls == [40]
    state["static"].clear()
    assert not owner._run_statics(
        state, (step,), execute, Path("."), lambda: 0, lambda: None
    )
    assert len(calls) == 1
    assert next(iter(state["waiting"].values()))["reason"] == "insufficient_budget"


def test_exhausted_first_shard_does_not_starve_healthy_work(monkeypatch):
    bad, good = "tests/bad.py::test_bad", "tests/good.py::test_good"
    state = state_for(bad, good)
    state["attempts"] = [
        {"kind": "shard", "nodes": [bad], "status": "failed", "seconds": 1}
        for _ in range(3)
    ]
    calls = []

    def execute(root, nodes, **kwargs):
        calls.append(nodes)
        return {"status": "ok", "exit_code": 0, "nodes": nodes, "seconds": 0.1}

    monkeypatch.setattr(owner.check_changed, "_test_file_durations", lambda: {})
    monkeypatch.setattr(owner, "run_workload", execute)
    owner._run_shards(state, Path("."), Path("."), lambda: 40, lambda: None)
    assert calls == [[good]]
    assert state["passed_nodes"] == [good]
    assert next(iter(state["waiting"].values()))["reason"] == "retry_exhausted"
    assert not owner._summary(state)["verification_complete"]


def test_failure_does_not_starve_next_shard_or_mint_split_retries(monkeypatch):
    state = state_for("tests/a.py::a", "tests/a.py::b", "tests/b.py::c")
    state["shards"] = [state["nodes"][:2], state["nodes"][2:]]
    calls = []

    def execute(root, nodes, **kwargs):
        calls.append(nodes)
        return {
            "status": "timeout" if len(nodes) > 1 else "ok",
            "nodes": nodes,
            "seconds": 1,
            "exit_code": 0,
        }

    monkeypatch.setattr(owner.check_changed, "_test_file_durations", lambda: {})
    monkeypatch.setattr(owner, "run_workload", execute)
    owner._run_shards(state, Path("."), Path("."), lambda: 40, lambda: None)
    assert len(calls) == 2
    assert state["passed_nodes"] == state["nodes"][2:]
    child = state["shards"][0]
    state["attempts"].extend({"kind": "shard", "nodes": child} for _ in range(2))
    assert owner._attempts_exhausted(state, "shard", child)
    assert not owner._attempts_exhausted(state, "shard", state["shards"][1])


def test_known_long_shard_waits_while_measured_short_shard_uses_tail(monkeypatch):
    slow, fast = "tests/slow.py::slow", "tests/fast.py::fast"
    state = state_for(slow, fast)
    state["workload_budget_seconds"] = 100
    monkeypatch.setattr(
        owner.check_changed,
        "_test_file_durations",
        lambda: {"tests/slow.py": 40, "tests/fast.py": 1},
    )
    calls = []

    def execute(root, nodes, **kwargs):
        calls.append(nodes)
        return {"status": "ok", "nodes": nodes, "seconds": 0.1}

    monkeypatch.setattr(owner, "run_workload", execute)
    owner._run_shards(state, Path("."), Path("."), lambda: 50, lambda: None)
    assert calls == [[fast]]
    assert next(iter(state["waiting"].values()))["required_seconds"] == 81


def test_status_is_bounded_and_full_history_remains_in_journal():
    import json

    state = state_for(*(f"tests/a.py::test_{n}" for n in range(12000)))
    state["attempts"] = [
        {
            "kind": "collection",
            "status": "ok",
            "seconds": 1,
            "workload": {"huge": "x" * 100000},
        }
        for _ in range(100)
    ]
    before = deepcopy(state)
    summary = owner._summary(state)
    assert len(json.dumps(summary)) < 6000
    assert summary["node_counts"] == {"required": 12000, "passed": 0, "pending": 12000}
    assert summary["attempt_count"] == summary["spent_seconds"] == 100
    assert summary["journal_sha256"] == digest(state)
    assert state == before
    assert not summary["release_authorized"]
    detailed = owner._summary(state, full=True)
    assert detailed["nodes"] == state["nodes"]
    assert detailed["steps"] == state["attempts"]


def test_cache_rejects_bad_static_before_collection():
    state = state_for()
    state.pop("nodes")
    state["binding"]["static_commands"] = [["probe", ["python"]]]
    state["identity"] = digest(state["binding"])
    state["static"]["probe"] = {"status": "ok", "exit_code": 1}
    with pytest.raises(ValueError, match="static exit"):
        validate_cached_state(state, state["binding"])


def test_historical_failure_is_not_current_failure_after_verified_retry():
    state = state_for("tests/a.py::a", "tests/b.py::b")
    state["attempts"] = [
        {
            "kind": "shard",
            "nodes": state["nodes"][:1],
            "status": "timeout",
            "seconds": 5,
        }
    ]
    assert owner._summary(state)["status"] == "pending"  # Retryable, not exhausted.
    state["passed_nodes"] = state["nodes"][:1]
    summary = owner._summary(state)
    assert summary["status"] == "pending"
    assert summary["spent_seconds"] == 5
    assert not summary["verification_complete"]


def test_typed_next_action_distinguishes_retry_repair_and_budget(monkeypatch):
    state = state_for("tests/a.py::a")
    state["workload_budget_seconds"] = 30
    monkeypatch.setattr(owner.check_changed, "_test_file_durations", lambda: {})
    owner._allowance(state, "shard", state["nodes"], 5)
    summary = owner._summary(state)
    assert summary["status"] == "pending"
    assert summary["required_seconds"] == 30
    assert summary["next_action"]["kind"] == "resume_verification"
    assert summary["phase_progress"]["tests"]["pending"] == 1
    state["attempts"] = [
        {"kind": "shard", "nodes": state["nodes"], "status": "timeout", "seconds": 30}
        for _ in range(3)
    ]
    owner._allowance(state, "shard", state["nodes"], 30)
    assert owner._summary(state)["status"] == "waiting_repair"
    state["attempts"] = []
    monkeypatch.setattr(
        owner.check_changed, "_test_file_durations", lambda: {"tests/a.py": 60}
    )
    owner._allowance(state, "shard", state["nodes"], 30)
    summary = owner._summary(state)
    assert summary["status"] == "waiting_budget"
    assert summary["required_seconds"] == 121
    assert summary["next_action"]["kind"] == "replan_verification_budget"


def test_design_bridge_entrypoint_identity_includes_path_and_bytes(
    tmp_path, monkeypatch
):
    from scripts import merge_verification_evidence as evidence

    monkeypatch.setattr(evidence.importlib.metadata, "distributions", lambda: [])
    cli = tmp_path / "design-cli.mjs"
    cli.write_text("export const contract = 1;\n")
    monkeypatch.setenv("DESIGN_MD_BRIDGE_CLI", str(cli))
    original = evidence.environment_identity()
    assert original["command_files"]["DESIGN_MD_BRIDGE_CLI"] == [
        str(cli),
        evidence.file_digest(cli),
    ]
    cli.write_text("export const contract = 2;\n")
    assert evidence.environment_identity() != original
    changed = evidence.environment_identity()
    alternate = tmp_path / "other-cli.mjs"
    alternate.write_bytes(cli.read_bytes())
    monkeypatch.setenv("DESIGN_MD_BRIDGE_CLI", str(alternate))
    assert evidence.environment_identity() != changed


def test_cached_collection_cannot_drop_or_duplicate_shards(monkeypatch):
    from scripts import merge_verification_evidence as evidence

    state = state_for("tests/a.py::a", "tests/b.py::b")
    state["identity"] = digest(state["binding"])
    # Partition checking is independent of the already-tested payload validation.
    monkeypatch.setattr(evidence, "_validate_cached_nodes", lambda *_: None)
    for shards in (
        [state["nodes"][:1]],
        [state["nodes"], state["nodes"]],
        [[], state["nodes"]],
    ):
        state["shards"] = shards
        with pytest.raises(ValueError, match="partition"):
            validate_cached_state(state, state["binding"])


@pytest.mark.parametrize("charge", [float("nan"), float("inf"), -1, True, None])
def test_cache_rejects_invalid_charges_before_collection(charge):
    state = state_for()
    state.pop("nodes")
    state["attempts"] = [{"seconds": charge}]
    state["identity"] = digest(state["binding"])
    with pytest.raises(ValueError, match="resource charge"):
        validate_cached_state(state, state["binding"])
