from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_docs_only_merge_verification_runs_static_and_completes(monkeypatch, tmp_path):
    from scripts import merge_verification as verifier

    state = {
        "identity": "a" * 64,
        "binding": {"targets": [], "static_commands": [], "isolation_enforced": False},
        "static": {},
        "attempts": [],
        "passed_nodes": [],
    }
    calls = []

    class Cache:
        directory = tmp_path
        def save(self, value):
            pass

    monkeypatch.setattr(verifier, "_identity_mismatch", lambda *_: "")
    monkeypatch.setattr(
        verifier, "run_isolated_phase",
        lambda _state, _name, fn, *, fast_persist: fn(),
    )
    monkeypatch.setattr(
        verifier, "_run_statics",
        lambda state, *_args: calls.append("static") or True,
    )
    monkeypatch.setattr(
        verifier, "_collect",
        lambda *_args, **_kwargs: pytest.fail("docs-only change must not collect tests"),
    )
    result = verifier._execute_pending(
        state, root=tmp_path, cache=Cache(), steps=(), run_step=lambda *_: None,
        deadline=time.monotonic() + 20, step_seconds=10,
    )
    assert calls == ["static"]
    assert state["no_tests_required"] is True
    assert result["verification_complete"] is True
    assert result["node_counts"] == {"required": 0, "passed": 0, "pending": 0}
    assert result["phase_progress"]["collection"]["completed"] == 1


def test_clean_candidate_selects_full_suite(monkeypatch, tmp_path):
    from scripts import merge_verification as verifier

    monkeypatch.setattr(verifier, "changed_paths", lambda *_: ("base-tree", []))
    monkeypatch.setattr(verifier, "source_identity", lambda *_: "source")
    monkeypatch.setattr(verifier, "environment_identity", lambda: {})
    monkeypatch.setattr(verifier, "runtime_identity", lambda _roots: "runtime")

    binding = verifier.verification_binding(
        tmp_path, "origin/main", (), isolated=False, runtimes=()
    )

    assert binding["changed_paths"] == []
    assert binding["targets"] == ["tests"]


def test_docs_only_is_not_complete_until_static_checks_pass():
    from scripts.merge_verification_summary import summarize

    state = {
        "identity": "b" * 64,
        "binding": {"targets": [], "static_commands": [("static", [])], "isolation_enforced": True},
        "static": {"static": {"status": "failed"}},
        "attempts": [], "passed_nodes": [], "nodes": [], "no_tests_required": True,
    }
    assert summarize(state)["verification_complete"] is False


def test_source_verification_scan_is_bounded_and_rotates(monkeypatch):
    from scripts import autotrain_verification as module

    events = [{"event_type": "source_verification_requested", "experiment_id": f"repair-{i}"} for i in range(100)]
    class Store:
        def verify_event_chain(self):
            return events
    class Runtime:
        store = Store()
        def snapshot(self):
            return {f"repair-{i}": SimpleNamespace(status="waiting_dependency", wake="wake", attempts=0) for i in range(100)}
        def claim_next(self, *, capabilities, activity_id):
            seen.append(activity_id)
            return None
    seen = []
    monkeypatch.setattr(module, "load_dependency", lambda _store, event: {"activity_id": event["experiment_id"], "wake": "wake"})
    monkeypatch.setattr(module, "WakeCondition", SimpleNamespace(model_validate=lambda _value: "wake"))
    monkeypatch.setattr(module, "dependency_plan", lambda dependency: dependency)
    monkeypatch.setattr(module, "wake_repair", lambda *_args: False)
    monkeypatch.setattr(module, "register_dependency", lambda *_args: None)
    import slm_training.autoresearch.heal.isolation as isolation
    monkeypatch.setattr(isolation, "probe_isolation", lambda: SimpleNamespace(available=False))
    monkeypatch.setattr(module, "KILL_GRACE_SECONDS", 100)
    monkeypatch.setattr(module.time, "monotonic", lambda: 1.0)

    module.drain_source_verification(Runtime(), {}, lambda _event: None, cycle=1)
    first = list(seen)
    module.drain_source_verification(Runtime(), {}, lambda _event: None, cycle=2)
    assert len(first) == 32
    assert len(seen) == 64
    assert len(set(seen)) == 64


def test_recovery_config_digest_is_checked_on_the_single_read(tmp_path, monkeypatch):
    from slm_training.autoresearch.heal import recovery_dispatch as dispatch

    path = tmp_path / "recovery.json"
    path.write_text("{}", encoding="utf-8")
    raw = b'{"exact":"bound bytes"}'
    reads = []
    monkeypatch.setattr(Path, "read_bytes", lambda self: reads.append(self) or raw)
    expected = hashlib.sha256(raw).hexdigest()
    sentinel = object()
    monkeypatch.setattr(dispatch.RecoveryConfig, "model_validate_json", lambda value: sentinel)
    assert dispatch.load_recovery_config(path, expected_sha256=expected) is sentinel
    assert reads == [path]
    with pytest.raises(ValueError, match="repair authority changed"):
        dispatch.load_recovery_config(path, expected_sha256="0" * 64)
    assert len(reads) == 2


def test_repair_controller_rejects_config_without_digest(tmp_path):
    from scripts.autotrain_controller_repair import dispatch_controller_repairs

    path = tmp_path / "recovery.json"
    path.write_text("{}", encoding="utf-8")
    request = {
        "lease": {"activity_id": "repair", "attempt_id": "attempt", "epoch": "epoch",
                  "generation": 1, "token": "fence", "owner_identity": "controller",
                  "expires_at": 1.0},
        "source_digest": "a" * 64, "environment_digest": "b" * 64,
        "parent_event": "parent", "campaign_id": "campaign",
        "hard_pending": [], "repair_config": str(path),
    }
    with pytest.raises(ValueError, match="repair config digest missing"):
        dispatch_controller_repairs(request, cwd=tmp_path, root=tmp_path / "campaigns", loop_id="loop")


def test_failed_precycle_advances_no_campaign_watchdog(monkeypatch, tmp_path):
    from scripts import autotrain_supervision as supervision

    class Store:
        def __init__(self):
            self.events = []
        def verify_event_chain(self):
            return list(self.events)
        def append_event(self, kind, **kwargs):
            self.events.append({"event_type": kind, "detail": kwargs.get("detail", {}), "event": kind})
    class Cancel:
        def __init__(self):
            self.waited = []
        def is_set(self):
            return False
        def wait(self, seconds):
            self.waited.append(seconds)
    cancel = Cancel()
    runtime = SimpleNamespace(store=Store(), cancel_event=cancel)
    args = SimpleNamespace(root=tmp_path, loop_id="bounded-precycle", max_cycles=1,
                           stop_after_pass=None, hard_backoff_seconds=1)
    observed = []
    monkeypatch.setattr(supervision, "pre_cycle", lambda *_args: None)
    def watchdog(**kwargs):
        observed.append(kwargs)
        return 7.0
    assert supervision.supervise(args, runtime, {"root": tmp_path},
                                  run_operation=lambda *_args, **_kwargs: None,
                                  watchdog=watchdog) == 0
    assert len(observed) == 1
    assert observed[0]["passes_without_campaign"] == 1
    assert observed[0]["campaign_id"] is None
    assert cancel.waited == [7.0]


def test_split_parent_timeouts_do_not_exhaust_unrun_child_obligations():
    from scripts.merge_verification_shards import attempts_exhausted

    nodes = ["test_x.py::test_a", "test_x.py::test_b"]
    state = {
        "binding": {"max_attempts_per_obligation": 8},
        "attempts": [{"kind": "shard", "status": "timeout", "nodes": nodes} for _ in range(8)],
    }
    child = [nodes[0]]
    assert not attempts_exhausted(state, "shard", child)
    state["attempts"].extend(
        {"kind": "shard", "status": "timeout", "nodes": child} for _ in range(7)
    )
    assert not attempts_exhausted(state, "shard", child)
    state["attempts"].append({"kind": "shard", "status": "timeout", "nodes": child})
    assert attempts_exhausted(state, "shard", child)


def test_collection_parent_timeouts_still_charge_split_batch_children():
    from scripts.merge_verification_shards import attempts_exhausted

    nodes = ["tests/a.py", "tests/b.py"]
    state = {"binding": {"max_attempts_per_obligation": 2}, "attempts": [
        {"kind": "collection", "status": "timeout", "nodes": nodes},
        {"kind": "collection", "status": "timeout", "nodes": nodes},
    ]}
    assert attempts_exhausted(state, "collection", nodes[:1])


def test_adaptive_shards_preserve_node_union_across_repeated_splits(tmp_path):
    from scripts.merge_verification_shards import run_shards

    original = [f"test_{i}.py::test_{i}" for i in range(4)]
    state = {
        "shards": [original], "passed_nodes": [], "attempts": [], "shard_cursor": 0,
        "binding": {"isolation_enforced": False, "runtime_roots": [],
                    "max_attempts_per_obligation": 8},
    }
    def workload(_root, nodes, **_kwargs):
        return {"status": "timeout", "nodes": list(nodes), "seconds": 50.0}
    def allowance(_state, _kind, _nodes, available, **_kwargs):
        return min(50.0, available)
    for _ in range(2):
        run_shards(state, tmp_path, tmp_path, lambda: 100.0, lambda: None,
                   allowance=allowance, run_workload=workload)
    flattened = [node for shard in state["shards"] for node in shard]
    assert sorted(flattened) == sorted(original)
    assert len(flattened) == len(set(flattened))
    assert all(len(shard) == 1 for shard in state["shards"])
