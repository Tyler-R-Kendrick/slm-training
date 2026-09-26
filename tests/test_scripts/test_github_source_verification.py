"""Source gate scheduling preserves exact base identity and finite grants."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import github_source_verification as owner
from tests.test_scripts.test_github_source_delivery import accepted as accepted


@pytest.fixture
def scheduling(accepted, monkeypatch):
    from slm_training.autoresearch.runtime import operations_verification
    from slm_training.autoresearch.heal import isolation

    store, wait, host, _ = accepted
    calls, registered = [], []
    complete = {"value": False}
    host.verification_plan["grant"] = {"interrupt_seconds": 90, "total_seconds": 180, "max_attempts": 2}
    plan = {"activity_id": "source-verification-" + "b" * 64}

    def gate(_):
        if not complete["value"]:
            raise ValueError("authenticated_local_merge_gate_incomplete_or_changed")

    def dependency(value):
        calls.append(copy.deepcopy(value))
        assert value["base_ref"] == host.base_ref
        assert value["root"] == host.verification_plan["source"]
        return plan

    monkeypatch.setattr(operations_verification, "require_local_gate", gate)
    monkeypatch.setattr(owner, "dependency_plan", dependency)
    monkeypatch.setattr(owner, "register_dependency", lambda runtime, value: registered.append(value))
    monkeypatch.setattr(owner, "authenticated_completion", lambda value: complete["value"])
    monkeypatch.setattr(isolation, "probe_isolation", lambda: SimpleNamespace(available=True))
    monkeypatch.setattr(owner, "execute_release_attempt", lambda *args: calls.append("attempt"))
    runtime = SimpleNamespace(store=store, claim_next=lambda **kw: calls.append(kw) or "lease")
    return runtime, wait, host, calls, registered, complete


def test_incomplete_gate_advances_existing_finite_verifier_without_writer(scheduling):
    runtime, wait, host, calls, registered, _ = scheduling
    assert not owner.advance_source_gate(runtime, wait, host)
    assert calls[0]["grant"] == host.verification_plan["grant"]
    from slm_training.autoresearch.heal.repair_acceptance import source_verification_activity_id
    assert calls[0]["activity_id"] == source_verification_activity_id(
        host.verification_plan["identity"], host.verification_plan["grant"]
    )
    assert calls[-1] == "attempt" and len(registered) == 1
    event = runtime.store.verify_event_chain()[-1]
    assert event["detail"]["predicate"] == "source_delivery_full_gate_pending"
    assert event["detail"]["writer_started"] is False


def test_no_gate_grant_never_borrows_delivery_attempt(scheduling):
    runtime, wait, host, calls, registered, _ = scheduling
    host.verification_plan.pop("grant")
    assert not owner.advance_source_gate(runtime, wait, host)
    assert calls == registered == []
    assert runtime.store.verify_event_chain()[-1]["detail"]["predicate"] == "source_delivery_verification_grant_required"


def test_complete_gate_replay_does_not_refill_or_reexecute(scheduling):
    runtime, wait, host, calls, registered, complete = scheduling
    complete["value"] = True
    assert owner.advance_source_gate(runtime, wait, host)
    assert calls == registered == []


def test_wrong_cumulative_base_binding_stops_before_verifier(scheduling, monkeypatch):
    runtime, wait, host, calls, registered, _ = scheduling

    def reject(dependency):
        raise ValueError("source_verification_binding_changed")

    monkeypatch.setattr(owner, "dependency_plan", reject)
    with pytest.raises(ValueError, match="binding_changed"):
        owner.advance_source_gate(runtime, wait, host)
    assert calls == registered == []


def test_true_base_preparation_replay_locks_host_candidate_and_gate(accepted, tmp_path, monkeypatch):
    from scripts import github_source_preparation as preparation

    store, wait, host, subject = accepted
    repository = tmp_path / "base-repository"
    repository.mkdir()
    plan = {"source": str(tmp_path / "controller/candidate"), "state_dir": str(tmp_path / "controller/cache"),
            "execution": subject["successor_execution"], "repository_source": str(repository)}
    host.verification_plan = plan
    host.model_copy = lambda update: SimpleNamespace(**{**vars(host), **update})
    calls = []

    def materialize(root, repo, candidate, base):
        import shutil
        calls.append((repo, base))
        shutil.copytree(candidate, root)

    monkeypatch.setattr(preparation, "_materialize", materialize)
    monkeypatch.setattr(preparation, "source_paths", lambda _: ["module.py"])
    monkeypatch.setattr(preparation, "verification_binding", lambda *a, **kw: {"true_base": a[1], "source": str(a[0])})
    monkeypatch.setattr(preparation, "verification_environment", lambda: {})
    first = preparation.resolve_source_host(store, wait, host)
    second = preparation.resolve_source_host(store, wait, host)
    assert first.verification_plan == second.verification_plan and len(calls) == 1
    assert "identity" not in plan  # Never silently rewrite the trusted host file.
    record = Path(plan["source"]).parent / "candidate.delivery-input.json"
    assert json.loads(record.read_text())["plan"] == first.verification_plan
    host.verification_plan["state_dir"] += "-other"
    with pytest.raises(ValueError, match="preparation_binding_changed"):
        preparation.resolve_source_host(store, wait, host)


def test_private_preparation_cannot_expose_cache_to_candidate(tmp_path):
    from scripts.github_source_preparation import _destinations

    with pytest.raises(ValueError, match="issuer_exposed"):
        _destinations(tmp_path / "candidate", tmp_path / "candidate/cache", ())
    with pytest.raises(ValueError, match="namespace_exposed"):
        _destinations(tmp_path / "repo/candidate", tmp_path / "cache", (tmp_path / "repo",))


def test_pinned_launcher_environment_preserves_authenticated_values(monkeypatch):
    from scripts.github_source_preparation import delivery_environment

    monkeypatch.setenv("PYTHONPATH", "/untrusted/attempt")
    monkeypatch.setenv("PYTHONHOME", "/untrusted/home")
    host = SimpleNamespace(verification_plan={"environment": {"PYTHONPATH": "/immutable/runtime/src"}})
    env = delivery_environment(host)
    assert env["PYTHONPATH"] == "/immutable/runtime/src"
    assert "PYTHONHOME" not in env and "PYTHONPYCACHEPREFIX" not in env


def test_private_true_base_metadata_keeps_actual_commit_and_candidate(tmp_path):
    import time
    from scripts.github_source_preparation import _materialize
    from scripts.merge_verification import changed_paths
    from slm_training.autoresearch.heal.repair_source_workspace import _private_git
    from slm_training.harness_core.github_delivery_tree import base_entries, git_object

    repository = tmp_path / "repository"
    for name in ("root", "base"):
        directory = repository / name
        directory.mkdir(parents=True)
        (directory / "module.py").write_text("broken\n")
    # Existing verifier fixture machinery creates only private temporary metadata.
    base = _private_git(repository, time.monotonic() + 90, ())
    candidate = tmp_path / "accepted"
    candidate.mkdir()
    (candidate / "module.py").write_text("fixed\n")
    destination = tmp_path / "controller/candidate"
    _materialize(destination, repository / "root", candidate, base)
    assert (destination / "module.py").read_text() == "fixed\n"
    assert base_entries(destination, base) == {"module.py": ("100644", git_object("blob", b"broken\n"))}
    assert changed_paths(destination, base)[1] == ["module.py"]
    assert (repository / "root/module.py").read_text() == "broken\n"
