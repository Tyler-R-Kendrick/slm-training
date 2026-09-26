"""Identity drift preserves the saved proposal and charges verifier use once."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

from scripts import autotrain_verification as owner
from scripts.merge_verification_evidence import digest
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import (
    ActivityOutcome, ActivitySpec, ResourceGrant, WakeCondition,
)
from slm_training.autoresearch.heal.repair_acceptance import source_verification_activity_id


def _dependency(tmp_path, monkeypatch):
    root = tmp_path / "candidate"
    root.mkdir()
    (tmp_path / "runtime").mkdir()
    binding = {"candidate_tree_sha256": "a" * 64, "environment": {},
               "runtime_identity": owner.runtime_identity((tmp_path / "runtime",)), "static_commands": [],
               "targets": ["tests"], "isolation_enforced": True}
    monkeypatch.setattr(owner, "verification_binding", lambda *a, **kw: binding)
    grant = ResourceGrant(interrupt_seconds=65, total_seconds=200, max_attempts=3)
    identity = digest(binding)
    dependency = {
        "schema_version": "repair_verification_dependency/v1",
        "activity_id": source_verification_activity_id(identity, grant),
        "root": str(root), "state_dir": str(tmp_path / "private-cache"),
        "runtime_roots": [str(tmp_path / "runtime")], "base_ref": "frozen-base",
        "verification_identity": identity, "request_digest": "c" * 64,
        "proposal_digest": "d" * 64, "candidate_snapshot_digest": "e" * 64,
        "grant": grant.model_dump(mode="json"),
        "wake": {"predicate": "complete_current_source_verification",
                 "source": "source_verification_completed", "identity_digest": identity},
    }
    (tmp_path / "manifest.json").write_text(json.dumps({
        "verification_identity": identity, "request_digest": dependency["request_digest"],
        "proposal_digest": dependency["proposal_digest"],
        "candidate_snapshot_digest": dependency["candidate_snapshot_digest"],
        "binding": binding,
    }))
    dependency["manifest_path"] = str(tmp_path / "manifest.json")
    return dependency, binding


def _park(runtime, dependency):
    runtime.register(ActivitySpec(
        activity_id="repair", family="repair", kind="repair",
        source_digest="a" * 64, environment_digest="b" * 64,
        input_digest="c" * 64, output_namespace="attempts/repair",
    ))
    lease = runtime.claim_next(capabilities={"local_process"}, activity_id="repair")
    artifact = runtime.store.write_artifact("source_verification_requests", dependency)
    event = runtime.store.append_event(
        "source_verification_requested", experiment_id="repair",
        artifact_sha256=artifact.stem,
        detail={"dependency_digest": artifact.stem, "repair_activity_id": "repair"},
    )
    runtime.finish(lease, outcome=ActivityOutcome.DEPENDENCY, spent_seconds=0,
                   wake=WakeCondition.model_validate(dependency["wake"]))
    owner.register_dependency(runtime, owner.dependency_plan(dependency))
    return event


def _materializer(tmp_path, gate, request_digest, proposal_digest):
    def materialize(_, predecessor, grant):
        journal = CampaignStore("repair", tmp_path / "repair-root")
        destination = journal.root / "source_verification" / gate.identity
        (destination / "root").mkdir(parents=True)
        gate.root, gate.state_dir = destination / "root", destination / "cache"
        (destination / "manifest.json").write_text(json.dumps({
            "verification_identity": gate.identity,
            "binding": gate.binding,
        }))
        artifact = journal.write_artifact("repair_source_verification_inputs", {
            "request_digest": request_digest, "proposal_digest": proposal_digest,
            "verification_identity": gate.identity,
        })
        journal.append_event("repair_source_verification_prepared", artifact_sha256=artifact.stem)
        config = SimpleNamespace(source_verification_grant=grant,
                                 runtime_roots=tuple(predecessor["runtime_roots"]))
        return journal, SimpleNamespace(digest=lambda: request_digest), \
            SimpleNamespace(digest=lambda: proposal_digest), config, gate, \
            tuple(predecessor["runtime_roots"])
    return materialize


def _next_dependency(previous, gate, grant, roots):
    identity = gate.identity
    return {
        **previous,
        "activity_id": source_verification_activity_id(identity, grant),
        "root": str(gate.root), "state_dir": str(gate.state_dir),
        "manifest_path": str(gate.root.parent / "manifest.json"),
        "verification_identity": identity, "grant": grant.model_dump(mode="json"),
        "runtime_roots": list(roots),
        "wake": {"predicate": "complete_current_source_verification",
                 "source": "source_verification_completed", "identity_digest": identity},
    }


def _chain(runtime, tmp_path, monkeypatch, hops, rollback=False):
    from scripts import autotrain_verification_successor as successor_owner
    from slm_training.autoresearch.heal import recovery_dispatch

    predecessor, binding = _dependency(tmp_path, monkeypatch)
    event = _park(runtime, predecessor)
    chain = [(event, predecessor)]
    for hop in range(hops):
        current = binding if rollback and hop == hops - 1 else {
            **binding, "environment": {"PATH": str(hop)}}
        gate = SimpleNamespace(identity=digest(current), binding=current)
        monkeypatch.setattr(successor_owner, "_materialize", _materializer(
            tmp_path, gate, predecessor["request_digest"], predecessor["proposal_digest"],
        ))
        monkeypatch.setattr(recovery_dispatch, "_verification_dependency",
                            lambda request, proposal, gate, grant, roots:
                            _next_dependency(predecessor, gate, grant, roots))
        lease = runtime.claim_next(capabilities={"local_process", "isolated_verifier"},
                                   activity_id=predecessor["activity_id"])
        runtime.finish(lease, outcome=ActivityOutcome.YIELDED, spent_seconds=3)
        spent = runtime.snapshot()[predecessor["activity_id"]]
        monkeypatch.setattr(owner, "verification_binding", lambda *a, **kw: current)
        successor_owner.plan_successor(runtime, event, predecessor, {
            "repair_config": "/approved/recovery.json", "repair_config_digest": "1" * 64,
            "loop_id": "loop",
        })
        event = next(row for row in reversed(runtime.store.verify_event_chain())
                     if row["event_type"] == "source_verification_requested")
        successor = owner.load_dependency(runtime.store, event)
        assert successor["grant"]["total_seconds"] == spent.spec.grant.total_seconds - spent.charged_seconds
        assert successor["grant"]["max_attempts"] == spent.spec.grant.max_attempts - spent.attempts
        assert runtime.snapshot()[predecessor["activity_id"]].status == "cancelled"
        chain.append((event, successor))
        predecessor = successor
    return chain


@pytest.mark.parametrize("hops,rollback", [(1, False), (2, False), (2, True)])
def test_successor_uses_remaining_grant_and_wakes_with_predecessor_identity(
    tmp_path, monkeypatch, hops, rollback,
):
    with ActivityRuntime(CampaignStore("runtime", tmp_path / "loops" / "loop")) as runtime:
        chain = _chain(runtime, tmp_path, monkeypatch, hops, rollback)
        successor_event, successor = chain[-1]
        if rollback:
            assert successor["verification_identity"] == chain[0][1]["verification_identity"]
            assert successor_event["detail"]["dependency_digest"] != chain[0][0]["detail"]["dependency_digest"]
            assert successor["activity_id"] != chain[0][1]["activity_id"]
        plan = owner.dependency_plan(successor)
        ancestors = {dep["activity_id"]: runtime.snapshot()[dep["activity_id"]]
                     for _, dep in chain[:-1]}
        from slm_training.autoresearch.heal import isolation
        observed = []
        events = []
        monkeypatch.setattr(isolation, "probe_isolation",
                            lambda: SimpleNamespace(available=True))
        def execute_successor(next_runtime, next_plan, lease):
            observed.append(next_plan["identity"])
            result = next_runtime.run(
                lease, [sys.executable, "-c", "print('latest-verifier')"],
                cwd=tmp_path, env={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            assert result.returncode == 0 and result.stdout == "latest-verifier\n"
            next_runtime.finish(lease, outcome=ActivityOutcome.YIELDED, spent_seconds=1)
            return {"status": "pending"}

        monkeypatch.setattr(owner, "execute_release_attempt", execute_successor)
        common = {"repair_config": "/approved/recovery.json", "repair_config_digest": "1" * 64,
                  "loop_id": "loop"}
        assert owner.drain_source_verification(runtime, common, events.append) == {"status": "pending"}
        assert observed == [successor["verification_identity"]]
        assert events[0]["event"] == "source_verification_pass"
        assert runtime.snapshot()[successor["activity_id"]].attempts == 1
        assert runtime.snapshot()[successor["activity_id"]].charged_seconds == 1
        assert all(runtime.snapshot()[key] == state for key, state in ancestors.items())
        owner.drain_source_verification(runtime, common, events.append)
        assert observed == [successor["verification_identity"]]  # Retry timer preserved.

        monkeypatch.setattr(owner, "authenticated_completion", lambda _: True)
        verify_chain = runtime.store.verify_event_chain

        def forged_activation():
            rows = verify_chain()
            for row in rows:
                if row["event_type"] == "source_verification_successor_activated":
                    row["detail"] = {**row["detail"], "predecessor_identity": "f" * 64}
            return rows

        monkeypatch.setattr(runtime.store, "verify_event_chain", forged_activation)
        assert not owner.wake_repair(runtime, successor_event, successor, plan)
        monkeypatch.setattr(runtime.store, "verify_event_chain", verify_chain)
        assert owner.wake_repair(runtime, successor_event, successor, plan)
        assert runtime.snapshot()["repair"].status == "runnable"
        completed = [row for row in runtime.store.verify_event_chain()
                     if row["event_type"] == "source_verification_completed"][-1]
        assert completed["detail"]["verification_identity"] == successor["verification_identity"]


def test_successor_cannot_wake_for_changed_proposal(tmp_path, monkeypatch):
    from scripts import autotrain_verification_successor as successor_owner
    from slm_training.autoresearch.heal import recovery_dispatch

    predecessor, old_binding = _dependency(tmp_path, monkeypatch)
    current = {**old_binding, "environment": {"PATH": "current"}}
    gate = SimpleNamespace(root=Path(predecessor["root"]),
                           state_dir=Path(predecessor["state_dir"]),
                           base_ref=predecessor["base_ref"], identity=digest(current),
                           binding=current)
    monkeypatch.setattr(successor_owner, "_materialize", _materializer(
        tmp_path, gate, predecessor["request_digest"], predecessor["proposal_digest"],
    ))
    monkeypatch.setattr(recovery_dispatch, "_verification_dependency",
                        lambda request, proposal, gate, grant, roots:
                        _next_dependency(predecessor, gate, grant, roots))
    with ActivityRuntime(CampaignStore("runtime", tmp_path / "loops" / "loop")) as runtime:
        event = _park(runtime, predecessor)
        monkeypatch.setattr(owner, "verification_binding", lambda *a, **kw: current)
        successor_owner.plan_successor(runtime, event, predecessor, {
            "repair_config": "/approved/recovery.json", "repair_config_digest": "1" * 64,
            "loop_id": "loop",
        })
        successor_event = next(row for row in runtime.store.verify_event_chain()
                               if row["event_type"] == "source_verification_requested"
                               and row["detail"]["dependency_digest"] != event["detail"]["dependency_digest"])
        successor = owner.load_dependency(runtime.store, successor_event)
        plan = owner.dependency_plan(successor)
        monkeypatch.setattr(owner, "authenticated_completion", lambda _: True)
        assert not owner.wake_repair(runtime, successor_event,
                                     {**successor, "proposal_digest": "f" * 64}, plan)
        assert runtime.snapshot()["repair"].status == "waiting_dependency"


def _activate(runtime, predecessor_event, predecessor, successor):
    from scripts.autotrain_verification_successor import _activation

    artifact = runtime.store.write_artifact("source_verification_requests", successor)
    runtime.store.append_event(
        "source_verification_requested", experiment_id="repair", artifact_sha256=artifact.stem,
        detail={"dependency_digest": artifact.stem, "repair_activity_id": "repair"},
    )
    _activation(runtime, predecessor_event, predecessor,
                SimpleNamespace(digest=lambda: predecessor["request_digest"]),
                SimpleNamespace(digest=lambda: predecessor["proposal_digest"]), successor)


@pytest.mark.parametrize("fault", case_values(__file__, "test_successor_chain_fails_closed_before_claim"))
def test_successor_chain_fails_closed_before_claim(tmp_path, monkeypatch, fault):
    with ActivityRuntime(CampaignStore("runtime", tmp_path / "loops" / "loop")) as runtime:
        chain = _chain(runtime, tmp_path, monkeypatch, 2)
        event, latest = chain[-1]
        if fault == "cycle":
            _activate(runtime, event, latest, chain[0][1])
        elif fault == "ambiguous":
            _activate(runtime, *chain[0], latest)
        elif fault == "divergent":
            orphan = {**event, "detail": {**event["detail"], "dependency_digest": "f" * 64}}
            _activate(runtime, orphan, latest, latest)
        elif fault in {"runtime", "environment"}:
            binding = owner.verification_binding()
            field = "runtime_identity" if fault == "runtime" else "environment"
            monkeypatch.setattr(owner, "verification_binding",
                                lambda *a, **kw: {**binding, field: "changed"})
        else:
            successor = {**latest, "verification_identity": "f" * 64,
                         "wake": {**latest["wake"], "identity_digest": "f" * 64}}
            successor[fault] = ({**latest["grant"], "total_seconds": 999}
                                if fault == "grant" else "f" * 64)
            successor["activity_id"] = source_verification_activity_id(
                successor["verification_identity"], successor["grant"])
            _activate(runtime, event, latest, successor)
        states = runtime.snapshot()
        observations = []
        monkeypatch.setattr(owner, "execute_release_attempt",
                            lambda *a: pytest.fail("invalid successor executed"))
        assert owner.drain_source_verification(runtime, {}, observations.append) is None
        assert observations and all(row["event"] == "source_verification_wait" for row in observations)
        assert runtime.snapshot() == states  # No claims, charges, grants, or wake changes.
        monkeypatch.setattr(owner, "authenticated_completion", lambda _: True)
        if fault not in {"runtime", "environment"}:
            assert not owner.wake_repair(runtime, event, latest, owner.dependency_plan(latest))
