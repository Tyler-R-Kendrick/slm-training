"""Supervisor hard-pending governance: dispatch, governed backoff, fail-soft."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

from tests.casefiles import case_values

from slm_training.autoresearch.heal.escalation import EscalationLedger

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_supervisor.py"
)
_SPEC = importlib.util.spec_from_file_location("run_autotrain_supervisor", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["git", "init"], cwd=path, env=_GIT_ENV, stdout=subprocess.DEVNULL
    )
    subprocess.check_call(
        ["git", "config", "user.email", "t@example.com"], cwd=path, env=_GIT_ENV
    )
    subprocess.check_call(
        ["git", "config", "user.name", "t"], cwd=path, env=_GIT_ENV
    )
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "-A"], cwd=path, env=_GIT_ENV)
    subprocess.check_call(
        ["git", "commit", "-m", "init"],
        cwd=path,
        env=_GIT_ENV,
        stdout=subprocess.DEVNULL,
    )
    return path


def test_handle_hard_pending_records_escalation_and_governs_backoff(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path / "repo")
    root = tmp_path / "ar"
    events: list[dict] = []
    blocker = {
        "campaign_id": "c1",
        "index": 0,
        "kind": "repair_formal",
        "reason": "theorem_backed_band_miss: parse_rate_pm",
        "blocker_class": "formal_contradiction",
    }
    outcome = _mod._handle_hard_pending(
        [blocker],
        cwd=repo,
        root=root,
        loop_id="loop-1",
        campaign_id="c1",
        max_heal_attempts=2,
        playbooks_enabled=True,
        log_event=events.append,
    )
    assert outcome["any_healed"] is False
    # Exact values pin the doubling contract (30 → 60), not just a floor.
    assert outcome["sleep_seconds"] == 30.0
    ledger = EscalationLedger.load(root, "loop-1")
    record = next(iter(ledger.records.values()))
    assert record.status == "escalated"
    assert record.owner_skill == "improve-lean-optimums"
    # Repeated sightings double the governed backoff (never a blind constant).
    second = _mod._handle_hard_pending(
        [blocker],
        cwd=repo,
        root=root,
        loop_id="loop-1",
        campaign_id="c2",
        max_heal_attempts=2,
        playbooks_enabled=True,
        log_event=events.append,
    )
    assert second["sleep_seconds"] == 60.0
    # Escalation may grow without bound, but supervision must keep retrying.
    third = _mod._handle_hard_pending(
        [blocker],
        cwd=repo,
        root=root,
        loop_id="loop-1",
        campaign_id="c3",
        max_heal_attempts=2,
        playbooks_enabled=True,
        log_event=events.append,
    )
    assert third["sleep_seconds"] == 60.0


def _stub_continuous_parked():
    """Continuous-module stub whose park predicate always says parked."""
    import types

    return types.SimpleNamespace(
        self_heal_unblock_loop=lambda **kwargs: {"soft_healed": []},
        _check_regime_parked=lambda **kwargs: "regime_parked",
    )


def _mock_park_operation(monkeypatch):
    from scripts import merge_verification_evidence as evidence

    monkeypatch.setattr(evidence, "source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(evidence, "environment_identity", lambda: {"fixture": True})

    def run_operation(runtime, request, **kwargs):
        assert request["operation"] == "inspect"
        return {"report": {"soft_healed": []}, "parked": "regime_parked", "campaign_id": None}

    monkeypatch.setattr(_mod, "_run_operation", run_operation)


def _park_events(root: Path, loop_id: str) -> list[str]:
    import json

    log = root / "loops" / loop_id / "supervisor.jsonl"
    return [
        json.loads(line)["event"]
        for line in log.read_text(encoding="utf-8").splitlines()
    ]


def test_park_is_wait_state_not_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Park must re-check (pre-cycle unblock can heal; fingerprint can move),
    # never end the process — exiting made the loop depend on an external
    # agent relaunch, the opposite of hands-off.
    monkeypatch.chdir(tmp_path)
    _mock_park_operation(monkeypatch)
    rc = _mod.main(
        [
            "--loop-id",
            "loop-1",
            "--root",
            str(tmp_path / "ar"),
            "--max-cycles",
            "2",
            "--park-backoff-seconds",
            "0.01",
        ]
    )
    assert rc == 0
    events = _park_events(tmp_path / "ar", "loop-1")
    assert events.count("regime_parked") == 2  # re-checked, did not exit
    assert "start_driver" not in events


def test_exit_on_park_preserves_legacy_single_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _mock_park_operation(monkeypatch)
    rc = _mod.main(
        [
            "--loop-id",
            "loop-1",
            "--root",
            str(tmp_path / "ar"),
            "--max-cycles",
            "5",
            "--exit-on-park",
        ]
    )
    assert rc == 0
    events = _park_events(tmp_path / "ar", "loop-1")
    assert events.count("regime_parked") == 1


def test_handle_hard_pending_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the heal layer itself to crash so the except branch is proven,
    # not merely reachable: a heal-layer bug must degrade, never propagate.
    import slm_training.autoresearch.heal as heal_pkg

    def _boom(**kwargs):
        raise RuntimeError("heal layer exploded")

    monkeypatch.setattr(heal_pkg, "run_playbooks", _boom)
    events: list[dict] = []
    outcome = _mod._handle_hard_pending(
        [{"kind": "repair_harness", "reason": "AgentV SDK is unavailable"}],
        cwd=tmp_path / "missing",
        root=tmp_path / "ar",
        loop_id="loop-1",
        campaign_id="c1",
        max_heal_attempts=2,
        playbooks_enabled=True,
        log_event=events.append,
    )
    assert outcome["any_healed"] is False
    assert outcome["sleep_seconds"] > 0
    assert any(e.get("event") == "hard_pending_heal_error" for e in events)


def test_handle_hard_pending_tolerates_hostile_blocker_shapes(
    tmp_path: Path,
) -> None:
    events: list[dict] = []
    outcome = _mod._handle_hard_pending(
        [{"kind": None, "reason": object()}],  # hostile blocker shape
        cwd=tmp_path / "missing",
        root=tmp_path / "ar",
        loop_id="loop-1",
        campaign_id="c1",
        max_heal_attempts=2,
        playbooks_enabled=True,
        log_event=events.append,
    )
    assert outcome["any_healed"] is False
    assert outcome["sleep_seconds"] > 0
    # Pin which path ran: either the heal layer degraded and logged, or it
    # completed and reported outcomes — a silent third path is a regression.
    assert (
        any(e.get("event") == "hard_pending_heal_error" for e in events)
        or "outcomes" in outcome
    )


def test_default_primary_metric_matches_climb_policy() -> None:
    from slm_training.autoresearch.climb_policy import load_climb_policy

    policy_metric = str(load_climb_policy().screening_primary["metric"])
    assert _mod._default_primary_metric() == policy_metric
    assert _mod._build_parser().parse_args([]).primary_metric == policy_metric


def test_canonical_driver_loader_registers_its_pinned_module():
    import sys

    driver = _mod._load_continuous()
    assert sys.modules[driver.__name__] is driver




@pytest.mark.parametrize("process_state,driver_code,expected", case_values(__file__, "test_operation_output_cannot_override_failed_execution"))
def test_operation_output_cannot_override_failed_execution(tmp_path, process_state, driver_code, expected):
    import json
    from scripts.autotrain_supervisor_operations import interpret_operation_result
    from scripts.merge_verification_evidence import digest
    from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome

    request = {"operation": "driver"}
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"schema_version": "supervisor_operation/v1",
        "request_digest": digest(request), "operation": "driver",
        "payload": {"returncode": driver_code, "campaign_id": "fixture"}}))
    result = BoundedProcessResult((), ProcessOutcome(process_state), 0, "", "", .01,
        interrupted=process_state == "interrupted", killed=process_state == "killed")
    outcome, payload = interpret_operation_result(result, output, request)
    assert outcome.value == expected
    assert (payload is not None) == (expected == "succeeded")


@pytest.mark.parametrize("missing_output", [False, True])
@pytest.mark.parametrize("process_state", ["completed", "interrupted", "killed"])
def test_supervisor_operation_uses_real_activity_contract(tmp_path, monkeypatch, missing_output, process_state):
    from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome
    from types import SimpleNamespace
    from scripts import merge_verification_evidence as evidence
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.autoresearch.storage import CampaignStore

    request = {"cwd": str(tmp_path), "root": str(tmp_path / "data"),
               "loop_id": "fixture", "operation": "inspect",
               "source_digest": "a" * 64, "environment_digest": evidence.digest({"fixture_environment": True})}
    monkeypatch.setattr(evidence, "source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(evidence, "environment_identity", lambda: {"fixture_environment": True})
    monkeypatch.setattr(_mod, "_load_continuous", lambda: SimpleNamespace(
        self_heal_unblock_loop=lambda **_: {"hard_pending": []},
        _check_regime_parked=lambda **_: None,
        _latest_cycle=lambda *_: (0, None)))
    journal = CampaignStore("controller", tmp_path / "events")
    with ActivityRuntime(journal) as runtime:
        launches = []

        def run(lease, argv, **kwargs):
            launches.append(lease.attempt_id)
            if not missing_output:
                from scripts import autotrain_supervisor_operations as operations
                from slm_training.harness_core.checkpoint_publication import champion_publication_scope
                # In-process fixture retains real fencing; separate child test proves delegation.
                monkeypatch.setattr(operations, "operation_publication_scope", lambda req, root, loop:
                                    champion_publication_scope(runtime, lease, loop_dir=root / "loops" / loop))
                _mod._operation_main(Path(argv[-3]), Path(argv[-1]))
            return BoundedProcessResult(tuple(argv), ProcessOutcome(process_state), 0, "", "", .01)

        monkeypatch.setattr(runtime, "run", run)
        result = _mod._run_operation(runtime, request, sequence=1, log_event=lambda _: None)
        states = list(runtime.snapshot().values())
        failed = missing_output or process_state != "completed"
        assert len(states) == 1
        assert states[0].status == ("waiting_repair" if failed else "succeeded")
        assert (result is None) == failed
        from slm_training.autoresearch.heal.operation_recovery import pending_operation_repairs
        assert len(pending_operation_repairs(runtime)) == int(failed)
        repeated = _mod._run_operation(runtime, request,
            sequence=2 if missing_output else 1, log_event=lambda _: None)
        assert repeated == result
        assert len(launches) == 1  # No fresh budget or repeated committed effect.
        assert len(runtime.snapshot()) == 1


@pytest.mark.parametrize("stale_handoff", [False, True])
@pytest.mark.parametrize("driver_code", [0, 2])
def test_operation_driver_zero_without_handoff_is_failure(tmp_path, monkeypatch, stale_handoff, driver_code):
    import json
    from types import SimpleNamespace
    from scripts import merge_verification_evidence as evidence
    from contextlib import nullcontext
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.harness_core.activity_contract import ActivitySpec
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core import checkpoint_publication as champion_publication

    # This unit check isolates handoff validation; real delegation has its own
    # process tests and is not established by this mocked context manager.
    monkeypatch.setattr(champion_publication, "champion_publication_scope", lambda *a, **k: nullcontext())
    with ActivityRuntime(CampaignStore("fixture-runtime", tmp_path)) as runtime:
        runtime.register(ActivitySpec(activity_id="driver", family="fixture", kind="control",
            source_digest="a" * 64, environment_digest="b" * 64, input_digest="c" * 64,
            output_namespace="attempts/driver"))
        lease = runtime.claim_next(capabilities={"local_process"})

    monkeypatch.setattr(evidence, "source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(evidence, "environment_identity", lambda: {"fixture_environment": True})
    monkeypatch.setattr(_mod, "_load_continuous", lambda: SimpleNamespace(
        main=lambda _: driver_code, _latest_cycle=lambda *_: (1, "missing")))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"cwd": str(tmp_path), "root": str(tmp_path),
        "source_digest": "a" * 64, "operation": "driver", "loop_id": "fixture",
        "environment_digest": evidence.digest({"fixture_environment": True}),
        "driver_argv": [], "lease": lease.model_dump(mode="json")}))
    if stale_handoff:
        (tmp_path / "missing").mkdir()
        (tmp_path / "missing" / "cycle_handoff.json").write_text("{}")
    expected = "same-campaign completion lacks one current retirement" if stale_handoff else "without its required handoff"
    if driver_code:
        expected = "driver operation returned unsuccessful status"
    with pytest.raises(ValueError, match=expected):
        _mod._operation_main(request, tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


def test_watchdog_no_campaign_governs_backoff_and_keeps_counting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ar"
    events: list[dict] = []
    common = dict(root=root, loop_id="loop-1", campaign_id="c1", log_event=events.append)
    assert (
        _mod._watchdog_no_campaign(
            passes_without_campaign=4, total_no_campaign_passes=4, **common
        )
        is None
    )
    assert not EscalationLedger.path_for(root, "loop-1").exists()
    backoffs = [
        _mod._watchdog_no_campaign(
            passes_without_campaign=passes,
            total_no_campaign_passes=passes,
            **common,
        )
        for passes in (5, 6, 7)
    ]
    # Ledger governor (30s doubling): the counter is never reset, so the same
    # fingerprint is re-observed and the backoff grows with the stall.
    assert backoffs == [30.0, 60.0, 120.0]
    ledger = EscalationLedger.load(root, "loop-1")
    record = next(iter(ledger.records.values()))
    assert record.kind == "loop_stalled_no_campaign"
    assert record.seen_count == 3
    assert record.status == "escalated"
    assert "total_no_campaign_passes=7" in record.note
    assert [e["passes"] for e in events] == [5, 6, 7]
    assert [e["next_backoff_seconds"] for e in events] == backoffs
