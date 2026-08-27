"""Supervisor hard-pending governance: dispatch, governed backoff, fail-soft."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

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


def test_iteration_summary_renders_after_child_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    campaign_id = "campaign-7"
    campaign = tmp_path / campaign_id
    campaign.mkdir()
    (campaign / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "cycle_index": 7,
                "cycle_intent": "screening",
                "measurement_complete": True,
                "positive": False,
                "control_id": "control",
                "candidate_id": "candidate",
                "arm_exits": {"control": 0, "candidate": 0},
                "control_metrics": {"eval_nll": 5.0, "parse_rate": 1.0},
                "candidate_metrics": {"eval_nll": 4.0, "parse_rate": 1.0},
                "reasons": ["primary_metric_win"],
            }
        ),
        encoding="utf-8",
    )

    _mod._print_iteration_summary(tmp_path, campaign_id)

    output = capsys.readouterr().out
    assert "AUTOTRAIN ITERATION SUMMARY | cycle=7" in output
    assert "| eval_nll | 5 | 4 | -1 |" in output
    assert "arms | control=0 candidate=0" in output
    assert output.rstrip().endswith("decision | primary_metric_win")


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
    stale = EscalationLedger.load(root, "loop-1")
    for _ in range(8):
        stale.observe(
            kind="repair_harness",
            reason="historical unrelated blocker",
            blocker_class="code",
            campaign_id="old",
        )
    stale.save()
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
    # Exact values pin current-blocker doubling (30 → 60); the unrelated
    # historical record at the one-hour cap must not govern this dispatch.
    assert outcome["sleep_seconds"] == 30.0
    ledger = EscalationLedger.load(root, "loop-1")
    record = next(r for r in ledger.records.values() if r.kind == "repair_formal")
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


def test_hard_backoff_wakes_when_typed_self_heal_clears_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ar"
    state_path = root / "loops" / "loop-1" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"blocker_count": 1, "next_action": "hard_pending"}),
        encoding="utf-8",
    )
    sleeps: list[float] = []

    def clear_during_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        state_path.write_text(
            json.dumps(
                {
                    "blocker_count": 0,
                    "next_action": "continue_after_self_heal:unblock:document_closeout",
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(_mod.time, "sleep", clear_during_sleep)

    assert _mod._sleep_hard_backoff(
        root=root, loop_id="loop-1", seconds=3600
    )
    assert sleeps == [5.0]


def _stub_continuous_parked():
    """Continuous-module stub whose park predicate always says parked."""
    import types

    return types.SimpleNamespace(
        self_heal_unblock_loop=lambda **kwargs: {"soft_healed": []},
        _check_regime_parked=lambda **kwargs: "regime_parked",
    )


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
    monkeypatch.setattr(_mod, "_load_continuous", _stub_continuous_parked)
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
    monkeypatch.setattr(_mod, "_load_continuous", _stub_continuous_parked)
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
