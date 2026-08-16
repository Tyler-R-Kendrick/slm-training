"""Supervisor hard-pending governance: dispatch, governed backoff, fail-soft."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from slm_training.autoresearch.heal.escalation import EscalationLedger

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_supervisor.py"
)
_SPEC = importlib.util.spec_from_file_location("run_autotrain_supervisor", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "init"], cwd=path, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "t@example.com"], cwd=path)
    subprocess.check_call(["git", "config", "user.name", "t"], cwd=path)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "-A"], cwd=path)
    subprocess.check_call(
        ["git", "commit", "-m", "init"], cwd=path, stdout=subprocess.DEVNULL
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
    assert outcome["sleep_seconds"] >= 30.0
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
    assert second["sleep_seconds"] >= 60.0


def test_handle_hard_pending_never_raises(tmp_path: Path) -> None:
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
