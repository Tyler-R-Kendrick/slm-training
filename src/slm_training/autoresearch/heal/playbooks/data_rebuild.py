"""Restore the frozen data predicate; global line-count growth is not healing.

Legacy count states remain historical. This playbook never acknowledges
campaign actions, changes gates, or replaces a sealed confirmation suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from slm_training.autoresearch.heal.classify import DATA_PREREQUISITE_MARKERS
from slm_training.autoresearch.heal.escalation import blocker_fingerprint
from slm_training.autoresearch.heal.schemas import (
    HealAttemptReceiptV1, HealPlanV1, HealStepResultV1, HealStepV1, HealVerifyV1,
)
from slm_training.autoresearch.schemas import utc_now
from slm_training.data.readiness_contract import DataReadinessRequest, SnapshotInput
from slm_training.harness_core.lineage.store import _atomic_write
from slm_training.harnesses.train_data.readiness import check_readiness, snapshot_input
from slm_training.levers import INTERRUPT_AFTER_SECONDS

PLAYBOOK_ID = "data_rebuild/v1"
STATE_DIRNAME = "heal_data_rebuild"
RebuildSeam = Callable[..., object]


def default_rebuild_seam(**kwargs):
    from slm_training.harness_core.bounded_process import ProcessOutcome, run_bounded_process
    from slm_training.levers import KILL_GRACE_SECONDS

    # Import/model stalls remain bounded; request data cannot select a code release.
    source = Path(__file__).resolve().parents[5]
    argv = [sys.executable, "-m", "scripts.autotrain_data_builder", "--cwd",
            str(kwargs["cwd"]), "--request-payload", kwargs["request"].model_dump_json()]
    result = run_bounded_process(
        argv, cwd=str(source), interrupt_after_seconds=float(INTERRUPT_AFTER_SECONDS) - 15,
        kill_grace_seconds=float(KILL_GRACE_SECONDS))
    if result.outcome != ProcessOutcome.COMPLETED or result.returncode != 0:
        raise RuntimeError(f"data builder incomplete:{result.outcome}:{result.returncode}")


def state_path(root: Path, loop_id: str, fingerprint: str) -> Path:
    return Path(root) / "loops" / loop_id / STATE_DIRNAME / f"{fingerprint[:16]}.json"


def _text_result(step_id: str, returncode: int, text: str) -> HealStepResultV1:
    return HealStepResultV1(
        step_id=step_id, returncode=returncode, outcome="completed",
        stdout_sha256=hashlib.sha256(text.encode()).hexdigest(),
        stderr_sha256=hashlib.sha256(b"").hexdigest(), tail=text[:2000])


def load_readiness_request(blocker: dict, *, root: Path, campaign_id: str):
    """Resolve committed controller input, or an explicitly supplied legacy input.

    No hand-created latest-request file is required by the supervised path.
    Multiple distinct requests require an exact selector, never a latest guess.
    """
    payload = blocker.get("data_readiness_request")
    if payload is None:
        payload = _committed_request(blocker, root=root, campaign_id=campaign_id)
    request = DataReadinessRequest.model_validate(payload)
    if request.campaign_id != campaign_id:
        raise ValueError("wrong campaign data readiness request")
    if blocker.get("data_action_id") not in (None, request.action_id):
        raise ValueError("wrong data readiness action")
    wanted = blocker.get("data_readiness_request_sha256")
    if wanted is not None and wanted != request.sha256:
        raise ValueError("wrong data readiness request identity")
    return request


def _committed_request(blocker, *, root, campaign_id):
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.data.readiness_contract import load_locked_readiness_request

    return load_locked_readiness_request(CampaignStore(campaign_id, root),
        wanted=blocker.get("data_readiness_request_sha256"),
        action_id=blocker.get("data_action_id")).model_dump(mode="json")


def _attempt(request, *, cwd, root, loop_id, campaign_id, seam):
    before = check_readiness(request, root=cwd)
    seam(cwd=cwd, root=root, loop_id=loop_id, campaign_id=campaign_id, request=request)
    # Ignore claimed results; resolve only the locked publication destination.
    destination = cwd / "outputs" / "data" / request.original.kind / request.successor_id
    candidate = snapshot_input(cwd, destination, exposure=request.original.exposure)
    after = check_readiness(request, root=cwd, candidate=candidate)
    return before, after, candidate


def execute(
    blocker: dict, *, cwd: Path, root: Path, loop_id: str, campaign_id: str,
    seam: RebuildSeam | None = None, count_records=None, write_receipt: bool = True,
) -> HealAttemptReceiptV1:
    """Execute a remedy and rerun its original predicate; never trust a counter.

    ``count_records`` is accepted for caller compatibility but is never called.
    Missing legacy contracts refuse before invoking a builder: no guessed role,
    dataset, suite, ancestry or volume requirement is permitted.
    """
    from slm_training.autoresearch.heal import write_heal_receipt

    cwd, root = Path(cwd), Path(root)
    campaign_id = str(blocker.get("campaign_id") or campaign_id)
    fingerprint = blocker_fingerprint(str(blocker.get("kind") or "rebuild_data"),
                                      str(blocker.get("reason") or ""), data_request=blocker.get("data_readiness_request"))
    request, before, after, candidate = None, None, None, None
    outcome, note = "postcondition_failed", "missing_data_readiness_contract"
    steps = []
    try:
        request = load_readiness_request(blocker, root=root, campaign_id=campaign_id)
    except (OSError, TypeError, ValueError) as exc:
        note = f"missing_data_readiness_contract:{type(exc).__name__}:{exc}"[:400]
    else:
        try:
            before, after, candidate = _attempt(
                request, cwd=cwd, root=root, loop_id=loop_id, campaign_id=campaign_id,
                seam=seam or default_rebuild_seam)
            outcome = "healed" if after["ready"] else "postcondition_failed"
            note = "original_predicate_restored" if after["ready"] else (
                f"heal_postcondition_failed:{after['errors']}")[:400]
            steps.append(_text_result("rebuild_seam", 0, note))
        except Exception as exc:  # noqa: BLE001 — failed builders cannot authorize healing
            outcome = "step_failed"
            note = f"rebuild_failed:{type(exc).__name__}:{exc}"[:400]
            steps.append(_text_result("rebuild_seam", 1, note))
    verify = _text_result("original_data_predicate", 0 if outcome == "healed" else 1, note)
    payload = {"schema_version": "heal_data_rebuild_state/v2", "loop_id": loop_id,
               "campaign_id": campaign_id, "blocker_fingerprint": fingerprint,
               "request": request.model_dump() if request else None,
               "request_sha256": request.sha256 if request else None,
               "before": before, "after": after,
               "candidate": candidate.model_dump() if candidate else None,
               "outcome": outcome, "note": note, "recorded_at": utc_now()}
    _atomic_write(state_path(root, loop_id, fingerprint), payload)
    receipt = HealAttemptReceiptV1(
        loop_id=loop_id, campaign_id=campaign_id, playbook_id=PLAYBOOK_ID,
        plan_sha256=request.sha256 if request else hashlib.sha256(note.encode()).hexdigest(),
        blocker_fingerprint=fingerprint, attempts_prior=int(blocker.get("_attempts_prior") or 0),
        step_results=tuple(steps), verify_result=verify, outcome=outcome,
        note=note, recorded_at=utc_now())
    if write_receipt:
        write_heal_receipt(root, receipt)
    return receipt


class _DataRebuildPlaybook:
    playbook_id = PLAYBOOK_ID
    handles = frozenset({"data"})

    def matches(self, blocker: dict) -> bool:
        if blocker.get("kind") == "rebuild_data":
            return True
        return any(m in str(blocker.get("reason") or "").lower()
                   for m in DATA_PREREQUISITE_MARKERS)

    def execute(self, blocker, **kwargs):
        return execute(blocker, **kwargs)

    def plan(self, blocker: dict, *, cwd: Path) -> HealPlanV1 | None:
        root = blocker.get("_root")
        loop_id = str(blocker.get("_loop_id") or "")
        campaign_id = str(blocker.get("campaign_id") or "")
        if root is None or not loop_id or not campaign_id:
            return None
        try:
            request = load_readiness_request(blocker, root=Path(root), campaign_id=campaign_id)
        except (OSError, TypeError, ValueError):
            return None
        fingerprint = blocker_fingerprint(str(blocker.get("kind") or "rebuild_data"),
                                          str(blocker.get("reason") or ""), data_request=blocker.get("data_readiness_request"))
        state = state_path(Path(root), loop_id, fingerprint)
        from slm_training.autoresearch.heal.playbooks.harness_crash import _writes_allowed_for

        writes = _writes_allowed_for(state.parent, cwd=Path(cwd), extra=(
            "outputs/runs/**", f"outputs/data/{request.original.kind}/{request.successor_id}/**",
            f"outputs/data/{request.original.kind}/.publication.lock"))
        module = "slm_training.autoresearch.heal.playbooks.data_rebuild"
        common = ("--root", str(root), "--loop-id", loop_id, "--campaign-id", campaign_id,
                  "--reason", str(blocker.get("reason") or ""),
                  "--kind", str(blocker.get("kind") or "rebuild_data"), "--cwd", str(cwd))
        return HealPlanV1(
            playbook_id=self.playbook_id, blocker_fingerprint=fingerprint, blocker_class="data",
            steps=(HealStepV1(step_id="rebuild_data_seam",
                             argv=(sys.executable, "-m", module, *common, "--no-receipt",
                                   "--request-payload", request.model_dump_json()), cwd="",
                             timeout_seconds=int(INTERRUPT_AFTER_SECONDS), writes_allowed=writes),),
            verify=HealVerifyV1(argv=(sys.executable, "-m", module, "--verify-state", str(state),
                                     "--request-sha", request.sha256, "--cwd", str(cwd)),
                                cwd="", timeout_seconds=60))


PLAYBOOK = _DataRebuildPlaybook()


def _verify_state(path: Path, *, cwd: Path, request_sha: str) -> int:
    try:
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != "heal_data_rebuild_state/v2" or not request_sha:
            return 1
        request = DataReadinessRequest.model_validate(payload["request"])
        if request.sha256 != request_sha or payload["campaign_id"] != request.campaign_id:
            return 1
        candidate = SnapshotInput.model_validate(payload["candidate"])
        return 0 if check_readiness(request, root=cwd, candidate=candidate)["ready"] else 1
    except (OSError, KeyError, TypeError, ValueError):
        return 1


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-state", type=Path)
    parser.add_argument("--request-sha", default="")
    parser.add_argument("--request-payload")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--loop-id", default="")
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--kind", default="rebuild_data")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--no-receipt", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_state is not None:
        return _verify_state(args.verify_state, cwd=args.cwd, request_sha=args.request_sha)
    if args.root is None or not args.loop_id or not args.campaign_id:
        parser.error("--root, --loop-id and --campaign-id are required")
    blocker = {"kind": args.kind, "reason": args.reason, "campaign_id": args.campaign_id}
    if args.request_payload:
        blocker["data_readiness_request"] = json.loads(args.request_payload)
    receipt = execute(blocker, cwd=args.cwd, root=args.root, loop_id=args.loop_id,
                      campaign_id=args.campaign_id, write_receipt=not args.no_receipt)
    print(receipt.model_dump_json())
    return 0 if receipt.outcome == "healed" else 1


if __name__ == "__main__":
    sys.exit(_main())
