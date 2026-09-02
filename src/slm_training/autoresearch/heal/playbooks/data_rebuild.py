"""Data-class playbook: the existing rebuild seam under a measured postcondition.

``rebuild_data`` blockers already have a local heal in the continuous driver
(``_self_heal_rebuild_data`` → ``scripts.build_train_data`` for train data, or
the screening-suite deficit path built on
``screening_sample_size.extra_smoke_fixtures_for_deficit``). What was missing
is an executor for the ``data`` blocker class in the heal layer, and a
postcondition that turns "the rebuild command exited 0" into evidence: this
playbook counts data records before and after invoking the seam and writes

- ``healed`` only when ``records_after > records_before`` (the count grew);
- ``postcondition_failed`` (note ``heal_postcondition_failed:...``) when the
  seam returned but the count did not grow;
- ``step_failed`` when the seam raised.

The seam is never reimplemented here: the default calls the driver's public
``_self_heal_rebuild_data`` entrypoint through the same module loader the
supervisor uses, and tests inject a stub seam.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from slm_training.autoresearch.heal.classify import DATA_PREREQUISITE_MARKERS
from slm_training.autoresearch.heal.escalation import blocker_fingerprint
from slm_training.autoresearch.heal.schemas import (
    HealAttemptReceiptV1,
    HealPlanV1,
    HealStepResultV1,
    HealStepV1,
    HealVerifyV1,
)
from slm_training.autoresearch.schemas import utc_now
from slm_training.levers import MAX_RUN_SECONDS

__all__ = [
    "PLAYBOOK",
    "PLAYBOOK_ID",
    "RebuildSeam",
    "RecordCounter",
    "count_data_records",
    "default_rebuild_seam",
    "execute",
]

PLAYBOOK_ID = "data_rebuild/v1"

STATE_DIRNAME = "heal_data_rebuild"

#: ``seam(cwd=..., root=..., loop_id=..., campaign_id=...) -> object``.
RebuildSeam = Callable[..., object]
#: ``count(cwd, root, campaign_id) -> int``.
RecordCounter = Callable[[Path, Path, str], int]

_DATA_DIRS: tuple[str, ...] = ("outputs/data/train", "outputs/data/test")
_SEED_FILE = "src/slm_training/resources/test_seeds.jsonl"


def _count_jsonl_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return sum(1 for line in stream if line.strip())
    except OSError:
        return 0


def count_data_records(cwd: Path, root: Path, campaign_id: str) -> int:
    """Total data records the rebuild seam can grow.

    Sums every ``*.jsonl`` under the local train/test data stores plus the
    tracked smoke seed file (the screening-suite deficit path appends there).
    A rebuild that allocates a new immutable version, or appends seeds,
    strictly grows this number; a no-op leaves it unchanged.
    """
    cwd = Path(cwd)
    total = 0
    for rel in _DATA_DIRS:
        base = cwd / rel
        if not base.is_dir():
            continue
        for path in base.rglob("*.jsonl"):
            if path.is_file():
                total += _count_jsonl_lines(path)
    seed = cwd / _SEED_FILE
    if seed.is_file():
        total += _count_jsonl_lines(seed)
    return total


def _load_driver_module():
    script = (
        Path(__file__).resolve().parents[5] / "scripts" / "run_autotrain_continuous.py"
    )
    if not script.is_file():
        # Installed layouts: fall back to the working directory's copy.
        script = Path.cwd() / "scripts" / "run_autotrain_continuous.py"
    spec = importlib.util.spec_from_file_location("run_autotrain_continuous", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load continuous driver from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_rebuild_seam(
    *, cwd: Path, root: Path, loop_id: str, campaign_id: str
) -> object:
    """Invoke the driver's ``_self_heal_rebuild_data`` (the existing seam)."""
    driver = _load_driver_module()
    return driver._self_heal_rebuild_data(
        cwd=Path(cwd), root=Path(root), loop_id=loop_id, campaign_id=campaign_id
    )


def state_path(root: Path, loop_id: str, fingerprint: str) -> Path:
    return Path(root) / "loops" / loop_id / STATE_DIRNAME / f"{fingerprint[:16]}.json"


def _text_result(step_id: str, returncode: int, text: str) -> HealStepResultV1:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    empty = hashlib.sha256(b"").hexdigest()
    return HealStepResultV1(
        step_id=step_id,
        returncode=returncode,
        outcome="completed",
        stdout_sha256=digest,
        stderr_sha256=empty,
        tail=text[:2000],
    )


def execute(
    blocker: dict,
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str,
    seam: RebuildSeam | None = None,
    count_records: RecordCounter | None = None,
    write_receipt: bool = True,
) -> HealAttemptReceiptV1:
    """Run the rebuild seam once; ``healed`` iff the record count grew."""
    from slm_training.autoresearch.heal import write_heal_receipt

    cwd = Path(cwd)
    root = Path(root)
    kind = str(blocker.get("kind") or "rebuild_data")
    reason = str(blocker.get("reason") or "")
    fingerprint = blocker_fingerprint(kind, reason)
    blocker_campaign = str(blocker.get("campaign_id") or campaign_id)
    seam = seam or default_rebuild_seam
    counter = count_records or count_data_records

    before = int(counter(cwd, root, blocker_campaign))
    steps: list[HealStepResultV1] = []
    outcome = "healed"
    note = ""
    seam_result: object = None
    try:
        seam_result = seam(
            cwd=cwd, root=root, loop_id=loop_id, campaign_id=blocker_campaign
        )
    except Exception as exc:  # noqa: BLE001 — seam crash is a failed attempt
        outcome = "step_failed"
        note = f"rebuild seam raised {type(exc).__name__}: {exc}"[:400]
        steps.append(_text_result("rebuild_seam", 1, note))
    else:
        steps.append(
            _text_result("rebuild_seam", 0, f"seam_result={seam_result!r}"[:2000])
        )
    after = int(counter(cwd, root, blocker_campaign))
    verify = _text_result(
        "records_grew",
        0 if after > before else 1,
        f"records_before={before} records_after={after}",
    )
    if outcome == "healed" and not after > before:
        outcome = "postcondition_failed"
        note = (
            f"heal_postcondition_failed:records_after>records_before "
            f"records_before={before} records_after={after} "
            f"seam_result={seam_result!r}"
        )[:400]
    elif outcome == "healed":
        note = f"records_before={before} records_after={after} seam_result={seam_result!r}"[
            :400
        ]
    payload = {
        "schema_version": "heal_data_rebuild_state/v1",
        "loop_id": loop_id,
        "campaign_id": blocker_campaign,
        "blocker_fingerprint": fingerprint,
        "recorded_at": utc_now(),
        "records_before": before,
        "records_after": after,
        "outcome": outcome,
        "note": note,
    }
    out = state_path(root, loop_id, fingerprint)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = HealAttemptReceiptV1(
        loop_id=loop_id,
        campaign_id=blocker_campaign,
        playbook_id=PLAYBOOK_ID,
        plan_sha256=hashlib.sha256(
            json.dumps(
                {"playbook": PLAYBOOK_ID, "fingerprint": fingerprint, "before": before},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        blocker_fingerprint=fingerprint,
        attempts_prior=int(blocker.get("_attempts_prior") or 0),
        step_results=tuple(steps),
        verify_result=verify,
        outcome=outcome,  # type: ignore[arg-type]
        note=note,
        recorded_at=utc_now(),
    )
    if write_receipt:
        write_heal_receipt(root, receipt)
    return receipt


class _DataRebuildPlaybook:
    playbook_id = PLAYBOOK_ID
    handles = frozenset({"data"})

    def matches(self, blocker: dict) -> bool:
        kind = str(blocker.get("kind") or "")
        if kind == "rebuild_data":
            return True
        reason = str(blocker.get("reason") or "").lower()
        return any(m in reason for m in DATA_PREREQUISITE_MARKERS)

    def execute(
        self,
        blocker: dict,
        *,
        cwd: Path,
        root: Path,
        loop_id: str,
        campaign_id: str,
        write_receipt: bool = True,
    ) -> HealAttemptReceiptV1:
        return execute(
            blocker,
            cwd=cwd,
            root=root,
            loop_id=loop_id,
            campaign_id=campaign_id,
            write_receipt=write_receipt,
        )

    def plan(self, blocker: dict, *, cwd: Path) -> HealPlanV1 | None:
        """Runner-compatible plan: seam in a subprocess, count-growth verify."""
        root = blocker.get("_root")
        loop_id = str(blocker.get("_loop_id") or "")
        campaign_id = str(blocker.get("campaign_id") or "")
        if root is None or not loop_id or not campaign_id:
            return None
        root = Path(root)
        kind = str(blocker.get("kind") or "")
        reason = str(blocker.get("reason") or "")
        fingerprint = blocker_fingerprint(kind, reason)
        state = state_path(root, loop_id, fingerprint)
        from slm_training.autoresearch.heal.playbooks.harness_crash import (
            _writes_allowed_for,
        )

        # The seam writes the rebuilt data under outputs/ and copies receipt
        # artifacts into root/<campaign>/; the state file lands next to the
        # heal ledger. Anything else it dirties is a scope violation.
        writes_allowed = _writes_allowed_for(
            state.parent,
            cwd=Path(cwd),
            extra=_writes_allowed_for(root / campaign_id, cwd=Path(cwd)),
        )
        module = "slm_training.autoresearch.heal.playbooks.data_rebuild"
        common = (
            "--root",
            str(root),
            "--loop-id",
            loop_id,
            "--campaign-id",
            campaign_id,
            "--reason",
            reason,
            "--kind",
            kind or "rebuild_data",
        )
        return HealPlanV1(
            playbook_id=self.playbook_id,
            blocker_fingerprint=fingerprint,
            blocker_class="data",
            steps=(
                HealStepV1(
                    step_id="rebuild_data_seam",
                    argv=(sys.executable, "-m", module, *common, "--no-receipt"),
                    cwd="",
                    timeout_seconds=int(MAX_RUN_SECONDS),
                    writes_allowed=writes_allowed,
                ),
            ),
            verify=HealVerifyV1(
                argv=(
                    sys.executable,
                    "-m",
                    module,
                    "--verify-state",
                    str(state),
                ),
                cwd="",
                timeout_seconds=60,
            ),
        )


PLAYBOOK = _DataRebuildPlaybook()


def _verify_state(path: Path) -> int:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"data_rebuild: missing state {path}")
        return 2
    before = int(payload.get("records_before") or 0)
    after = int(payload.get("records_after") or 0)
    print(f"records_before={before} records_after={after}")
    return 0 if after > before else 1


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-state", type=Path, default=None)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--loop-id", default="")
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--kind", default="rebuild_data")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--no-receipt", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_state is not None:
        return _verify_state(args.verify_state)
    if args.root is None or not args.loop_id or not args.campaign_id:
        parser.error("--root, --loop-id and --campaign-id are required")
    receipt = execute(
        {"kind": args.kind, "reason": args.reason, "campaign_id": args.campaign_id},
        cwd=args.cwd,
        root=args.root,
        loop_id=args.loop_id,
        campaign_id=args.campaign_id,
        write_receipt=not args.no_receipt,
    )
    print(receipt.model_dump_json())
    return 0 if receipt.outcome == "healed" else 1


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(_main())
