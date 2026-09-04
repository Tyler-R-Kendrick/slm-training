"""Parking the loop when it cannot honestly proceed.

One responsibility: the park decisions -- a screening sample-size deficit, a
stalled loop, and a terminal park once the thrash bank is exhausted. Parking is
how the loop refuses to manufacture a result it did not measure.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.autotrain_records import write_loop_state
from slm_training.autoresearch.schemas import (
    AutotrainActionV1,
    AutotrainCycleHandoffV1,
    AutotrainLoopStateV1,
    utc_now,
)


def terminal_park_on_exhaust(policy: Any | None = None) -> bool:
    """Policy-gated bank-exhaust parking (``terminal`` block; defaults off)."""
    try:
        from slm_training.autoresearch.climb_policy import load_climb_policy

        pol = policy if policy is not None else load_climb_policy()
        payload = getattr(pol, "payload", None) or {}
        block = payload.get("terminal") if isinstance(payload, Mapping) else None
        return bool(isinstance(block, Mapping) and block.get("park_on_exhaust"))
    except Exception:  # noqa: BLE001 — parking never blocks the legacy path
        return False


def park_screening_n_deficit(
    *,
    root: Path,
    loop_id: str,
    campaign_id: str,
    cycle_index: int,
    report: dict[str, Any],
) -> str:
    """Skip screening; queue the action the binding constraint actually needs.

    The report distinguishes two causes of an empty range and only one of them
    is a data deficit. ``suite_volume`` means the published suite is smaller
    than the decidability floor, and generating records clears it. But
    ``wall_budget`` means the arm wall affords fewer records than the floor at
    the declared decode cost, and no amount of generated data clears that: a
    ``rebuild_data`` ask there is an action nobody can discharge -- the
    synthesis owner publishes records, the range stays empty, and the next
    cycle parks again having spent real work. The report's own suggestion
    names that remedy (cheaper per-record decode or a larger stage share,
    never a silent wall++), which is harness territory. Both can bind at once.
    """

    handoff_path = root / campaign_id / "cycle_handoff.json"
    if not handoff_path.is_file():
        raise RuntimeError("screening n deficit without a typed predecessor handoff")
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    evidence_ids = (f"campaign:{campaign_id}",)
    binding = tuple(report.get("binding_constraints") or ())
    n_min = report.get("n_min") or 6
    remedies: list[AutotrainActionV1] = []
    if "suite_volume" in binding or report.get("must_generate") or not binding:
        # No binding constraint recorded means the cause is unknown; the
        # historical ask stays, so an unclassified deficit never parks silently.
        remedies.append(
            AutotrainActionV1(
                kind="rebuild_data",
                owner="synthesis-feedback",
                reason=(
                    "screening suite_volume binds "
                    f"(suite_ceiling_n={report.get('suite_ceiling_n')}): generate "
                    f"and persist smoke n>={n_min} instead of screening at an "
                    "undecidable n"
                ),
                evidence_ids=evidence_ids,
            )
        )
    wall_budget_binds = "wall_budget" in binding
    if wall_budget_binds:
        remedies.append(
            AutotrainActionV1(
                kind="repair_harness",
                owner="improve-openui-harnesses",
                harness_family="model_build",
                reason=(
                    "screening wall_budget binds: the arm wall affords "
                    f"{report.get('budget_ceiling_n')} records at the declared "
                    f"decode floor and the decidability floor is {n_min}; "
                    "cheaper per-record decode or a larger stage share, never "
                    "a silent wall++ (generating records cannot clear this)"
                ),
                evidence_ids=evidence_ids,
            )
        )
    if not remedies:
        # A constraint neither branch recognizes (a third one added later).
        # The old code always queued a data ask, so falling back to it keeps
        # the park from silently requesting nothing at all; the reason names
        # the constraint so the owner can see it was not understood here.
        remedies.append(
            AutotrainActionV1(
                kind="rebuild_data",
                owner="synthesis-feedback",
                reason=(
                    "screening range is empty under an unrecognized binding "
                    f"constraint ({', '.join(binding)}); n_min={n_min}, "
                    f"suite_ceiling_n={report.get('suite_ceiling_n')}, "
                    f"budget_ceiling_n={report.get('budget_ceiling_n')}"
                ),
                evidence_ids=evidence_ids,
            )
        )
    actions = (
        *remedies,
        AutotrainActionV1(
            kind="next_experiment",
            owner="autotrain",
            reason=(
                "resume screening only once the binding constraint clears: "
                f"{', '.join(binding) or 'cause unrecorded'}"
            ),
            evidence_ids=evidence_ids,
        ),
    )
    handoff_path.write_text(
        handoff.model_copy(update={"actions": actions}).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    write_loop_state(
        root,
        AutotrainLoopStateV1(
            loop_id=loop_id,
            state="BLOCKED",
            phase="blocked",
            active_campaign_id=None,
            last_completed_campaign_id=campaign_id,
            cycle_index=cycle_index,
            next_action="rebuild_data",
            blocker_fingerprint="screening_n_suite_volume",
            blocker_count=1,
            pid=os.getpid(),
        ),
    )
    print(
        f"SCREENING_N_PARK loop={loop_id} campaign={campaign_id} "
        f"must_generate={report.get('must_generate')} "
        f"binding={report.get('binding_constraints')}",
        flush=True,
    )
    return "screening-n-deficit"


STALL_FINGERPRINT = "loop_stalled_no_campaign"


def park_loop_stalled(
    *,
    root: Path,
    loop_id: str,
    cycle_index: int,
    campaign_id: str | None,
    consecutive: int,
    last_non_vacuous: dict[str, Any] | None,
    reason: str | None,
) -> Path:
    """Write the typed ``loop_stalled_no_campaign`` park (state=BLOCKED)."""
    if last_non_vacuous:
        last = (
            f"last non-vacuous pass: {last_non_vacuous.get('outcome')} "
            f"campaign={last_non_vacuous.get('campaign_after')} "
            f"at {last_non_vacuous.get('recorded_at')}"
        )
    else:
        last = "no non-vacuous pass recorded for this loop"
    if reason:
        last = f"{last}; last cycle error: {reason}"
    next_action = (
        f"{STALL_FINGERPRINT}: {consecutive} consecutive vacuous passes "
        f"(no campaign, no verified heal, no typed action); {last}"
    )[:1000]
    path = write_loop_state(
        root,
        AutotrainLoopStateV1(
            loop_id=loop_id,
            state="BLOCKED",
            phase="blocked",
            last_completed_campaign_id=campaign_id,
            cycle_index=max(0, int(cycle_index)),
            next_action=next_action,
            blocker_fingerprint=STALL_FINGERPRINT,
            blocker_count=int(consecutive),
            pid=os.getpid(),
            heartbeat_at=utc_now(),
        ),
    )
    try:
        from slm_training.autoresearch.heal.escalation import EscalationLedger

        ledger = EscalationLedger.load(root, loop_id)
        record = ledger.observe(
            kind=STALL_FINGERPRINT,
            reason="consecutive vacuous driver passes without a new campaign",
            blocker_class="unknown",
            campaign_id=campaign_id or "unknown",
            owner_skill="autotrain",
        )
        ledger.escalate(record.fingerprint, note=next_action[:400])
        ledger.save()
    except Exception as exc:  # noqa: BLE001 — ledger bugs never mask the park
        print(f"LOOP_PARKED_LEDGER_WARN {exc!r}", flush=True)
    print(
        f"LOOP_PARKED fingerprint={STALL_FINGERPRINT} consecutive={consecutive} "
        f"state={path}",
        flush=True,
    )
    return path
