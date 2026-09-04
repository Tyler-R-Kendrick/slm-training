"""Ordering confirmation replays across completed candidates.

One responsibility: deciding which completed candidate is replayed next and
reconciling the replays already done. Per-candidate state lives in
``autotrain_candidate_state``.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from scripts.autotrain_candidate_state import (
    confirmation_quality_reheld,
)
from scripts.autotrain_io import read_json
from scripts.autotrain_levers import knobs_fingerprint, lever_knobs
from slm_training.autoresearch.schemas import (
    NextRunPriorityV1,
)


def confirmation_replay_entry(
    entries: list[dict[str, Any]], replay: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Return the original champion resolved by a frozen confirmation replay."""

    if replay is None or replay["handoff"].cycle_intent != "confirm":
        return None
    frozen = replay["candidate"]["experiment"]
    if not str(frozen.get("experiment_id") or "").endswith("-confirm"):
        return None
    fingerprint = knobs_fingerprint(lever_knobs(frozen.get("knobs") or {}))
    source_campaign = str(replay["handoff"].campaign_id)
    return next(
        (
            row
            for row in entries
            if row.get("knobs_fingerprint") == fingerprint
            and row.get("confirm_campaign_id") == source_campaign
            and row.get("status")
            in {"queued", "confirming", "confirmation_inconclusive"}
        ),
        None,
    )


def reconcile_completed_confirmation_replays(
    root: Path, entries: list[dict[str, Any]]
) -> bool:
    """Repair historical duplicate queue rows emitted by completed retries."""

    changed = False
    for duplicate in entries:
        if duplicate.get("status") != "queued" or not str(
            duplicate.get("source_candidate_id") or ""
        ).endswith("-confirm"):
            continue
        campaign_id = str(duplicate.get("source_campaign_id") or "")
        handoff_path = root / campaign_id / "cycle_handoff.json"
        delivery_path = root / campaign_id / "sdlc_delivery.json"
        if not handoff_path.is_file() or not delivery_path.is_file():
            continue
        handoff = read_json(handoff_path)
        delivery = read_json(delivery_path)
        if (
            handoff.get("cycle_intent") != "retry_measurement"
            or delivery.get("measurement_complete") is not True
        ):
            continue
        original = next(
            (
                row
                for row in entries
                if row is not duplicate
                and row.get("knobs_fingerprint") == duplicate.get("knobs_fingerprint")
                and row.get("status") == "confirmation_inconclusive"
            ),
            None,
        )
        if original is None:
            continue
        reasons = list(delivery.get("reasons") or [])
        confirmed = confirmation_quality_reheld(delivery)
        if not confirmed:
            reasons.append("confirmation_rejected:primary_quality_not_reheld")
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        original.update(
            status="confirmed" if confirmed else "rejected",
            confirm_campaign_id=campaign_id,
            confirm_cycle_index=handoff.get("cycle_index"),
            resolved_at=stamp,
            resolve_reasons=reasons,
        )
        duplicate.update(
            status="skipped_duplicate",
            resolved_at=stamp,
            resolve_reasons=[
                f"confirmation_replay_resolved:{original.get('entry_id')}"
            ],
        )
        changed = True
        print(
            "CHAMPION_REPLAY_RECONCILE "
            f"entry_id={original.get('entry_id')} status={original.get('status')} "
            f"duplicate={duplicate.get('entry_id')} campaign={campaign_id}",
            flush=True,
        )
    return changed


def consecutive_frozen_replays(
    root: Path, loop_id: str, campaign_id: str, cycle_intent: str
) -> int:
    """Count the current and immediately preceding frozen replay cycles."""

    if cycle_intent != "retry_measurement":
        return 0
    count = 1
    cursor = str(
        read_json(root / campaign_id / "campaign.json").get("predecessor_campaign_id")
        or ""
    )
    seen = {campaign_id}
    while cursor and cursor not in seen:
        seen.add(cursor)
        handoff = read_json(root / cursor / "cycle_handoff.json")
        if (
            handoff.get("loop_id") != loop_id
            or handoff.get("campaign_id") != cursor
            or handoff.get("cycle_intent") != "retry_measurement"
        ):
            break
        if any(
            action.get("kind") == "repair_harness"
            for action in handoff.get("actions") or []
            if isinstance(action, dict)
        ):
            break
        count += 1
        cursor = str(
            read_json(root / cursor / "campaign.json").get("predecessor_campaign_id")
            or ""
        )
    return count


def completed_confirmation_priorities(
    matrix: dict[str, Any],
    candidate_id: str,
    delivery: dict[str, Any],
    resolution: dict[str, Any] | None,
) -> tuple[NextRunPriorityV1, ...]:
    """Replace confirm-time hypotheses with observed successor steering."""

    rows = [dict(item) for item in matrix.get("next_run_priorities") or []]
    if not rows:
        return ()
    for row in rows:
        if row.get("disposition") == "experiment_next":
            row["disposition"] = "monitor"
            row["proposed_experiment_id"] = None

    status = str((resolution or {}).get("status") or "")
    confirmed = status == "confirmed"
    if confirmed:
        rows[0].update(
            {
                "area": "promotion",
                "hypothesis": (
                    "The champion re-held on confirmation; test the exact matched "
                    "recipes under the next promotion suite and Lean preflight."
                ),
                "confidence": 0.9,
                "expected_information_gain": (
                    "Separates a repeatable fixture signal from held-out and formal "
                    "promotion evidence."
                ),
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": candidate_id,
            }
        )
        return tuple(NextRunPriorityV1.model_validate(item) for item in rows)

    control = dict(delivery.get("control_metrics") or {})
    candidate = dict(delivery.get("candidate_metrics") or {})
    primary = str(delivery.get("primary_metric") or "primary metric")
    control_primary = control.get(primary)
    candidate_primary = candidate.get(primary)
    if control_primary is None and "." in primary:
        control_primary = control.get(primary.split(".", 1)[1])
        candidate_primary = candidate.get(primary.split(".", 1)[1])
    meaning_before = control.get("meaningful_program_rate")
    meaning_after = candidate.get("meaningful_program_rate")
    observed = (
        f"{primary} {control_primary!r}->{candidate_primary!r}; "
        f"meaningful_program_rate {meaning_before!r}->{meaning_after!r}"
    )
    future_batch_id = (
        candidate_id[: -len("-confirm")] + "-batch1"
        if candidate_id.endswith("-confirm")
        else f"{candidate_id}-batch1"
    )
    replacements = (
        {
            "area": "model",
            "hypothesis": (
                "Fresh-seed confirmation rejected the champion fingerprint "
                f"({observed}). Exhaust it and test a distinct size-matched "
                "quality-targeted objective instead of spending more scalar steps."
            ),
            "confidence": 0.95,
            "expected_information_gain": (
                "Tests objective alignment against certified structural quality "
                "rather than treating lower token loss as model progress."
            ),
            "authority": "observed_result",
            "disposition": "monitor",
            "proposed_experiment_id": None,
        },
        {
            "area": "evaluation",
            "hypothesis": (
                "Training loss and certified program quality diverged on the "
                "confirmation; retain loss as a diagnostic, not a promotion proxy."
            ),
            "confidence": 0.9,
            "expected_information_gain": (
                "Prevents optimization progress from masking regressions in meaning, "
                "structure, recall, or latency."
            ),
            "authority": "observed_result",
            "disposition": "monitor",
            "proposed_experiment_id": None,
        },
        {
            "area": "experiments",
            "hypothesis": (
                "Run the next non-exhausted batch-size arm only as a runtime "
                "diagnostic while a new quality-targeted objective is preregistered."
            ),
            "confidence": 0.65,
            "expected_information_gain": (
                "Keeps the bounded loop executable without mislabeling a throughput "
                "lever as a model-quality hypothesis."
            ),
            "authority": "speculative",
            "disposition": "experiment_next",
            "proposed_experiment_id": future_batch_id,
        },
        {
            "area": "infrastructure",
            "hypothesis": (
                "Exact source-control reconstruction and pre-execution attempt "
                "recovery succeeded; preserve both as campaign provenance."
            ),
            "confidence": 0.95,
            "expected_information_gain": (
                "Maintains attributable control-versus-candidate recipes across "
                "confirmation and promotion."
            ),
            "authority": "reproduced_harness_signal",
            "disposition": "monitor",
            "proposed_experiment_id": None,
        },
        {
            "area": "model_build",
            "hypothesis": (
                "Recent registered quality families are exhausted; prioritize a "
                "new preregistered structural or meaningful-quality objective before "
                "recycling them."
            ),
            "confidence": 0.75,
            "expected_information_gain": (
                "Tests a new causal training mechanism instead of repeating a closed "
                "approach or weakening a gate."
            ),
            "authority": "speculative",
            "disposition": "monitor",
            "proposed_experiment_id": None,
        },
    )
    for row, replacement in zip(rows, replacements, strict=False):
        row.update(replacement)
    return tuple(NextRunPriorityV1.model_validate(item) for item in rows)
