"""Running the formal-lane preflight before a promotion.

One responsibility: deciding whether the formal lane is required for this
cycle, how much budget it may take, and driving the preflight to a recorded
outcome. The claim records themselves live in ``autotrain_formal_claims``.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import time

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.autotrain_budget import remaining_timeout
from scripts.autotrain_formal_claims import (
    PROMOTE_FORMAL_TEMPLATE_ID,
    promote_formal_claim_dict,
)
from scripts.autotrain_io import read_json
from scripts.autotrain_stage import stage_process_callbacks
from scripts.autotrain_timeout_state import formal_status_is_timeout
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    MAX_RUN_SECONDS,
)

PROMOTE_FORMAL_TIMEOUT_S = float(MAX_RUN_SECONDS)


def formal_lane_required(*, cycle_intent: str, replay: dict[str, Any] | None) -> bool:
    """Reserve formal time only for work whose locked plan requires it."""

    if cycle_intent == "promote":
        return True
    if replay is None:
        return False
    return any(
        bool(replay[arm]["manifest"].formal_obligations)
        for arm in ("control", "candidate")
    )


def post_formal_arm_budget_request(
    *,
    policy_minutes: float,
    initial_arm_wall_minutes: float,
    formal_completed: bool,
) -> float:
    """Return unused formal-lane time to the matched decision arms."""

    if formal_completed:
        return float(policy_minutes)
    return float(initial_arm_wall_minutes)


def promotion_formal_budget_seconds(
    *, deadline: float, arm_count: int, arm_wall_minutes: float
) -> float:
    """Return only wall time left after reserving complete arms + finalization."""

    reserved = arm_count * arm_wall_minutes * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
    available = remaining_timeout(deadline) - reserved
    if available <= 0:
        raise subprocess.TimeoutExpired("promotion formal preflight budget", reserved)
    return min(float(PROMOTE_FORMAL_TIMEOUT_S), available)


def ensure_promote_formal_preflight(
    *,
    camp_dir: Path,
    campaign_id: str,
    experiment_id: str,
    run_lean: bool = False,
    timeout_seconds: float = PROMOTE_FORMAL_TIMEOUT_S,
    root: Path | None = None,
    loop_id: str | None = None,
) -> tuple[str, str | None]:
    """Record formal preflight; return ``(status, content_sha256|None)``.

    When proved, writes a content-addressed artifact under
    ``artifacts/formal_preflights/<sha>.json`` matching
    ``validate_formal_preflights`` binding. Fail closed on any error.
    """
    from slm_training.autoresearch.schemas import FormalClaimV1

    claim = FormalClaimV1(**promote_formal_claim_dict())
    status_path = camp_dir / "formal_preflight_status.json"
    if status_path.is_file():
        data = read_json(status_path)
        status = str(data.get("status") or "missing")
        sha = data.get("preflight_sha256")
        if status == "proved" and sha:
            artifact = camp_dir / "artifacts" / "formal_preflights" / f"{sha}.json"
            try:
                from slm_training.autoresearch.formal import (
                    validate_formal_preflight_artifact,
                )

                validated = validate_formal_preflight_artifact(
                    artifact,
                    campaign_id=campaign_id,
                    experiment_id=experiment_id,
                    claim=claim,
                    expected_sha256=str(sha),
                )
                if validated.status != "proved":
                    raise ValueError(
                        f"required cached formal status is {validated.status!r}"
                    )
            except Exception as exc:  # noqa: BLE001 - stale cache fails closed
                if not run_lean:
                    record_formal_preflight_status(
                        camp_dir,
                        status="unknown",
                        template_id=PROMOTE_FORMAL_TEMPLATE_ID,
                        reason=f"cached_formal_preflight_invalid:{exc}",
                    )
                    return "unknown", None
            else:
                data["binding_validated_sha256"] = str(sha)
                status_path.write_text(
                    json.dumps(data, indent=2) + "\n", encoding="utf-8"
                )
                return "proved", str(sha)
        elif not run_lean:
            return status, str(sha) if sha else None

    if not run_lean:
        record_formal_preflight_status(
            camp_dir,
            status="missing",
            template_id=PROMOTE_FORMAL_TEMPLATE_ID,
            reason="formal_preflight_not_run",
        )
        return "missing", None

    try:
        from slm_training.lineage.records import canonical_json

        from slm_training.autoresearch.formal import (
            formal_preflight_payload,
            run_formal_preflight,
            validate_formal_preflight_artifact,
        )
        from slm_training.autoresearch.schemas import (
            ExperimentKnobs,
            ExperimentSpec,
        )

        exp = ExperimentSpec(
            experiment_id=experiment_id,
            campaign_id=campaign_id,
            hypothesis="Continuous promote requires proved structural-similarity mono.",
            rationale="Proof driver: required formal preflight before promote train.",
            expected_effect="Block promote when formal status is not proved.",
            falsification_criteria=("Required formal preflight is not proved.",),
            stop_conditions=("Stop before promote train when formal preflight fails.",),
            citations=("docs/design/formal-autoresearch.md",),
            knobs=ExperimentKnobs(steps=1),
            formal_claims=(claim,),
        )
        # The caller can further tighten the repository-wide wall.
        on_start, on_heartbeat = stage_process_callbacks(
            root=root, loop_id=loop_id, stage="promotion-formal-preflight"
        )
        preflight, _obligation = run_formal_preflight(
            campaign_id,
            exp,
            claim,
            timeout_seconds=timeout_seconds,
            on_start=on_start,
            on_heartbeat=on_heartbeat,
        )
        status = str(preflight.status)
        duration = float(getattr(preflight, "duration_seconds", 0.0) or 0.0)
        payload = formal_preflight_payload(preflight)
        content_sha = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
        art = camp_dir / "artifacts" / "formal_preflights"
        art.mkdir(parents=True, exist_ok=True)
        # Content-addressed name required by validate_formal_preflights.
        out = art / f"{content_sha}.json"
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        timed_out = formal_status_is_timeout(status)
        if timed_out:
            reason = (
                f"formal_preflight_timed_out:wall_s={timeout_seconds:g}:"
                f"duration_s={duration:.3f}"
            )
        elif status == "proved":
            reason = None
        else:
            reason = f"preflight_status={status}"
        record_formal_preflight_status(
            camp_dir,
            status=status,
            template_id=PROMOTE_FORMAL_TEMPLATE_ID,
            reason=reason,
            timeout_seconds=timeout_seconds,
            duration_seconds=duration,
            timed_out=timed_out,
        )
        # Persist sha for manifest binding.
        status_path = camp_dir / "formal_preflight_status.json"
        st = read_json(status_path)
        st["preflight_sha256"] = content_sha
        if status == "proved":
            validate_formal_preflight_artifact(
                out,
                campaign_id=campaign_id,
                experiment_id=experiment_id,
                claim=claim,
                expected_sha256=content_sha,
            )
            st["binding_validated_sha256"] = content_sha
        status_path.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
        return status, content_sha
    except Exception as exc:  # noqa: BLE001 — fail closed for non-timeout errors
        msg = str(exc)
        timed_out = "timed out" in msg.lower() or "timeout" in msg.lower()
        status = "timed_out" if timed_out else "unknown"
        record_formal_preflight_status(
            camp_dir,
            status=status,
            template_id=PROMOTE_FORMAL_TEMPLATE_ID,
            reason=(
                f"formal_preflight_timed_out:wall_s={timeout_seconds:g}:{msg}"
                if timed_out
                else f"formal_preflight_error:{exc}"
            ),
            timeout_seconds=timeout_seconds,
            timed_out=timed_out,
        )
        return status, None


def record_formal_preflight_status(
    camp_dir: Path,
    *,
    status: str,
    template_id: str,
    reason: str | None = None,
    timeout_seconds: float | None = None,
    duration_seconds: float | None = None,
    timed_out: bool | None = None,
) -> Path:
    camp_dir.mkdir(parents=True, exist_ok=True)
    path = camp_dir / "formal_preflight_status.json"
    payload: dict[str, Any] = {
        "schema": "autotrain_formal_preflight_status/v1",
        "status": status,
        "template_id": template_id,
        "reason": reason,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if timeout_seconds is not None:
        payload["timeout_seconds"] = float(timeout_seconds)
    if duration_seconds is not None:
        payload["duration_seconds"] = float(duration_seconds)
    if timed_out is not None:
        payload["timed_out"] = bool(timed_out)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
