"""Rendering the documents a cycle owes.

One responsibility: writing the cycle's design-doc narrative and the five-lane
successor matrix -- the iron law's paperwork, emitted from the cycle's own
record rather than written by hand afterwards.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ExperimentCampaignV1,
)
from slm_training.autoresearch.schemas import (
    AutotrainCycleHandoffV1,
    utc_now,
)
from slm_training.harness_core.versioning import build_version_stamp

FIVE_LANES = (
    "measurement_control",
    "training_method",
    "architecture",
    "lean_model",
    "assumptions",
)


def render_continuous_cycle_docs(
    *,
    campaign_id: str,
    loop_id: str,
    handoff: AutotrainCycleHandoffV1,
    delivery: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Honest fixture-screening closeout payload (not a ship claim)."""
    reasons = list(delivery.get("reasons") or handoff.reasons or [])
    # Stamp the eval-comparability components so this record lands in a real
    # cross-version partition of the evidence ledger instead of "unstamped".
    # Without this, `eval_key_from_stamp` returns None and the record's delta is
    # pooled with every other unstamped cycle regardless of the scorer/gate
    # version it actually ran under. Never let stamping fail the closeout.
    try:
        from slm_training.autoresearch.evidence_ledger import EVAL_KEY_COMPONENTS

        version_stamp: dict[str, Any] | None = build_version_stamp(*EVAL_KEY_COMPONENTS)
    except Exception:
        version_stamp = None
    payload: dict[str, Any] = {
        "schema": "continuous_cycle_results/v1",
        "campaign_id": campaign_id,
        "loop_id": loop_id,
        "cycle_index": handoff.cycle_index,
        "cycle_role": handoff.cycle_role,
        "cycle_intent": handoff.cycle_intent,
        "positive": bool(delivery.get("positive")),
        "stack_layer": bool(delivery.get("stack_layer")),
        "measurement_complete": delivery.get("measurement_complete"),
        "primary_metric": handoff.primary_metric,
        "control_metrics": delivery.get("control_metrics"),
        "candidate_metrics": delivery.get("candidate_metrics"),
        "reasons": reasons,
        "evidence_class": handoff.evidence_class,
        "honesty": "fixture_screening_only_not_ship",
        "auto": True,
    }
    if version_stamp is not None:
        payload["version_stamp"] = version_stamp
    # Embed the rich delivery record (candidate_id/arm_seed/policy_sha256) so
    # future ledger mining never falls back to reasons-string recovery.
    if delivery.get("schema") == "autotrain_sdlc_delivery/v1":
        payload["delivery"] = dict(delivery)
    md = (
        f"# Continuous cycle `{campaign_id}`\n\n"
        f"- loop_id: `{loop_id}`\n"
        f"- cycle_index: `{handoff.cycle_index}`\n"
        f"- role/intent: `{handoff.cycle_role}` / `{handoff.cycle_intent}`\n"
        f"- primary_metric: `{handoff.primary_metric}`\n"
        f"- positive: **{payload['positive']}**\n"
        f"- stack_layer: **{payload['stack_layer']}**\n"
        f"- measurement_complete: `{payload['measurement_complete']}`\n"
        f"- evidence_class: `{handoff.evidence_class}`\n"
        f"- reasons: {', '.join(str(r) for r in reasons) or '—'}\n"
        f"- control_metrics: `{payload['control_metrics']}`\n"
        f"- candidate_metrics: `{payload['candidate_metrics']}`\n\n"
    )
    from slm_training.autoresearch.hillclimb import hillclimb_iteration_report

    hill = hillclimb_iteration_report(
        campaign_id=campaign_id,
        cycle_index=handoff.cycle_index,
        positive=bool(payload["positive"]),
        measurement_complete=payload.get("measurement_complete"),
        reasons=reasons,
        control_metrics=payload.get("control_metrics")
        if isinstance(payload.get("control_metrics"), dict)
        else None,
        candidate_metrics=payload.get("candidate_metrics")
        if isinstance(payload.get("candidate_metrics"), dict)
        else None,
        primary_metric=str(handoff.primary_metric or ""),
    )
    payload["hillclimb"] = hill
    md += (
        "## Hill-climb this cycle\n\n"
        f"- went well: {', '.join(hill['went_well']) or '—'}\n"
        f"- went wrong: {', '.join(hill['went_wrong']) or '—'}\n"
        f"- speculate: {', '.join(hill['speculate']) or '—'}\n"
        f"- deltas: `{hill.get('deltas')}`\n\n"
        "Auto-documented by the continuous driver self-heal closeout. "
        "Fixture screening only — not a ship claim.\n"
    )
    return md, payload


def build_five_lane_successor_matrix(
    *,
    campaign_id: str,
    entry: dict[str, Any],
    breaches: list[dict[str, Any]],
    cert_policy: str | None,
) -> dict[str, Any]:
    """Preregistered five-lane diagnosis matrix after assumption-backed miss."""
    lanes = list(FIVE_LANES)
    hypotheses = []
    for i, lane in enumerate(lanes, start=1):
        hypotheses.append(
            {
                "rank": i,
                "lane": lane,
                "hypothesis": (
                    f"Lane '{lane}' explains the assumption-backed band miss "
                    f"for champion {entry.get('knobs_fingerprint')} under "
                    f"cert_policy={cert_policy}."
                ),
                "falsification": (
                    "Controlled retest of this lane alone fails to move the "
                    "missed metric into the locked band."
                ),
                "breaches": breaches,
            }
        )
    return {
        "schema": "autotrain_five_lane_successor/v1",
        "campaign_id": campaign_id,
        "champion_entry_id": entry.get("entry_id"),
        "knobs_fingerprint": entry.get("knobs_fingerprint"),
        "cert_policy": cert_policy,
        "lanes": lanes,
        "hypotheses": hypotheses,
        "breaches": breaches,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_five_lane_successor(
    camp_dir: Path,
    *,
    campaign_id: str,
    entry: dict[str, Any],
    disposition: dict[str, Any],
) -> Path | None:
    if not disposition.get("emit_five_lane_matrix"):
        return None
    payload = build_five_lane_successor_matrix(
        campaign_id=campaign_id,
        entry=entry,
        breaches=list(disposition.get("breaches") or []),
        cert_policy=disposition.get("cert_policy"),
    )
    path = camp_dir / "five_lane_successor_matrix.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"FIVE_LANE_SUCCESSOR path={path}", flush=True)
    return path


def replay_successor_manifest(
    frozen: ExperimentCampaignV1,
    *,
    frozen_manifest_sha256: str,
    campaign_id: str,
    experiment_id: str,
    integration_commit: str,
) -> ExperimentCampaignV1:
    successor = frozen.model_copy(
        update={
            "campaign_id": campaign_id,
            "experiment_id": experiment_id,
            "source_commit": integration_commit,
            "source_dirty": False,
            "author": "autotrain-frozen-replay-successor",
            "created_at": utc_now(),
            "replay_of_manifest_sha256": frozen_manifest_sha256,
            "replay_reason": (
                "Current-main successor after an infrastructure-incomplete measurement."
            ),
            # A proof is commit- and experiment-bound. Never carry the source
            # campaign's proof digest into a current-main successor.
            "formal_obligations": (),
        }
    )
    return ExperimentCampaignV1.model_validate(successor.model_dump(mode="json"))
