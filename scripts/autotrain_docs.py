"""Rendering the documents a cycle owes.

One responsibility: writing the cycle's design-doc narrative and the five-lane
successor matrix -- the iron law's paperwork, emitted from the cycle's own
record rather than written by hand afterwards.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
import hashlib
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

FIVE_LANES = (
    "measurement_control",
    "training_method",
    "architecture",
    "lean_model",
    "assumptions",
)


def with_evidence_ledger(cwd: Path, files: dict[str, str]) -> dict[str, str]:
    """Bind the derived ledger required by the same documentation successor gate."""
    from slm_training.autoresearch.evidence_ledger import DEFAULT_LEDGER_PATH, build_ledger

    replacements = {cwd / name: text for name, text in files.items()
                    if name.startswith("docs/design/") and name.endswith(".json")}
    ledger = build_ledger(cwd / "docs/design", replacements=replacements)
    relative = DEFAULT_LEDGER_PATH.relative_to(Path(__file__).resolve().parents[1])
    return {**files, relative.as_posix(): json.dumps(ledger, indent=2, sort_keys=True) + "\n"}


class MeasurementProvenanceUnavailable(ValueError):
    """Documentation prerequisite missing; never a model-quality verdict."""


def _measurement_stamp(stamp):
    from slm_training.autoresearch.evidence_ledger import EVAL_KEY_COMPONENTS

    if (not isinstance(stamp, dict) or stamp.get("stamp_schema") != "version_stamp/v1"
            or not isinstance(stamp.get("components"), dict)
            or not stamp.get("stamped_at") or stamp.get("code_dirty") is not False
            or not isinstance(stamp.get("code_commit"), str)
            or len(stamp["code_commit"]) != 40
            or any(c not in "0123456789abcdef" for c in stamp["code_commit"])
            or any(not isinstance(stamp["components"].get(key), str)
                   or stamp["components"][key] in {"", "unknown", "UNKNOWN"}
                   for key in EVAL_KEY_COMPONENTS if key != "gates.ship")):
        raise MeasurementProvenanceUnavailable("measurement_stamp_missing_or_invalid")
    return json.loads(json.dumps(stamp))


def _arm_measurement_stamp(store, events, arm):
    from scripts.autotrain_cycle_lock import load_context, read_artifact
    from slm_training.autoresearch.engine import _expected_gate_rejection

    if not isinstance(arm, str) or Path(arm).name != arm or arm in {".", ".."}:
        raise MeasurementProvenanceUnavailable("measurement_arm_identity_invalid")
    finishes = [row for row in events if row["event_type"] == "experiment_finished"
                and row.get("experiment_id") == arm]
    if not finishes:
        raise MeasurementProvenanceUnavailable("measurement_finish_missing")
    event = finishes[-1]
    if any(row["event_type"] == "experiment_started" and row.get("experiment_id") == arm
           for row in events[events.index(event) + 1:]):
        raise MeasurementProvenanceUnavailable("measurement_newer_attempt_unfinished")
    outcome = read_artifact(store, "outcomes", event["artifact_sha256"])
    manifest = store.load_experiment_campaign(arm).manifest_sha256
    if (outcome.get("campaign_id") != store.campaign_id or outcome.get("experiment_id") != arm
            or outcome.get("campaign_manifest_sha256") != manifest
            or event.get("detail", {}).get("campaign_manifest_sha256") != manifest):
        raise MeasurementProvenanceUnavailable("measurement_outcome_identity_mismatch")
    path = store.root / "runs" / arm / "scoreboard.json"
    if path.is_symlink():
        raise MeasurementProvenanceUnavailable("measurement_scoreboard_symlink")
    scoreboard = json.loads(path.read_text())
    if scoreboard.get("measurement_complete") is not True or not scoreboard.get("suites"):
        raise MeasurementProvenanceUnavailable("measurement_scoreboard_incomplete")
    stages = [row["parsed_output"] for row in outcome.get("stage_telemetry", [])
              if "scripts.evaluate_model" in row.get("command", [])
              and isinstance(row.get("parsed_output"), dict)
              and row["parsed_output"].get("measurement_complete") is True
              and type(row.get("exit_code")) is int
              and (row["exit_code"] == 0 or _expected_gate_rejection(
                  row["command"], row["exit_code"], row["parsed_output"], artifact_root=Path(load_context(store)["cwd"])))]
    fields = ("version_stamp", "suites", "checkpoint_sha256", "eval_data_manifest_sha")
    if not any(all(stage.get(key) == scoreboard.get(key) for key in fields) for stage in stages):
        raise MeasurementProvenanceUnavailable("measurement_stage_scoreboard_mismatch")
    return _measurement_stamp(scoreboard.get("version_stamp"))


def _workload_gate_version(store, result, delivery):
    from scripts.autotrain_cycle_lock import load_context
    from slm_training.harness_core.execution_release import _runtime_manifest
    from slm_training.autoresearch.evidence_ledger import EVAL_KEY_COMPONENTS

    locked = load_context(store)
    if locked is None:
        raise MeasurementProvenanceUnavailable("measurement_workload_lock_missing")
    if locked["order"] != delivery["arm_order"] or locked["loop_id"] != delivery["loop_id"]:
        raise MeasurementProvenanceUnavailable("measurement_workload_pair_mismatch")
    source = Path(locked["cwd"])
    manifest = _runtime_manifest(source)
    proof = locked["publication_source"]
    if (manifest is None or manifest["source_digest"] != proof["source_digest"]
            or proof["commit"] != result["code_commit"]
            or manifest["git_provenance"]["integration_commit"] != result["code_commit"]
            or manifest["git_provenance"]["code_dirty"] is not False):
        raise MeasurementProvenanceUnavailable("measurement_workload_source_mismatch")
    relative = "src/slm_training/resources/versions.json"
    raw = (source / relative).read_bytes()
    if manifest["files"][relative] != ["file", False, hashlib.sha256(raw).hexdigest()]:
        raise MeasurementProvenanceUnavailable("measurement_registry_bytes_changed")
    registry = json.loads(raw)
    if registry.get("schema") != "version_registry/v1":
        raise MeasurementProvenanceUnavailable("measurement_registry_schema_invalid")
    versions = {key: registry["components"][key]["version"] for key in EVAL_KEY_COMPONENTS}
    if any(result["components"][key] != versions[key] for key in EVAL_KEY_COMPONENTS if key in result["components"]):
        raise MeasurementProvenanceUnavailable("measurement_registry_conflict")
    return versions["gates.ship"]


def resolve_measurement_provenance(store, delivery):
    """Read committed producer evidence; imported controller registry is irrelevant."""

    from scripts.autotrain_cycle_lock import read_artifact
    from scripts.autotrain_ledgers import validate_cycle_delivery

    try:
        events = store.verify_event_chain()
        publication = next(row for row in reversed(events) if row["event_type"] == "cycle_delivery_published")
        if delivery != read_artifact(store, "cycle_deliveries", publication["artifact_sha256"]):
            raise MeasurementProvenanceUnavailable("measurement_delivery_projection_mismatch")
        validate_cycle_delivery(delivery, campaign_id=store.campaign_id)
        arms = delivery.get("arm_order") or [delivery.get("control_id"), delivery.get("candidate_id")]
        if len(arms) != 2 or arms[0] == arms[1]:
            raise MeasurementProvenanceUnavailable("measurement_pair_missing")
        stamps = {arm: _arm_measurement_stamp(store, events, arm) for arm in arms}
        first, second = (stamps[arm] for arm in arms)
        def identity(stamp):
            return {key: value for key, value in stamp.items() if key != "stamped_at"}
        if identity(first) != identity(second):
            raise MeasurementProvenanceUnavailable("measurement_arm_stamps_conflict")
        result = _measurement_stamp(first)
        result["components"]["gates.ship"] = _workload_gate_version(store, result, {**delivery, "arm_order": arms})
        if not isinstance(result["components"]["gates.ship"], str) or result["components"]["gates.ship"] in {"", "unknown", "UNKNOWN"}:
            raise MeasurementProvenanceUnavailable("measurement_gate_version_missing")
        return result, {"arm_stamps": stamps}
    except MeasurementProvenanceUnavailable:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, StopIteration) as exc:
        raise MeasurementProvenanceUnavailable("measurement_provenance_unavailable") from exc


def document_measurement_provenance(store, delivery):
    """Leave existing document action pending until producer proof is available."""
    try:
        return resolve_measurement_provenance(store, delivery)
    except MeasurementProvenanceUnavailable as exc:
        handoff = hashlib.sha256((store.root / "cycle_handoff.json").read_bytes()).hexdigest()
        store.append_event("documentation_waiting_provenance", detail={"reason": str(exc),
            "handoff_sha256": handoff, "diagnostic_measurement_complete": delivery.get("measurement_complete"),
            "wake_source": "authenticated_measurement_provenance", "publication_complete": False},
            idempotency_key="documentation-provenance:" + handoff + ":" + str(exc))
        return None


def document_provenance_pending(root, campaign_id):
    """Replay only current handoff's unresolved provenance prerequisite."""
    from slm_training.autoresearch.storage import CampaignStore, _sha

    store = CampaignStore(campaign_id, Path(root))
    digest = hashlib.sha256((store.root / "cycle_handoff.json").read_bytes()).hexdigest()
    for event in reversed(store.verify_event_chain()):
        if event["event_type"] == "documentation_materialized":
            return None
        if event["event_type"] != "documentation_waiting_provenance":
            continue
        detail = event["detail"]
        if detail.get("handoff_sha256") != digest:
            return None
        return {"schema_version": "driver_pending/v1", "outcome": "dependency",
                "reason": detail["reason"], "campaign_id": campaign_id,
                "measurement_complete": False, "publication_complete": False,
                "diagnostic_measurement_complete": detail.get("diagnostic_measurement_complete"),
                "wake": {"predicate": "original measurement provenance authenticated",
                         "source": "authenticated_measurement_provenance", "identity_digest": _sha(detail)}}
    return None


def render_document_closeout(store, handoff, pending):
    """Resolve document prerequisites before creating any delivery workspace."""
    if not pending:
        return None
    try:
        delivery = json.loads((store.root / "sdlc_delivery.json").read_text())
    except (OSError, ValueError):
        delivery = {}  # Resolver records missing authority; never creates a stamp.
    resolved = document_measurement_provenance(store, delivery)
    if resolved is None:
        return None
    stamp, provenance = resolved
    return render_continuous_cycle_docs(campaign_id=store.campaign_id, loop_id=handoff.loop_id,
        handoff=handoff, delivery=delivery, measurement_version_stamp=stamp,
        measurement_provenance=provenance)


def render_continuous_cycle_docs(
    *,
    campaign_id: str,
    loop_id: str,
    handoff: AutotrainCycleHandoffV1,
    delivery: Mapping[str, Any],
    measurement_version_stamp: dict,
    measurement_provenance: dict | None = None,
) -> tuple[str, dict[str, Any]]:
    """Honest fixture-screening closeout payload (not a ship claim)."""
    reasons = list(delivery.get("reasons") or handoff.reasons or [])
    version_stamp = _measurement_stamp(measurement_version_stamp)
    gate = version_stamp["components"].get("gates.ship")
    if not isinstance(gate, str) or gate in {"", "unknown", "UNKNOWN"}:
        raise MeasurementProvenanceUnavailable("measurement_gate_version_missing")
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
    payload["version_stamp"] = version_stamp
    payload["measurement_provenance"] = measurement_provenance or {}
    if delivery.get("version_stamp") is not None:
        payload["controller_provenance"] = {key: delivery[key] for key in
            ("version_stamp", "controller_execution", "measurement_source") if key in delivery}
    # Embed the rich delivery record (candidate_id/arm_seed/policy_sha256) so
    # future ledger mining never falls back to reasons-string recovery.
    if delivery.get("schema") == "autotrain_sdlc_delivery/v1":
        payload["delivery"] = dict(delivery)
        # Ledger inherits outer measurement stamp, never diagnostic controller stamp.
        payload["delivery"].pop("version_stamp", None)
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
