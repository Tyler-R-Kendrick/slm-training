"""Acknowledging and shaping the loop's self-heal actions.

One responsibility: turning a diagnosis into an acknowledged action -- document
it, rebuild data, refresh a capability objective -- and recording the signals
those actions leave behind.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from scripts.autotrain_arms import current_rung_label
from scripts.autotrain_diagnosis import BANK_EXHAUST_MARKERS
from scripts.autotrain_io import read_json
from scripts.autotrain_paths import heal_retired_versions_path
from slm_training.autoresearch.schemas import (
    AutotrainActionReceiptV1,
    AutotrainActionV1,
    AutotrainCycleHandoffV1,
    HarnessSignalV1,
    HypothesisFeedback,
)
from slm_training.autoresearch.storage import (
    CampaignStore,
    append_autotrain_action_receipt,
    autotrain_action_sha256,
    bind_autotrain_action_evidence,
)


def latest_hypothesis_feedback(root: Path, campaign_id: str) -> HypothesisFeedback:
    """Load the terminal typed feedback that grounds an objective change.

    Incomplete retries may have no hypothesizer event; walk predecessor
    campaigns until a recorded feedback exists.
    """

    seen: set[str] = set()
    current: str | None = campaign_id
    while current and current not in seen:
        seen.add(current)
        store = CampaignStore(current, root)
        try:
            events = store.verify_event_chain()
        except Exception:  # noqa: BLE001 — missing/broken chain, try predecessor
            events = []
        for event in reversed(events):
            if event.get("event_type") != "hypothesizer_feedback_recorded":
                continue
            digest = str(event.get("artifact_sha256") or "")
            path = store.root / "artifacts" / "hypothesizer_feedback" / f"{digest}.json"
            if not path.is_file():
                continue
            feedback = HypothesisFeedback.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if feedback.campaign_id == current:
                return feedback
        spec = read_json(root / current / "campaign.json")
        nxt = str(spec.get("predecessor_campaign_id") or "") if spec else ""
        current = nxt or None
    raise RuntimeError(
        "screening objective change requires terminal HypothesisFeedback"
    )


def capability_objective_refresh_actions(
    *,
    root: Path,
    campaign_id: str,
    preserved_actions: Sequence[AutotrainActionV1] = (),
) -> tuple[AutotrainActionV1, ...]:
    """Route exhausted smoke search into the existing rung/data/research loop.

    Every action targets the policy's *current* rung. Naming a later rung
    (e.g. simplified-NL while grammar_2_ast is uncertified) turns the pending
    action into an I10 skip that no legal heal can execute.
    """

    feedback = latest_hypothesis_feedback(root, campaign_id)
    evidence_ids = (feedback.feedback_id, f"campaign:{campaign_id}")
    rung = current_rung_label()
    return (
        AutotrainActionV1(
            kind="rebuild_data",
            owner="synthesis-feedback",
            reason=(
                f"rebuild the current-rung ({rung}) training corpus from the "
                "climb-policy plan; preserve I10 rung gates and inspect the "
                "quality report before any new training"
            ),
            evidence_ids=evidence_ids,
        ),
        *preserved_actions,
        AutotrainActionV1(
            kind="next_experiment",
            owner="autotrain",
            reason=(
                "after the data receipt and objective change, invoke the configured "
                "Researcher once with the terminal HypothesisFeedback and "
                f"preregister a size-matched capability objective for the current "
                f"rung ({rung}); do not rotate the exhausted decoder-lever bank"
            ),
            evidence_ids=evidence_ids,
        ),
    )


def is_bank_exhaust_repair_action(action: AutotrainActionV1) -> bool:
    """True when repair_harness is thrash bank exhaust (soft if compose can reopen)."""
    if action.kind != "repair_harness":
        return False
    reason = str(action.reason or "").lower()
    return any(m in reason for m in BANK_EXHAUST_MARKERS)


def ack_document_action(
    root: Path,
    handoff: AutotrainCycleHandoffV1,
    *,
    action_index: int,
    evidence_uris: Sequence[str],
) -> None:
    action = handoff.actions[action_index]
    if action.kind != "document":
        raise ValueError(f"refusing to auto-ack non-document action: {action.kind}")
    uris = tuple(evidence_uris)
    evidence = bind_autotrain_action_evidence(root, handoff, action, uris)
    append_autotrain_action_receipt(
        root,
        AutotrainActionReceiptV1(
            loop_id=handoff.loop_id,
            campaign_id=handoff.campaign_id,
            action_index=action_index,
            action_sha256=autotrain_action_sha256(action),
            action_kind="document",
            status="completed",
            evidence_uris=uris,
            evidence=evidence,
        ),
    )


def rebuild_data_artifact_sources(train_dir: Path) -> dict[str, Path] | None:
    """Map receipt names to files written by build_train_data."""
    quality = train_dir / "quality_report.json"
    feedback = train_dir / "synthesis_feedback.json"
    manifest = train_dir / "data_manifest.json"
    if not manifest.is_file():
        manifest = train_dir / "manifest.json"
    if not (quality.is_file() and feedback.is_file() and manifest.is_file()):
        return None
    return {
        "quality_report.json": quality,
        "synthesis_feedback.json": feedback,
        "data_manifest.json": manifest,
    }


def ack_rebuild_data_action(
    root: Path,
    handoff: AutotrainCycleHandoffV1,
    *,
    action_index: int,
    evidence_uris: Sequence[str],
    counts: tuple[int, int] | None = None,
) -> None:
    """Acknowledge one ``rebuild_data`` action with bound evidence.

    ``counts`` is the heal's ``(records_before, records_after)`` postcondition:
    when given, an ack is refused unless the count actually grew — a sidecar
    path alone is never evidence that data changed.
    """
    action = handoff.actions[action_index]
    if action.kind != "rebuild_data":
        raise ValueError(f"refusing to ack non-rebuild_data action: {action.kind}")
    if counts is not None:
        before, after = int(counts[0]), int(counts[1])
        if after <= before:
            raise ValueError(
                "refusing to ack rebuild_data without a count postcondition: "
                f"records_before={before} records_after={after}"
            )
    uris = tuple(evidence_uris)
    evidence = bind_autotrain_action_evidence(root, handoff, action, uris)
    append_autotrain_action_receipt(
        root,
        AutotrainActionReceiptV1(
            loop_id=handoff.loop_id,
            campaign_id=handoff.campaign_id,
            action_index=action_index,
            action_sha256=autotrain_action_sha256(action),
            action_kind="rebuild_data",
            status="completed",
            evidence_uris=uris,
            evidence=evidence,
        ),
    )


def retired_heal_versions(root: Path, loop_id: str) -> set[str]:
    """Train versions whose heal arm was measured and retired (tombstones)."""
    path = heal_retired_versions_path(root, loop_id)
    if not path.is_file():
        return set()
    versions: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        version = str(row.get("train_version") or "")
        if version:
            versions.add(version)
    return versions


def persist_selector_harness_signal(
    root: Path,
    campaign_id: str,
    loop_id: str,
    source_campaigns: Sequence[str],
) -> None:
    """Persist one content-addressed signal for a proven selector regression."""

    if not source_campaigns:
        return
    signal = HarnessSignalV1(
        family="autoresearch",
        code="screening_selector_reintroduced_retired_arm",
        evidence_uri=f"loops/{loop_id}/exhausted_knob_ledger.json",
        reproduced_on_frozen_input=True,
        primary=True,
    )
    store = CampaignStore(campaign_id, root=root)
    path = store.write_artifact("harness_signals", signal)
    store.append_event(
        "harness_signal_recorded",
        status="reproduced",
        artifact_sha256=path.stem,
        detail={"source_campaigns": list(source_campaigns)},
    )
