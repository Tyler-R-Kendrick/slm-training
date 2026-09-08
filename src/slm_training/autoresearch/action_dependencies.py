"""Separate a materialized document's delivery wait from scientific execution.

No acknowledgment or scientific gate is issued here. Legacy actions remain
campaign prerequisites; only explicitly scoped new document actions qualify.
"""
from __future__ import annotations

import hashlib
import json

from slm_training.lineage.records import content_sha


def _materialized_delivery(store, handoff):
    events = [row for row in store.verify_event_chain() if row["event_type"] == "documentation_waiting_delivery"]
    if not events:
        return None
    sha = events[-1]["artifact_sha256"]
    path = store.root / "artifacts/delivery_documents" / f"{sha}.json"
    payload = json.loads(path.read_text())
    handoff_sha = hashlib.sha256((store.root / "cycle_handoff.json").read_bytes()).hexdigest()
    required = {f"docs/design/{handoff.campaign_id}-results.md", f"docs/design/{handoff.campaign_id}-results.json"}
    if handoff.checkpoint_documentation_required:
        required.update({"README.md", "docs/MODEL_CARD.md"})
    if (content_sha(payload) != sha or payload.get("schema") != "autotrain_document_materialization/v1"
        or payload.get("campaign_id") != handoff.campaign_id or payload.get("handoff_sha256") != handoff_sha
        or not required.issubset(payload.get("files", {}))
        or any(not isinstance(payload["files"][name], str) or not payload["files"][name] for name in required)):
        raise ValueError("document delivery dependency lacks current materialized evidence")
    return {"campaign_id": handoff.campaign_id, "kind": "document",
            "state": "waiting_capability", "artifact_sha256": sha,
            "required_capability": "authorized_github_connector_delivery",
            "unblock_predicate": "committed documents match the materialized bundle",
            "wake_source": "authorized_delivery_receipt"}


def campaign_prerequisites(root, handoff):
    """Return blocking work and durable scoped waits from the same action history."""
    from slm_training.autoresearch.storage import CampaignStore, pending_autotrain_actions

    pending, waits = [], []
    for index, action in pending_autotrain_actions(root, handoff):
        wait = (_materialized_delivery(CampaignStore(handoff.campaign_id, root), handoff)
                if action.dependency_scope == "delivery" else None)
        if wait is not None:
            waits.append({**wait, "action_index": index})
        else:
            pending.append((index, action))
    return tuple(pending), waits


def require_data_receipt(root, receipt):
    """New receipts require a current predicate; old bytes remain historical."""
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
    from slm_training.autoresearch.storage import autotrain_action_sha256, _receipt_satisfies_action

    from pathlib import Path

    path = Path(root) / receipt.campaign_id / "cycle_handoff.json"
    handoff = AutotrainCycleHandoffV1.model_validate_json(path.read_text())
    if not 0 <= receipt.action_index < len(handoff.actions):
        raise ValueError("wrong data action index")
    action = handoff.actions[receipt.action_index]
    if (receipt.status != "completed" or handoff.loop_id != receipt.loop_id
            or action.kind != "rebuild_data" or autotrain_action_sha256(action) != receipt.action_sha256
            or not _receipt_satisfies_action(root, handoff, action, receipt)):
        raise ValueError("data receipt lacks a current independently restored predicate")
