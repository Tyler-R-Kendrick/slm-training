"""Controller-owned validation of delivery and data prerequisites."""

import hashlib
import json
from pathlib import Path

from slm_training.harness_core.lineage.records import content_sha


def _materialized_delivery(store, handoff):
    events = [e for e in store.verify_event_chain() if e["event_type"] == "documentation_waiting_delivery"]
    if not events:
        return None
    sha = events[-1]["artifact_sha256"]
    payload = json.loads((store.root / "artifacts/delivery_documents" / f"{sha}.json").read_text())
    required = {f"docs/design/{handoff.campaign_id}-results.{x}" for x in ("md", "json")}
    if handoff.checkpoint_documentation_required:
        required.update(("README.md", "docs/MODEL_CARD.md"))
    files = payload.get("files", {})
    valid = (content_sha(payload) == sha and payload.get("schema") == "autotrain_document_materialization/v1"
             and payload.get("campaign_id") == handoff.campaign_id
             and payload.get("handoff_sha256") == hashlib.sha256((store.root / "cycle_handoff.json").read_bytes()).hexdigest()
             and all(isinstance(files.get(n), str) and files[n] for n in required))
    if not valid:
        raise ValueError("document delivery dependency lacks current materialized evidence")
    return {"campaign_id": handoff.campaign_id, "kind": "document", "state": "waiting_capability",
            "artifact_sha256": sha, "required_capability": "authorized_github_connector_delivery",
            "unblock_predicate": "committed documents match the materialized bundle", "wake_source": "authorized_delivery_receipt"}


def campaign_prerequisites(root, handoff):
    from slm_training.autoresearch.storage import CampaignStore, pending_autotrain_actions
    pending, waits, store = [], [], CampaignStore(handoff.campaign_id, root)
    for index, action in pending_autotrain_actions(root, handoff):
        wait = _materialized_delivery(store, handoff) if action.dependency_scope == "delivery" else None
        (waits if wait is not None else pending).append({**wait, "action_index": index} if wait else (index, action))
    return tuple(pending), waits


def require_data_receipt(root, receipt):
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
    from slm_training.autoresearch.storage import autotrain_action_sha256, _receipt_satisfies_action
    handoff = AutotrainCycleHandoffV1.model_validate_json((Path(root) / receipt.campaign_id / "cycle_handoff.json").read_text())
    if not 0 <= receipt.action_index < len(handoff.actions):
        raise ValueError("wrong data action index")
    action = handoff.actions[receipt.action_index]
    if (receipt.status != "completed" or handoff.loop_id != receipt.loop_id or action.kind != "rebuild_data"
            or autotrain_action_sha256(action) != receipt.action_sha256 or not _receipt_satisfies_action(root, handoff, action, receipt)):
        raise ValueError("data receipt lacks a current independently restored predicate")


def validate_connector_document(store, handoff, action, path):
    from slm_training.autoresearch.storage import autotrain_action_sha256
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1

    parent = store.root / "artifacts/connector_document_verifications"
    if path.is_symlink() or path.resolve().parent != parent.resolve():
        raise ValueError("document_receipt_requires_connector_verification")
    proof = json.loads(path.read_text())
    if (content_sha(proof) != path.stem or proof.get("schema") != "connector_document_verification/v1"
            or proof.get("action_sha256") != autotrain_action_sha256(action)
            or not any(e["event_type"] == "connector_document_verified" and e["artifact_sha256"] == path.stem
                       for e in store.verify_event_chain())):
        raise ValueError("unverified_connector_document_receipt")
    current = AutotrainCycleHandoffV1.model_validate_json((store.root / "cycle_handoff.json").read_text())
    wait = proof["wait"]
    index = wait.get("action_index")
    materialized = _materialized_delivery(store, current)
    if (current != handoff or type(index) is not int or not 0 <= index < len(current.actions)
            or current.actions[index] != action or action.dependency_scope != "delivery"
            or materialized is None or {**materialized, "action_index": index} != wait):
        raise ValueError("connector_document_receipt_binding_changed")
    bundle = store.root / "artifacts/delivery_documents" / f"{materialized['artifact_sha256']}.json"
    expected = {name: hashlib.sha256(value.encode()).hexdigest()
                for name, value in json.loads(bundle.read_text())["files"].items()}
    if proof["remote"]["files"] != expected:
        raise ValueError("connector_document_receipt_binding_changed")
