"""Bind a new (not yet locked) campaign template to its real compiled arms.

This is a controller construction helper, not a migration of historical locks.
Scientific endpoints, thresholds and analysis plans are preserved unchanged.
"""

from slm_training.evals.measurement_identity import content_digest
from slm_training.harness_core.lineage.records import content_sha


def bind_search_manifest(template, pair, control, candidate):
    if len(template.arms) != 2 or sorted(a.role for a in template.arms) != [
        "candidate",
        "control",
    ]:
        raise ValueError("treatment_design:search_requires_two_arm_template")
    endpoints = [e for e in template.endpoints if e.role == "primary"]
    if len(endpoints) != 1 or endpoints[0].metric != "smoke.eval_nll":
        raise ValueError("treatment_design:search_requires_smoke_loss_primary")
    if not pair.get("search_slug"):
        raise ValueError("treatment_design:explicit_search_slug_required")
    arms = (control, candidate)
    if pair["arm_ids"] != [arm.experiment_id for arm in arms]:
        raise ValueError("treatment_design:search_arm_mismatch")
    measurement = pair["design"]["bindings"]["endpoint"]["loss_measurement"]
    if measurement["endpoint_id"] != endpoints[0].endpoint_id:
        raise ValueError("treatment_design:search_endpoint_mismatch")
    hashes = [content_digest(arm.knobs.model_dump(mode="json")) for arm in arms]
    payload = template.model_dump(mode="json")
    payload["arms"] = [
        {
            **next(a.model_dump(mode="json") for a in template.arms if a.role == role),
            "arm_id": arm.experiment_id,
            "config_sha256": sha,
        }
        for role, arm, sha in zip(("control", "candidate"), arms, hashes, strict=True)
    ]
    payload["locked_eval_manifest_sha256"] = measurement["suite_manifest_sha256"]
    manifest = type(template).model_validate(payload)
    pair["campaign_manifest_id"] = manifest.experiment_id
    pair["arm_config_sha256s"] = hashes
    return manifest


def lock_search_manifests(store, manifest, pair):
    """Both CLI executions receive their own immutable applicable manifest."""
    if (
        manifest.campaign_id != store.campaign_id
        or manifest.experiment_id != pair["arm_ids"][1]
    ):
        raise ValueError("treatment_design:campaign_store_or_candidate_mismatch")
    for arm_id in pair["arm_ids"]:
        payload = manifest.model_dump(mode="json")
        payload["experiment_id"] = arm_id
        store.lock_experiment_campaign(type(manifest).model_validate(payload))


def require_attempt_locks(store, pair, arm_id):
    """Strong new designs cannot begin before both controller locks exist."""
    manifest_id = pair.get("campaign_manifest_id")
    if manifest_id is None:
        return  # Legacy design remains weak; ingestion refuses missing bindings.
    manifest = store.load_experiment_campaign(manifest_id).manifest
    applicable = store.load_experiment_campaign(arm_id).manifest
    if applicable.model_dump(exclude={"experiment_id"}) != manifest.model_dump(
        exclude={"experiment_id"}
    ):
        raise ValueError("treatment_design:applicable_campaign_binding_mismatch")
    declared = {arm.arm_id: arm.config_sha256 for arm in manifest.arms}
    if [declared.get(arm) for arm in pair["arm_ids"]] != pair["arm_config_sha256s"]:
        raise ValueError("treatment_design:attempt_campaign_binding_mismatch")
    if not any(
        event["event_type"] == "experiment_design_locked"
        and event.get("artifact_sha256") == content_sha(pair)
        and event["detail"].get("design_digest") == pair["design_digest"]
        and event["detail"].get("replicate_id") == pair["replicate_id"]
        for event in store.verify_event_chain()
    ):
        raise ValueError("treatment_design:attempt_requires_design_lock")
