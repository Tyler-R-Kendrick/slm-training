"""Controller-only ingestion of locked broad-loss comparisons into search credit.

CampaignStore remains the only history owner. Missing legacy measurement bindings
are refused, not inferred from a successful run. Hash checks establish integrity,
not authenticity: only the isolated trusted controller may call this boundary.
"""

from __future__ import annotations

import json

from .evidence import (
    SearchIdentity,
    compatible_effects,
    contract_digest,
    effect_from_loss_reports,
    record_search_effect,
)


def _required(mapping, key):
    if not isinstance(mapping, dict) or key not in mapping or mapping[key] is None:
        raise ValueError(f"search_ingestion:missing_locked_input:{key}")
    return mapping[key]


def _artifact(store, kind, digest, artifact_digest):
    path = store.root / "artifacts" / kind / f"{digest}.json"
    payload = json.loads(path.read_text())
    if artifact_digest(payload) != digest:
        raise ValueError("search_ingestion:artifact_changed")
    return payload


def _design(store, events, digest, artifact_digest):
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("search_ingestion:invalid_design_artifact_identity")
    locks = [
        (i, event)
        for i, event in enumerate(events)
        if event["event_type"] == "experiment_design_locked"
        and event.get("artifact_sha256") == digest
    ]
    if len(locks) != 1:
        raise ValueError("search_ingestion:design_not_uniquely_locked")
    pair = _artifact(store, "treatment_designs", digest, artifact_digest)
    if pair.get("schema") != "compiled_treatment_design/v1":
        raise ValueError("search_ingestion:unsupported_design")
    return locks[0][0], pair


def _attempts(events, pair, lineage, lock_index):
    result = []
    for index, role in enumerate(("control", "candidate")):
        attempt_id = _required(lineage, f"{role}_attempt_id")
        attempts = [
            (i, event)
            for i, event in enumerate(events)
            if event["event_type"] == "experiment_attempt_started"
            and event["detail"].get("attempt_id") == attempt_id
        ]
        if len(attempts) != 1:
            raise ValueError("search_ingestion:attempt_not_uniquely_recorded")
        position, event = attempts[0]
        detail = event["detail"]
        if (
            position <= lock_index
            or event.get("experiment_id") != pair["arm_ids"][index]
            or detail.get("replicate_id") != pair["replicate_id"]
            or detail.get("treatment_id") != pair["treatment_ids"][index]
            or detail.get("design_digest") != pair["design_digest"]
        ):
            raise ValueError("search_ingestion:attempt_binding_mismatch")
        result.append(attempt_id)
    return result


def _identity(manifest, pair):
    design = pair["design"]
    bindings = design["bindings"]
    measurement = _required(bindings["endpoint"], "loss_measurement")
    endpoint_id = _required(measurement, "endpoint_id")
    endpoints = [e for e in manifest.endpoints if e.endpoint_id == endpoint_id]
    if len(endpoints) != 1 or not endpoints[0].metric.endswith("eval_nll"):
        raise ValueError("search_ingestion:locked_endpoint_is_not_broad_loss")
    if not manifest.locked_eval_manifest_sha256:
        raise ValueError("search_ingestion:missing_locked_eval_manifest")
    if pair["independence"] != "conditional_on_shared_ancestor":
        raise ValueError("search_ingestion:independence_not_certified")
    if bindings["starting_checkpoint_role"] not in {"warm_start", "exact_resume"}:
        raise ValueError("search_ingestion:shared_ancestor_required")
    if design["intervention"]["kind"] != "mechanism":
        raise ValueError("search_ingestion:only_matched_mechanism_credit_supported")
    if (
        pair["source"][0] != pair["source"][1]
        or pair["runtime"][0] != pair["runtime"][1]
    ):
        raise ValueError("search_ingestion:unmatched_source_or_runtime")
    # These are actual preflight host/runtime observations, not the current host.
    if not pair["source"][0] or not pair["runtime"][0]:
        raise ValueError("search_ingestion:missing_source_or_runtime")
    selection = _required(measurement, "selection")
    identity = SearchIdentity(
        endpoint=endpoints[0].metric,
        endpoint_version=contract_digest(
            {
                "version": _required(measurement, "version"),
                "scorer": _required(measurement, "evaluator_sha256"),
                "source": pair["source"][0],
            }
        ),
        units=_required(measurement, "units"),
        direction="minimize" if endpoints[0].direction == "decrease" else "maximize",
        estimator=_required(measurement, "estimator_id"),
        suite_digest=manifest.locked_eval_manifest_sha256,
        selection_digest=selection["selection_sha256"],
        data_digest=bindings["training_snapshot"],
        ancestor_digest=bindings["starting_checkpoint"],
        model_digest=contract_digest(
            {
                "architecture": bindings["architecture"],
                "tokenizer_layout": bindings["tokenizer_layout"],
            }
        ),
        host_digest=contract_digest(pair["runtime"][0]),
        fidelity_digest=contract_digest(bindings["resource_contract"]),
        analysis_digest=contract_digest(
            {
                "endpoints": [e.model_dump(mode="json") for e in manifest.endpoints],
                "stopping_rules": manifest.stopping_rules,
                "families": [
                    f.model_dump(mode="json") for f in manifest.multiplicity_families
                ],
            }
        ),
        independence="conditional_on_ancestor",
    )
    return identity, measurement


def locked_search_context(
    store, manifest, *, design_sha256, artifact_digest, history_stores=()
):
    """Current prospective locked context and compatible persisted effects.

    The caller supplies its actual next design, never whichever historical
    effect happened to finish last. No completed reports are needed to select.
    ``history_stores`` is the controller's complete loop-scoped store set; every
    event chain and referenced artifact is verified, irrespective of recency.
    """
    _, pair = _locked_pair(store, manifest, design_sha256, artifact_digest)
    return prospective_search_context(
        manifest,
        pair,
        artifact_digest=artifact_digest,
        history_stores=(*history_stores, store),
    )


def prospective_search_context(manifest, pair, *, artifact_digest, history_stores=()):
    """Read prior compatible credit for actual resolved inputs before arm selection.

    No experiment is locked or executed here. The selected pair must subsequently
    pass the ordinary campaign/design locks before it can produce any evidence.
    """
    identity, _ = _identity(manifest, pair)
    stores = {str(source.root.resolve()): source for source in history_stores}
    rows = [
        _artifact(source, "search_effects", event["artifact_sha256"], artifact_digest)
        for source in stores.values()
        for event in source.verify_event_chain()
        if event["event_type"] == "search_effect_recorded"
    ]
    return identity, compatible_effects(rows, identity)[0]


def _locked_pair(store, manifest, design_sha256, artifact_digest):
    lock = store.load_experiment_campaign(manifest.experiment_id)
    if manifest.campaign_id != store.campaign_id or manifest.model_dump(
        mode="json"
    ) != lock.manifest.model_dump(mode="json"):
        raise ValueError("search_ingestion:campaign_lock_mismatch")
    position, pair = _design(
        store, store.verify_event_chain(), design_sha256, artifact_digest
    )
    if (
        pair.get("campaign_manifest_id", manifest.experiment_id)
        != manifest.experiment_id
    ):
        raise ValueError("search_ingestion:design_campaign_mismatch")
    arm_hashes = _required(pair, "arm_config_sha256s")
    declared = {arm.arm_id: arm for arm in manifest.arms}
    for index, role in enumerate(("control", "candidate")):
        arm = declared.get(pair["arm_ids"][index])
        if arm is None or arm.role != role or arm.config_sha256 != arm_hashes[index]:
            raise ValueError("search_ingestion:campaign_arm_binding_mismatch")
    return position, pair


def ingest_loss_reports(
    store, manifest, *, lineage, control, candidate, artifact_digest
):
    """Validate locked controller inputs, then record one diagnostic SearchEffect.

    ``lineage`` names the treatment-design artifact and both actual attempt IDs.
    The design must additionally bind ``search_slug``, ``arm_config_sha256s``
    (in arm order), and ``design.bindings.endpoint.loss_measurement`` before
    attempts start. See the refusal keys: no missing field receives a default.
    Selection/estimator/scorer/seed must match the real loss producer exactly.
    This does not certify the reports' execution or authorize model promotion;
    the controller must validate its worker receipts before calling it.
    It injects CampaignStore's canonical ``content_sha`` as ``artifact_digest``;
    search must not import the orchestration/lineage package back into selection.
    """
    events = store.verify_event_chain()
    lock_index, pair = _locked_pair(
        store, manifest, _required(lineage, "design_sha256"), artifact_digest
    )
    campaign_index = next(
        i
        for i, event in enumerate(events)
        if event["event_type"] == "experiment_campaign_locked"
        and event.get("experiment_id") == manifest.experiment_id
    )
    attempts = _attempts(events, pair, lineage, max(lock_index, campaign_index))
    identity, measurement = _identity(manifest, pair)
    seed = _required(pair["randomness"], "loss_mask_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("search_ingestion:invalid_locked_loss_seed")
    if pair["randomness"]["seed"] not in manifest.seeds:
        raise ValueError("search_ingestion:unplanned_training_seed")
    for report in (control, candidate):
        if report.get("selection") != measurement["selection"]:
            raise ValueError("search_ingestion:unlocked_loss_selection")
        selected = set(measurement["selection"]["selected_record_ids"])
        for row in report["per_record"]:
            if row["id"] in selected and (
                type(row.get("seed")) is not int
                or row.get("seed") != seed
                or row.get("evaluator_sha256") != measurement["evaluator_sha256"]
            ):
                raise ValueError("search_ingestion:unlocked_loss_scorer_or_seed")
    effect = effect_from_loss_reports(
        identity,
        {
            "slug": _required(pair, "search_slug"),
            "treatment_id": pair["treatment_ids"][1],
            "replicate_id": pair["replicate_id"],
            "comparison_id": contract_digest(
                {
                    "design": pair["design_digest"],
                    "replicate": pair["replicate_id"],
                    "identity": identity.model_dump(mode="json"),
                }
            ),
            "attempt_id": contract_digest(attempts),
        },
        control,
        candidate,
    )
    previous = [
        _artifact(store, "search_effects", event["artifact_sha256"], artifact_digest)
        for event in events
        if event["event_type"] == "search_effect_recorded"
    ]
    compatible_effects([*previous, effect], identity)
    record_search_effect(store, effect)
    return effect
