"""Current compiled treatments feed the existing UCB and campaign evidence store.

Only two-arm, shared-ancestor mechanism screens acquire this strong diagnostic
credit. Unsupported designs retain their existing explicit preflight/gates.
"""

import hashlib
import json

from slm_training.autoresearch.preflight.compiled_treatment import prepare_pair
from slm_training.autoresearch.preflight.search_binding import bind_search_manifest
from slm_training.autoresearch.schemas import ExperimentSpec
from slm_training.autoresearch.search.ingestion import (
    ingest_loss_reports,
    prospective_search_context,
)
from slm_training.autoresearch.storage import CampaignStore, loop_campaigns, _sha


def manifest_options(continuous, *, store, candidate, integration, policy, slug):
    """Resolve new templates only; this helper never rewrites historical locks."""
    from slm_training.versioning import build_version_stamp

    template = continuous._manifest(
        store.campaign_id,
        candidate,
        integration,
        role="screening",
        policy=policy,
        cycle_intent="screening",
        continuation_grant=store.load_campaign().budget.continuation_grant,
    )
    observed = build_version_stamp("harness.autoresearch.experiment_campaign")
    template = template.model_copy(
        update={"source_dirty": observed.get("code_dirty") is not False}
    )
    eid = candidate["experiment_id"]
    return {"manifest_templates": {eid: template}, "search_slugs": {eid: slug}}


def _prospective(store, control, candidate, continuous, context):
    slug = continuous._slug_from_candidate_id(candidate.experiment_id)
    if not slug:
        raise ValueError("search_preflight:unregistered_slug")
    options = manifest_options(
        continuous,
        store=store,
        candidate=candidate.model_dump(mode="json"),
        integration=context["integration"],
        policy=context["policy"],
        slug=slug,
    )
    template = options["manifest_templates"][candidate.experiment_id]
    primary = next(e for e in template.endpoints if e.role == "primary")
    pair = prepare_pair(
        store.load_campaign(),
        control,
        candidate,
        output_root=store.root.parent,
        endpoint={"kind": "denoising_loss", "primary": primary.model_dump(mode="json")},
        search_slug=slug,
    )
    manifest = bind_search_manifest(template, pair, control, candidate)
    identity, _ = prospective_search_context(manifest, pair, artifact_digest=_sha)
    return slug, identity, manifest, pair


def choose_matrix(matrix, continuous, *, root, loop_id, integration, policy):
    """Choose within one currently resolved evidence identity before any lock.

    A historical winner never supplies the current identity. Rejected preflight
    designs do not become measured nulls or permanently retired approaches.
    """
    store = CampaignStore(matrix["campaign_id"], root)
    specs = [
        ExperimentSpec.model_validate(row["experiment"]) for row in matrix["hypotheses"]
    ]
    control, prepared, refused = specs[0], {}, {}
    for candidate in specs[1:]:
        try:
            prepared[candidate.experiment_id] = _prospective(
                store,
                control,
                candidate,
                continuous,
                {"integration": integration, "policy": policy},
            )
        except ValueError as exc:
            refused[candidate.experiment_id] = str(exc)
    reference = prepared.get(matrix["recommended_experiment_id"])
    if reference is None:
        return matrix
    _, identity, manifest, pair = reference
    histories = [
        CampaignStore(c.campaign_id, root)
        for c in loop_campaigns(root, loop_id, last=None)
    ]
    _, effects = prospective_search_context(
        manifest, pair, artifact_digest=_sha, history_stores=histories
    )
    compatible = {row[0]: eid for eid, row in prepared.items() if row[1] == identity}
    selected = (
        continuous._evidence_ranked_slug(
            list(compatible),
            stats={},
            boosts={},
            search_identity=identity,
            search_effects=effects,
        )
        if effects
        else None
    )
    selected_id = compatible.get(selected, matrix["recommended_experiment_id"])
    evidence = {
        "schema_version": "search_selection/v1",
        "identity": identity.model_dump(mode="json"),
        "compatible_candidates": compatible,
        "effect_count": len(effects),
        "refused": refused,
        "selected_experiment_id": selected_id,
        "scientific_completion": False,
    }
    artifact = store.write_artifact("search_selections", evidence)
    store.append_event(
        "search_selection_observed",
        artifact_sha256=artifact.stem,
        idempotency_key="search-selection:" + artifact.stem,
    )
    return {**matrix, "recommended_experiment_id": selected_id}


def _loss_report(store, eid, *, attempt_id=None, attempt_count=1):
    from scripts.autotrain_metrics import EVAL_NLL_RECORDS_NAME, EVAL_NLL_RECORDS_SCHEMA

    run = store.root / "runs" / eid
    path = run / EVAL_NLL_RECORDS_NAME
    scoreboard = json.loads((run / "scoreboard.json").read_text())
    smoke = scoreboard["suites"]["smoke"]
    if smoke.get("diagnostic_complete") is not True or (
        smoke.get("eval_nll_records_sha256")
        != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("search_ingestion:unverified_diagnostic_report")
    saved = json.loads(path.read_text())
    if saved.get("schema") != EVAL_NLL_RECORDS_SCHEMA:
        raise ValueError("search_ingestion:unknown_report_schema")
    recorded_attempt = saved.get("attempt_id")
    if recorded_attempt is not None and recorded_attempt != attempt_id:
        raise ValueError("search_ingestion:attempt_identity_mismatch")
    # Legacy reports predate attempt binding.  They are usable for the first
    # attempt only; reusing one after a retry would turn stale rows into new
    # scientific evidence.
    if recorded_attempt is None and attempt_count > 1:
        raise ValueError("search_ingestion:legacy_report_reused_after_retry")
    return {
        "selection": saved["selection"],
        "estimator_id": saved["estimator_id"],
        "per_record": saved["row_evidence"],
    }


def record_cycle_credit(journal):
    """Idempotent finalization side effect; never promotion or report fabrication."""
    store, value = journal.store, journal.value
    for eid, pair in value["locked_designs"].items():
        if not pair.get("search_slug"):
            continue  # Legacy/non-mechanism evidence retains its original claim.
        events = store.verify_event_chain()
        attempts, started, attempt_counts = {}, {}, {}
        for event in events:
            if event["event_type"] == "experiment_attempt_started":
                arm_id = event["experiment_id"]
                started[arm_id] = event["detail"]["attempt_id"]
                attempt_counts[arm_id] = attempt_counts.get(arm_id, 0) + 1
                attempts.pop(arm_id, None)
            elif event["event_type"] == "experiment_attempt_returned":
                detail = event["detail"]
                arm_id = event["experiment_id"]
                if started.get(arm_id) != detail.get("attempt_id"):
                    continue
                attempts.pop(arm_id, None)
                if type(detail.get("exit_code")) is int and detail["exit_code"] == 0:
                    attempts[arm_id] = detail["attempt_id"]
        control, candidate = pair["arm_ids"]
        try:
            ingest_loss_reports(
                store,
                store.load_experiment_campaign(eid).manifest,
                lineage={
                    "design_sha256": _sha(pair),
                    "control_attempt_id": attempts[control],
                    "candidate_attempt_id": attempts[candidate],
                },
                control=_loss_report(
                    store, control, attempt_id=attempts[control],
                    attempt_count=attempt_counts.get(control, 0),
                ),
                candidate=_loss_report(
                    store, candidate, attempt_id=attempts[candidate],
                    attempt_count=attempt_counts.get(candidate, 0),
                ),
                artifact_digest=_sha,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            refusal = {
                "design_digest": pair["design_digest"],
                "reason": str(exc),
                "claim_class": "unusable_search_evidence",
            }
            artifact = store.write_artifact("search_refusals", refusal)
            store.append_event(
                "search_ingestion_refused",
                artifact_sha256=artifact.stem,
                idempotency_key="search-refusal:" + artifact.stem,
            )

