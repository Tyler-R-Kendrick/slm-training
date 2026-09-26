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
        endpoint={"kind": "denoising_loss", "primary": primary.model_dump(mode="json"),
                  **({"loss_probe": context["loss_probe"]} if context.get("loss_probe") is not None else {})},
        search_slug=slug,
    )
    manifest = bind_search_manifest(template, pair, control, candidate)
    identity, _ = prospective_search_context(manifest, pair, artifact_digest=_sha)
    return slug, identity, manifest, pair


def choose_matrix(matrix, continuous, *, root, loop_id, integration, policy, loss_probe=None):
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
                {"integration": integration, "policy": policy, "loss_probe": loss_probe},
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


def _attempt_event(event, eid, pair, treatment_id):
    detail = event["detail"]
    return all((
        event["event_type"] == "experiment_attempt_started",
        event.get("experiment_id") == eid,
        detail.get("design_digest") == pair["design_digest"],
        detail.get("replicate_id") == pair["replicate_id"],
        detail.get("treatment_id") == treatment_id,
    ))


def _attempt_window(events, eid, pair, attempt_id, attempt_count):
    arm = pair["arm_ids"].index(eid)
    starts = [(i, e["detail"]) for i, e in enumerate(events)
              if _attempt_event(e, eid, pair, pair["treatment_ids"][arm])]
    returns = [(i, e["detail"]) for i, e in enumerate(events)
               if e["event_type"] == "experiment_attempt_returned"
               and e.get("experiment_id") == eid
               and e["detail"].get("attempt_id") == attempt_id]
    if len(starts) != attempt_count or len(returns) != 1:
        return None
    start, started = starts[-1]
    end, returned = returns[0]
    valid = (started.get("attempt_id") == attempt_id
             and started.get("attempt_ordinal") == attempt_count - 1
             and end > start and returned.get("exit_code") == 0
             and all(returned.get(k) == v for k, v in started.items()))
    return (start, end) if valid else None


def _locked_measurement(events, pair, saved):
    from slm_training.autoresearch.storage import _sha

    locks = [e for e in events if e["event_type"] == "experiment_design_locked"
             and e.get("artifact_sha256") == _sha(pair)
             and e.get("detail", {}).get("design_digest") == pair["design_digest"]]
    measurement = pair["design"]["bindings"]["endpoint"]["loss_measurement"]
    return measurement if len(locks) == 1 and saved.get("selection") == measurement.get("selection") else None


def _cursor_artifacts(events, store, eid):
    from slm_training.autoresearch.storage import _sha

    found = {"command_cursor_locked": [], "command_cursor_committed": []}
    for i, event in enumerate(events):
        kind = event["event_type"]
        if event.get("experiment_id") != eid or kind not in found:
            continue
        digest = event.get("artifact_sha256")
        folder = "command_cursor_inputs" if kind.endswith("locked") else "command_cursors"
        path = store.root / "artifacts" / folder / f"{digest}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if _sha(payload) != digest:
            continue
        input_digest = event["detail"].get("input_digest")
        if kind.endswith("locked") and input_digest != digest:
            continue
        if kind.endswith("committed") and input_digest != payload.get("input_digest"):
            continue
        found[kind].append((i, digest, payload))
    return found["command_cursor_locked"], found["command_cursor_committed"]


def _locked_eval_command(locked, store, eid, before, start, manifest_sha256):
    experiment = locked.get("experiment", {})
    commands = locked.get("commands") or []
    valid = all((
        before < start, locked.get("schema") == "command_cursor/v1",
        bool(locked.get("identity")), experiment.get("experiment_id") == eid,
        experiment.get("campaign_id") == store.campaign_id,
        locked.get("manifest") == manifest_sha256,
    ))
    evaluations = [cmd for cmd in commands if "scripts.evaluate_model" in cmd]
    return (commands, evaluations[0]) if valid and len(evaluations) == 1 else None


def _cursor_started(events, input_digest, eid, after, before):
    return any(e["event_type"] == "command_cursor_started"
               and e.get("experiment_id") == eid
               and after < i < before
               and e["detail"].get("input_digest") == input_digest
               for i, e in enumerate(events))


def _committed_outcomes(events, commits, input_digest, commands, binding):
    eid, store, manifest_sha256, start, end = binding
    for index, _, cursor in commits:
        outcome = cursor.get("outcome", {})
        valid = all((
            start < index < end, cursor.get("input_digest") == input_digest,
            cursor.get("record_type") == "committed",
            cursor.get("position") == len(commands), outcome.get("status") == "completed",
            outcome.get("exit_code") == 0, outcome.get("experiment_id") == eid,
            outcome.get("campaign_id") == store.campaign_id,
            outcome.get("campaign_manifest_sha256") == manifest_sha256,
        ))
        if valid and _cursor_started(events, input_digest, eid, start, index):
            yield outcome


def _matching_eval_stage(outcome, evaluation):
    rows = [r for r in outcome.get("stage_telemetry", [])
            if isinstance(r, dict) and isinstance(r.get("parsed_output"), dict)
            and r["parsed_output"].get("measurement_complete") is True]
    if not rows:
        return None
    row = rows[-1]
    try:
        expected = evaluation[evaluation.index("--checkpoint") + 1]
        actual = row["command"][row["command"].index("--checkpoint") + 1]
    except (ValueError, IndexError, KeyError):
        return None
    stage = row["parsed_output"]
    return stage if actual == expected and stage.get("checkpoint") == expected else None


def _stage_report_matches(stage, measurement, saved, scoreboard, report_sha):
    if stage is None:
        return False
    smoke_stage = stage.get("suites", {}).get("smoke", {})
    smoke = scoreboard.get("suites", {}).get("smoke", {})
    source = smoke_stage.get("harness_provenance", {}).get("source_eval_sha256")
    current_source = smoke.get("harness_provenance", {}).get("source_eval_sha256")
    selection = measurement.get("selection", {}).get("selection_sha256")
    checks = (
        source and source == current_source,
        stage.get("checkpoint") is not None,
        stage.get("checkpoint_sha256") == smoke.get("checkpoint_sha256"),
        stage.get("version_stamp") == scoreboard.get("version_stamp"),
        stage.get("eval_data_manifest_sha") == scoreboard.get("eval_data_manifest_sha"),
        smoke_stage.get("selection_sha256") == selection,
        smoke.get("selection_sha256") == selection,
        smoke.get("eval_nll_records_sha256") == report_sha,
        smoke.get("eval_nll_definition_hash") == saved.get("definition_hash"),
        smoke.get("eval_nll") == saved.get("mean_nll"),
    )
    locked = all(saved.get(a) == measurement.get(b) for a, b in (
        ("eval_version", "version"), ("units", "units"), ("estimator_id", "estimator_id")))
    return all(checks) and locked


def _legacy_report_matches_attempt(store, eid, pair, attempt_id, attempt_count,
                                   saved, scoreboard, report_sha):
    """Rebind legacy rows only to their exact locked, committed producer run."""
    events = store.verify_event_chain()
    window = _attempt_window(events, eid, pair, attempt_id, attempt_count)
    measurement = _locked_measurement(events, pair, saved)
    if window is None or measurement is None:
        return False
    start, end = window
    campaign_lock = store.load_experiment_campaign(eid)
    manifest_sha256 = campaign_lock.manifest_sha256
    inputs, commits = _cursor_artifacts(events, store, eid)
    matches = 0
    for lock_index, input_digest, locked in inputs:
        command_data = _locked_eval_command(
            locked, store, eid, lock_index, start, manifest_sha256
        )
        if command_data is None:
            continue
        commands, evaluation = command_data
        binding = (eid, store, manifest_sha256, start, end)
        for outcome in _committed_outcomes(events, commits, input_digest, commands, binding):
            stage = _matching_eval_stage(outcome, evaluation)
            matches += _stage_report_matches(stage, measurement, saved, scoreboard, report_sha)
    return matches == 1


def _loss_report(store, eid, *, attempt_id=None, attempt_count=1, pair=None):
    from scripts.autotrain_metrics import EVAL_NLL_RECORDS_NAME, EVAL_NLL_RECORDS_SCHEMA

    run = store.root / "runs" / eid
    path = run / EVAL_NLL_RECORDS_NAME
    scoreboard = json.loads((run / "scoreboard.json").read_text())
    smoke = scoreboard["suites"]["smoke"]
    report_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if smoke.get("diagnostic_complete") is not True or smoke.get(
        "eval_nll_records_sha256"
    ) != report_sha:
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
    if recorded_attempt is None and attempt_count > 1 and (
        pair is None
        or not _legacy_report_matches_attempt(
            store, eid, pair, attempt_id, attempt_count, saved, scoreboard, report_sha
        )
    ):
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
                    pair=pair,
                ),
                candidate=_loss_report(
                    store, candidate, attempt_id=attempts[candidate],
                    attempt_count=attempt_counts.get(candidate, 0),
                    pair=pair,
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
