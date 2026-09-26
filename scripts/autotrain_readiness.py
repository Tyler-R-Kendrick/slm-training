"""Controller readiness inputs from real compiled commands, before model execution.

No science manifest is invented. The returned context is consumed directly by
bootstrap_screening_request and locked as a readiness activity. Run in the
declared cwd, as the actual CLI will: changing process-global cwd here would
race controller threads. This producer never trains or loads a checkpoint.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

from slm_training.data.readiness_contract import LearnedTableInput, evidence_digest, file_digest, scoped_path
from slm_training.data.readiness_lineage import ancestor_snapshots, snapshot_input, snapshot_paths
from slm_training.data.store import DataStore
from slm_training.harnesses.model_build.readiness import configured_ranker


class ReadinessContextUnavailable(ValueError):
    """Explicitly missing controller identity/capability; never inferred from a crash."""


class ReadinessDataFailure(ValueError):
    """A concrete data-owner predicate failed and needs its repair executor."""


def _data_predicate(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except (OSError, ValueError) as exc:
        raise ReadinessDataFailure(str(exc)) from exc


def _argv(command, module):
    if (not isinstance(command, (list, tuple)) or len(command) < 3
            or any(not isinstance(arg, str) for arg in command)
            or list(command[1:3]) != ["-m", module]):
        raise ValueError(f"expected compiled {module} command")
    return list(command[3:])


def _directory(root, kind, value):
    path = _data_predicate(DataStore(root).resolve_path, kind, Path(value))
    path = path if path.is_absolute() else root / path
    return scoped_path(root, path.relative_to(root).as_posix())


def _training_input(root, value):
    directory = _directory(root, "train", value)
    # This is the exact file load_train_records consumes, not a guessed
    # manifest records override which the actual loader would ignore.
    ref = _data_predicate(snapshot_input, root, directory, exposure="train_only")
    _data_predicate(snapshot_paths, root, ref)
    return ref


def _evaluation_inputs(root, commands):
    from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES

    refs = {}
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--suites")
    parser.add_argument("--ship-gates", action="store_true")
    for command in commands:
        args, _ = parser.parse_known_args(_argv(command, "scripts.evaluate_model"))
        directory = _directory(root, "eval", args.test_dir)
        manifest = json.loads((directory / "manifest.json").read_text())
        suites = manifest.get("suites", {})
        required = (args.suites or (",".join(DEFAULT_SHIP_GATES) if args.ship_gates else args.suite)).split(",")
        # Include complete published suites, even for a six-record probe.
        # This is a leakage boundary, not a claim that all rows were scored.
        for name in sorted(set(suites) | {item.strip() for item in required}):
            path = Path(suites.get(name, directory / "suites" / name / "records.jsonl"))
            path = path if path.is_absolute() else root / path
            path = scoped_path(root, path.relative_to(root).as_posix())
            if not path.is_relative_to(directory):
                raise ValueError("scored suite outside its dataset")
            ref = _data_predicate(snapshot_input, root, directory, exposure="public_regression",
                                 records=path.relative_to(directory).as_posix())
            _data_predicate(snapshot_paths, root, ref)
            refs[(ref.directory, ref.records)] = ref
    if not refs:
        raise ValueError("missing scored evaluation commands")
    return list(refs.values())


def _learned_inputs(root, trainer):
    path = configured_ranker(trainer, root)
    if path is None:
        return []
    source = json.loads(path.read_text()).get("source", {})
    if not isinstance(source.get("dataset_id"), str):
        raise ValueError("ranker lacks train dataset provenance")
    ref = _training_input(root, source["dataset_id"])
    if ref.records_sha256 != source.get("records_sha256"):
        raise ValueError("ranker training source identity mismatch")
    return [LearnedTableInput(path=path.relative_to(root).as_posix(), sha256=file_digest(path),
                             training_sources=[ref])]


def _parent_inputs(root, config, lineage_root, starting_run_id):
    from slm_training.harness_core.checkpoint_bundle import published_resume_state, validate_bundle

    checkpoint = config.initialize_from or config.resume_from
    if checkpoint is None:
        if lineage_root or starting_run_id:
            raise ValueError("scratch configuration cannot declare a checkpoint parent")
        return [], None, None, None
    if not lineage_root or not starting_run_id:
        raise ReadinessContextUnavailable("capability_unavailable:starting checkpoint lineage")
    checkpoint = published_resume_state(checkpoint) if config.resume_from else Path(checkpoint)
    checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
    checkpoint = scoped_path(root, checkpoint.relative_to(root).as_posix())
    directory, manifest = validate_bundle(checkpoint.parent.parent.parent, checkpoint.parent.name)
    if checkpoint.name not in manifest["files"] or manifest["metadata"].get("run_id") != starting_run_id:
        raise ValueError("starting checkpoint/run lineage mismatch")
    lineage_path = Path(lineage_root)
    relative = lineage_path.relative_to(root).as_posix() if lineage_path.is_absolute() else lineage_path.as_posix()
    ancestors = ancestor_snapshots(root, relative, starting_run_id)
    return ancestors, relative, starting_run_id, directory.name


def readiness_context_from_commands(*, train_command, eval_commands, cwd,
                                    lineage_root=None, starting_run_id=None):
    """Produce the strict bootstrap/factory context from actual compiler output.

    Call once per arm; unequal contexts must not be silently collapsed. Missing
    published inputs or ancestry are typed capability errors, not an empty
    fabricated training corpus. No positive case count is needed to lock this
    readiness activity. Supports the local TwoTower model-build contract.
    """
    from scripts.train_model import resolve_config
    from slm_training.harnesses.model_build.factory import _twotower_config_from_build
    from slm_training.harnesses.model_build.feature_flags import resolve
    from slm_training.harnesses.model_build import data
    from slm_training.harnesses.model_build import factory
    from slm_training.harnesses.model_build.readiness import prepare_records
    from slm_training.models.twotower import TwoTowerModel
    from slm_training.data.record_admission import assert_training_record
    from slm_training.harnesses.train_data.readiness import screening_request_context

    root = Path(cwd).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("readiness producer must execute in its declared cwd")
    config, flags = resolve(resolve_config(_argv(train_command, "scripts.train_model")), phase="training")
    if config.model_name != "twotower":
        raise ReadinessContextUnavailable("capability_unavailable:non-TwoTower readiness")
    if config.initialize_from and config.resume_from:
        raise ValueError("initialize_from and resume_from are mutually exclusive")
    trainer = asdict(_twotower_config_from_build(config))
    current = [_training_input(root, config.train_dir)]
    for source in (config.replay_train_dir, config.semantic_contrast_dir):
        if source is not None:
            current.append(_training_input(root, source))
    ancestors, lineage, parent, bundle = _parent_inputs(root, config, lineage_root, starting_run_id)
    suites = _evaluation_inputs(root, eval_commands)
    if config.test_dir is not None and (config.eval_every or config.loss_eval_every):
        suites.extend(_evaluation_inputs(root, [[train_command[0], "-m", "scripts.evaluate_model",
            "--test-dir", str(config.test_dir), "--suites", config.eval_suites or "smoke"]]))
    tables = _learned_inputs(root, trainer)
    # Bind resolved config, feature switches, code and checkpoint, not source
    # defaults or a dataset slug. This digest is compatibility, not novelty.
    from pydantic_core import to_jsonable_python
    preprocessing = evidence_digest(to_jsonable_python({"config": asdict(config), "flags": flags,
        "checkpoint_bundle": bundle, "commands": [train_command, *eval_commands],
        "owners": {owner.__module__ + "." + owner.__name__: file_digest(Path(inspect.getfile(owner)))
                   for owner in (resolve_config, resolve, prepare_records, TwoTowerModel, assert_training_record)},
        "data_owner": file_digest(Path(data.__file__)), "factory_owner": file_digest(Path(factory.__file__))}))
    return to_jsonable_python(screening_request_context({"trainer_config": trainer,
        "scored_suites": [ref.model_dump() for ref in {ref.model_dump_json(): ref for ref in suites}.values()],
        "training_ancestors": [ref.model_dump() for ref in {ref.directory: ref for ref in [*current, *ancestors]}.values()],
        "learned_tables": [table.model_dump() for table in tables],
        "initialization": "parent" if parent else "scratch", "lineage_root": lineage,
        "starting_run_id": parent, "starting_checkpoint_bundle": bundle,
        "preprocessing_identity": "compiled-readiness:" + preprocessing}))


def merge_readiness_contexts(contexts):
    """One data action, all distinct actual preparation obligations; no equivalence claim."""
    from pydantic_core import to_jsonable_python
    from slm_training.harnesses.train_data.readiness import screening_request_context

    contexts = [to_jsonable_python(screening_request_context(ctx)) for ctx in contexts]
    if not contexts:
        raise ValueError("missing actual arm readiness contexts")
    first = contexts[0]
    identity = ("initialization", "lineage_root", "starting_run_id", "starting_checkpoint_bundle")
    if any(any(ctx.get(key) != first.get(key) for key in identity) for ctx in contexts):
        raise ValueError("incompatible arm initialization/ancestry assumptions")
    if first["initialization"] == "parent" and not first.get("starting_checkpoint_bundle"):
        raise ValueError("joint parent readiness requires exact checkpoint bundle")
    if any(ctx.get("target_kind", "document") != first.get("target_kind", "document") for ctx in contexts):
        raise ValueError("incompatible arm target contracts")
    configurations = {}
    for ctx in contexts:
        for cfg in [ctx["trainer_config"], *ctx.get("additional_trainer_configs", [])]:
            configurations[evidence_digest(cfg)] = cfg
    configs = list(configurations.values())
    merged = {**first, "trainer_config": configs[0], "additional_trainer_configs": configs[1:],
              "preprocessing_identity": "joint-readiness:" + evidence_digest(
                  [ctx["preprocessing_identity"] for ctx in contexts])}
    for field in ("scored_suites", "training_ancestors", "learned_tables"):
        merged[field] = list({evidence_digest(ref): ref for ctx in contexts for ref in ctx[field]}.values())
    return merged


def _matrix_contexts(matrix, store, cwd, root):
    from slm_training.autoresearch.engine import compile_commands
    from slm_training.autoresearch.schemas import ExperimentSpec

    campaign = store.load_campaign()
    rows = matrix["hypotheses"]
    selected = {rows[0]["experiment"]["experiment_id"], matrix["recommended_experiment_id"],
                *matrix.get("selected_experiment_ids", [])}
    specs = [ExperimentSpec.model_validate(row["experiment"]) for row in rows
             if row["experiment"]["experiment_id"] in selected]
    if len(specs) != len(selected) or len(selected) < 2:
        raise ValueError("missing or duplicate selected matrix arms")
    versions = {spec.knobs.eval_version for spec in specs}
    if len(versions) != 1 or None in versions:
        raise ValueError("joint readiness requires one explicit shared evaluation snapshot")
    contexts = []
    for spec in specs:
        commands = compile_commands(campaign, spec, output_root=root)
        trains = [cmd for cmd in commands if cmd[2] == "scripts.train_model"]
        if len(trains) != 1 or any(cmd[2] == "scripts.build_train_data" for cmd in commands):
            raise ReadinessDataFailure("published training inputs required before readiness; compiled data producer has not completed")
        contexts.append(readiness_context_from_commands(train_command=trains[0],
            eval_commands=[cmd for cmd in commands if cmd[2] == "scripts.evaluate_model"],
            cwd=cwd, **_matrix_parent(spec, cwd)))
    train_version = specs[0].knobs.train_version
    if not train_version:
        raise ReadinessContextUnavailable("locked selected arm lacks explicit train_version")
    return merge_readiness_contexts(contexts), next(iter(versions)), train_version


def _matrix_parent(spec, cwd):
    from slm_training.harness_core.checkpoint_bundle import validate_bundle
    from slm_training.harness_core.lineage.store import LineageStore

    if not spec.knobs.initialize_from:
        return {}
    path = Path(spec.knobs.initialize_from)
    path = path if path.is_absolute() else cwd / path
    _, manifest = validate_bundle(path.parent.parent.parent, path.parent.name)
    return {"lineage_root": LineageStore().root,
            "starting_run_id": manifest["metadata"].get("run_id")}


def _data_budget_exhausted(root, loop_id, bootstrap):
    from slm_training.autoresearch.heal import run_playbooks
    from slm_training.autoresearch.heal.escalation import EscalationLedger, blocker_fingerprint

    fingerprint = blocker_fingerprint(bootstrap["kind"], bootstrap.get("reason", ""),
                                      data_request=bootstrap.get("data_readiness_request"))
    ledger = EscalationLedger.load(root, loop_id)
    limit = inspect.signature(run_playbooks).parameters["max_attempts_per_fingerprint"].default
    return ledger.budget_attempts(fingerprint) >= limit


def resolve_matrix_readiness(matrix, *, cwd, root, loop_id, minimum):
    """Resolve one joint readiness activity before hypothesis/model execution.

    Re-entry reloads exact inputs and independently revalidates the original
    predicate. Parent must rebuild its matrix with the returned eval_version
    before lock. Extra chosen arms can be listed in selected_experiment_ids;
    unselected bank proposals are not execution obligations.
    """
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.data.readiness_contract import load_locked_readiness_request
    from slm_training.data.readiness_receipt import current_successors
    from slm_training.harnesses.train_data.readiness import bootstrap_screening_request, check_readiness
    from scripts.autotrain_controller_repair import dispatch_screening_rebuild

    cwd, root = Path(cwd).resolve(), Path(root).resolve()
    store = CampaignStore(matrix["campaign_id"], root)
    inputs = {"matrix": matrix, "cwd": str(cwd), "loop_id": loop_id, "minimum": minimum}
    artifact = store.write_artifact("matrix_readiness_inputs", inputs)
    digest, bootstrap, version = evidence_digest(inputs), {}, None
    reason, failure_kind = "locked_data_readiness_context", "capability_unavailable"
    try:
        context, version, train_version = _matrix_contexts(matrix, store, cwd, root)
        bootstrap = _data_predicate(bootstrap_screening_request, cwd=cwd, root=root, loop_id=loop_id,
            train_version=train_version, eval_version=version, minimum=minimum, context=context)
        digest = bootstrap["data_readiness_request_sha256"]
        activity = CampaignStore(bootstrap["campaign_id"], root)
        request = load_locked_readiness_request(activity, wanted=digest)
        observed = check_readiness(request, root=cwd)
        failure_kind = observed.get("failure_kind") or "data_failure"
        if observed["ready"]:
            ready = activity.write_artifact("data_readiness", observed)
            event = activity.append_event("matrix_readiness_verified", artifact_sha256=ready.stem,
                detail={"request_sha256": digest}, idempotency_key="original-ready:" + ready.stem)
            return {"status": "ready", "eval_version": version, "request_digest": digest,
                    "readiness_campaign_id": activity.campaign_id, "evidence_event_id": event["event_id"]}
        if failure_kind == "data_failure" and not _data_budget_exhausted(root, loop_id, bootstrap):
            dispatch_screening_rebuild(cwd=cwd, root=root, loop_id=loop_id, campaign_id=None,
                train_version=train_version, eval_version=version, minimum=minimum, readiness_context=context)
        accepted = current_successors(activity, cwd=cwd, loop_id=loop_id,
                                      action_sha=request.action_id, kind="eval")
        if len(accepted) == 1:
            _, candidate, event = accepted[0]
            return {"status": "ready", "eval_version": candidate.dataset_id, "request_digest": digest,
                    "readiness_campaign_id": activity.campaign_id, "evidence_event_id": event["event_id"]}
        # No producer/timer is configured for unlimited resampling. Never invent
        # an asynchronous dependency with no executor or wake source.
        reason = "screening_readiness_not_restored:" + ",".join(observed["errors"])
        if failure_kind == "data_failure" and _data_budget_exhausted(root, loop_id, bootstrap):
            failure_kind, reason = "diagnosis_required", "data_repair_budget_exhausted:" + reason
    except ReadinessContextUnavailable as exc:
        failure_kind, reason = "capability_unavailable", str(exc)
    except (ReadinessDataFailure, FileNotFoundError) as exc:
        failure_kind, reason = "data_failure", f"{type(exc).__name__}:{exc}"
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, SystemExit) as exc:
        failure_kind, reason = "code_failure", f"{type(exc).__name__}:{exc}"
    pending = {"status": "waiting_capability" if failure_kind == "capability_unavailable" else "waiting_dependency",
        "failure_kind": failure_kind, "eval_version": version, "reason": reason,
        "request_digest": digest, "input_artifact_sha256": artifact.stem,
        "readiness_campaign_id": bootstrap.get("campaign_id"),
        "input_path": str(artifact), "input_campaign_id": store.campaign_id,
        "wake": {"predicate": "locked_data_readiness_context", "source": "controller_policy",
                 "identity_digest": digest}}
    if failure_kind != "capability_unavailable":
        # Without a constructible request the data builder cannot execute.
        # Diagnose/fix the context producer instead of asking it to await itself.
        data = failure_kind == "data_failure" and bool(bootstrap)
        pending["blocker"] = {**bootstrap, "kind": "rebuild_data" if data else "repair_harness",
            "blocker_code": ("data_not_ready" if data else "data_readiness_context_failure"
                             if failure_kind == "data_failure" else "screening_constraint_unknown"
                             if failure_kind == "diagnosis_required" else "harness_code_failure"),
            "reason": reason, "campaign_id": bootstrap.get("campaign_id", store.campaign_id),
            "required_capability": "target_data_rebuild" if data else "configured_source_repair",
            "original_reproducer": {"argv": [sys.executable, "-m", "scripts.autotrain_readiness_probe",
                "--input", str(artifact), "--input-digest", evidence_digest(inputs), "--root", str(root)],
                                    "cwd": str(cwd),
                                    "input_artifact_sha256": artifact.stem, "input_identity_digest": evidence_digest(inputs)},
            "executor": "data_rebuild/v1" if data else "configured_source_repair",
            "unmet_predicate": "all_locked_arm_data_preparations_ready"}
        pending["wake"] = {"predicate": "original_data_predicate_restored" if data else "original_reproducer_verified",
            "source": "data_successor_ready" if data else "source_verification_completed", "identity_digest": digest}
    evidence = store.write_artifact("matrix_readiness_wait", pending)
    store.append_event("matrix_readiness_wait", status=pending["status"], artifact_sha256=evidence.stem,
                       detail=pending, idempotency_key="matrix-readiness-wait:" + evidence.stem)
    return pending
