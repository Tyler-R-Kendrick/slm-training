"""Explicit controller-locked inference-only remeasurement, never clean replay.

The finite fixture prepares this contract before work starts. cmd_run still
owns execution and continuation; this module cannot execute worker commands.
"""

from pathlib import Path

from slm_training.autoresearch.engine import compile_commands
from slm_training.autoresearch.experiment_campaign import campaign_manifest_sha256
from slm_training.autoresearch.storage import _sha
from slm_training.evals.measurement_identity import content_digest, selected_identity
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harness_core.checkpoint_bundle import validate_bundle

from .measurement_fixture_plan import _read, _sha as file_sha


def continuous_source_commits(integration):
    """Resolve the actual upstream ref; HEAD is not an upstream identity."""
    from scripts.autoresearch import _git, _validate_continuous_source_refs
    from slm_training.harness_core.execution_release import runtime_git_provenance

    frozen = runtime_git_provenance(Path.cwd())
    upstream = (frozen["upstream_commit"] if frozen is not None else
                _git("rev-parse", "--verify", "origin/main^{commit}").stdout.strip())
    _validate_continuous_source_refs(upstream, integration)
    return {"upstream_commit": upstream, "integration_commit": integration}


def validate_continuous_source(campaign, diagnostic_receipt):
    from scripts.autoresearch import (
        _validate_continuous_commits,
        _validate_continuous_source_refs,
    )

    validator = (
        _validate_continuous_source_refs
        if diagnostic_receipt is not None
        else _validate_continuous_commits
    )
    validator(campaign.upstream_commit, campaign.integration_commit)


def execution_context():
    """Diagnostic identity components, never a weaker continuation identity."""
    from scripts.merge_verification_evidence import (
        environment_identity,
        source_identity,
    )
    from slm_training.harness_core.execution_release import runtime_source_identity

    return {
        "source": runtime_source_identity(Path.cwd()) or source_identity(Path.cwd()),
        "environment": environment_identity(),
    }


def load_contract(store, path):
    path = Path(path)
    payload = _read(path)
    if payload.get("schema") != "supervised_measurement/v1":
        raise ValueError("not a supervised diagnostic measurement contract")
    digest = _sha(payload)
    if not any(
        event["event_type"] == "supervised_measurement_locked"
        and event["artifact_sha256"] == digest
        for event in store.verify_event_chain()
    ):
        raise ValueError("diagnostic contract lacks its controller event lock")
    if (
        payload["campaign_id"] != store.campaign_id
        or payload["role"] != "inference_only_diagnostic"
        or payload["new_training"] is not False
        or payload["promotion_allowed"] is not False
        or payload["ship_eligible"] is not False
    ):
        raise ValueError("diagnostic contract cannot grant training or promotion")
    from scripts.autoresearch_command_cursor import resolved_continuation_grant
    from slm_training.levers import MAX_RUN_SECONDS

    if (
        payload["execution_identity"]
        != resolved_continuation_grant(Path.cwd(), MAX_RUN_SECONDS).execution_identity
    ):
        current = execution_context()
        previous = payload.get("execution_context", {})
        changed = [key for key in current if current[key] != previous.get(key)]
        raise ValueError(f"diagnostic source/environment changed: {changed}")
    inputs = payload["inputs"]
    for filename, expected in inputs["data_manifests"].items():
        if file_sha(filename) != expected:
            raise ValueError("diagnostic dataset manifest changed")
    selection = selected_identity(
        load_suite_records(Path(inputs["test_dir"]), "smoke")[:6]
    )
    if selection != inputs["selection"] or len(selection["selected_record_ids"]) != 6:
        raise ValueError("diagnostic selection changed or is not six cases")
    return payload


def prepare_bundle_remeasurement(args, store, experiment, manifest):
    """Return only compiler-derived commands bound before any workload starts."""
    path = getattr(args, "diagnostic_bundle_plan", None)
    if path is None:
        return None
    if not args.execute or getattr(args, "reuse_train_run", None):
        raise ValueError("diagnostic remeasurement requires exclusive locked execution")
    plan = load_contract(store, path)
    arm_id = experiment.experiment_id
    arm = plan["arms"][arm_id]
    if (
        manifest.claim_class != "fixture"
        or manifest.requires_rl
        or manifest.campaign_id != store.campaign_id
        or manifest.experiment_id != arm_id
        or campaign_manifest_sha256(manifest) != arm["manifest_sha256"]
        or experiment.model_dump(mode="json") != arm["experiment"]
    ):
        raise ValueError("diagnostic arm/manifest differs from its lock")
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        primary_for_role,
    )

    primary = primary_for_role(load_climb_policy(), "screening")
    locked = next(
        endpoint for endpoint in manifest.endpoints if endpoint.role == "primary"
    )
    if any(
        getattr(locked, key) != primary[key]
        for key in ("metric", "direction", "minimum_effect")
    ):
        raise ValueError("diagnostic endpoint differs from current policy")
    checkpoint = Path(plan["inputs"]["arms"][arm_id]["checkpoint"])
    bundle = plan["inputs"]["arms"][arm_id]
    directory, metadata = validate_bundle(
        checkpoint.parent.parent.parent, bundle["bundle_digest"]
    )
    if (
        checkpoint != directory / "last.pt"
        or file_sha(checkpoint) != bundle["checkpoint_sha256"]
        or not {
            "last.pt",
            "last.meta.json",
            "last.tokenizer.json",
            "last.context.tokenizer.json",
        }
        <= set(metadata["files"])
    ):
        raise ValueError("diagnostic inference bundle is incomplete or changed")
    summary = _read(arm["source_summary"])
    if file_sha(arm["source_summary"]) != arm["source_summary_sha256"]:
        raise ValueError("original training provenance changed")
    params = summary["track"]["trainable_params"]
    if (
        type(params) is not int
        or params <= 0
        or params != bundle["trainable_parameters"]
    ):
        raise ValueError("diagnostic parameter count lacks valid original evidence")
    compiled = compile_commands(
        store.load_campaign(), experiment, output_root=store.root.parent
    )
    if compiled != arm["compiled"]:
        raise ValueError("diagnostic compiler output differs from its lock")
    commands = inference_commands(
        compiled, checkpoint, plan["inputs"], store.root / "runs" / arm_id
    )
    if commands != arm["commands"]:
        raise ValueError("diagnostic commands differ from compiler-derived plan")
    return {
        "stage_kind": "diagnostic_checkpoint_remeasurement",
        "executed": False,
        "training_continued": False,
        "new_independent_replication": False,
        "promotion_authority": False,
        "trainable_params": params,
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "source_training_version_stamp": summary["version_stamp"],
        "contract_path": str(path),
        "contract_sha256": _sha(plan),
        "arm_id": arm_id,
        "commands": commands,
    }


def inference_commands(compiled, checkpoint, inputs, run_dir):
    """No raw argv from the contract is executable without this reconstruction."""
    import sys

    train = [cmd for cmd in compiled if "scripts.train_model" in cmd]
    evaluate = [list(cmd) for cmd in compiled if "scripts.evaluate_model" in cmd]
    if len(compiled) != 2 or len(train) != 1 or len(evaluate) != 1:
        raise ValueError(
            "diagnostic remeasurement requires one ordinary train/decode plan"
        )
    decode = evaluate[0]
    if "--checkpoint" in decode or "--resume-run" in decode:
        raise ValueError("diagnostic checkpoint cannot override another binding")
    decode.extend(["--checkpoint", str(checkpoint), "--resume-run", str(run_dir)])
    loss = [
        sys.executable,
        "-m",
        "scripts.evaluate_loss_suites",
        "--checkpoint",
        str(checkpoint),
        "--test-dir",
        inputs["test_dir"],
        "--base-suite",
        "smoke",
        "--ood-suite",
        "smoke",
        "--limit",
        "6",
        "--mask-seed",
        "7301",
        "--no-legal-support",
        "--out",
        str(run_dir / "loss_suites.json"),
        "--campaign-dir",
        str(run_dir.parent.parent),
        "--experiment-id",
        run_dir.name,
    ]
    return [loss, decode]


def finalize_bundle_remeasurement(store, receipt):
    """Attach real complete loss rows before the driver attempts disposition."""
    from scripts.autotrain_nll import campaign_attempt_id, run_arm_eval_nll
    from slm_training.autoresearch.climb_policy import screening_nll_definition_hash
    from .measurement_fixture_evidence import checked_arm

    plan = load_contract(store, receipt["contract_path"])
    arm = checked_arm(plan, receipt["arm_id"])
    loss = arm["loss"]
    producer = verified_loss_producer(store, receipt, arm["run_dir"] / "loss_suites.json")
    run_arm_eval_nll(
        arm["run_dir"],
        {
            "eval_nll": loss["categories"]["broad"]["aggregate"]["mean_nll"],
            "records": arm["records"],
            "selection": loss["selection"],
            "row_evidence": loss["per_record"],
            "estimator_id": loss["estimator_id"],
            "definition_hash": content_digest(
                {
                    "screening_definition": screening_nll_definition_hash(),
                    "loss_definition": loss["definition"],
                }
            ),
            "eval_version": plan["inputs"]["eval_version"],
            "attempt_id": campaign_attempt_id(store, receipt["arm_id"]),
            "producer_evidence": producer,
        },
    )


def verified_loss_producer(store, receipt, path):
    """Bind reused rows to their original successful cursor stage, never relabel it."""
    loss = _read(path)
    producer = loss.get("attempt_id")
    if not isinstance(producer, str) or not producer:
        raise ValueError("loss producer lacks attempt binding; replay required")
    digest = file_sha(path)
    active = None
    attempts = {}
    for event in store.verify_event_chain():
        if event["experiment_id"] != receipt["arm_id"]:
            continue
        kind, detail = event["event_type"], event["detail"]
        active = _active_attempt_after_event(kind, detail, active)
        if kind == "command_cursor_started":
            attempts[(detail["input_digest"], detail["attempt"])] = active
        elif kind == "command_cursor_committed":
            payload = _read(store.root / "artifacts/command_cursors" / f"{event['artifact_sha256']}.json")
            if _sha(payload) != event["artifact_sha256"]:
                raise ValueError("loss producer cursor artifact changed")
            key = (payload["input_digest"], payload["attempt"])
            if attempts.get(key) != producer:
                continue
            _verify_loss_cursor_inputs(store, key[0], receipt)
            expected = {"out": str(path), "report_sha256": digest, "attempt_id": producer}
            if _bound_loss_stage(payload["outcome"], receipt["commands"][0], expected):
                return {"attempt_id": producer, "loss_report_sha256": digest,
                        "cursor_artifact_sha256": event["artifact_sha256"],
                        "cursor_input_digest": key[0]}
    raise ValueError("loss report lacks a successful digest-bound producer stage")


def _active_attempt_after_event(kind, detail, active):
    if kind == "experiment_attempt_started":
        return detail["attempt_id"]
    if kind == "experiment_attempt_returned" and detail.get("attempt_id") == active:
        return None
    return active


def _verify_loss_cursor_inputs(store, digest, receipt):
    inputs = _read(store.root / "artifacts/command_cursor_inputs" / f"{digest}.json")
    if (_sha(inputs) != digest or inputs["commands"] != receipt["commands"]
            or inputs["experiment"]["experiment_id"] != receipt["arm_id"]):
        raise ValueError("loss producer cursor inputs changed")


def _bound_loss_stage(outcome, command, expected):
    import math

    for stage in outcome["stage_telemetry"]:
        if stage.get("command") != command:
            continue
        parsed = stage.get("parsed_output") or {}
        seconds = stage.get("duration_seconds")
        if (type(stage.get("exit_code")) is int and stage["exit_code"] == 0
                and type(seconds) in (int, float)
                and math.isfinite(seconds) and seconds >= 0
                and not any(stage.get(k) for k in ("timed_out", "interrupted", "killed"))
                and all(parsed.get(key) == value for key, value in expected.items())):
            return True
    return False
