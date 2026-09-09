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

    upstream = _git("rev-parse", "--verify", "origin/main^{commit}").stdout.strip()
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
    ]
    return [loss, decode]


def finalize_bundle_remeasurement(store, receipt):
    """Attach real complete loss rows before the driver attempts disposition."""
    from scripts.autotrain_nll import run_arm_eval_nll
    from slm_training.autoresearch.climb_policy import screening_nll_definition_hash
    from .measurement_fixture_evidence import checked_arm

    plan = load_contract(store, receipt["contract_path"])
    arm = checked_arm(plan, receipt["arm_id"])
    loss = arm["loss"]
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
        },
    )
