"""Finite real supervisor/cmd_run acceptance on locked inference-only bundles.

Preparation is an explicitly controlled public fixture, not autonomous research
selection. The ordinary driver, command cursor, evaluator and verdict run real.
"""

import argparse
import sys
from pathlib import Path

from slm_training.harness_core.activity_contract import contract_digest
from scripts.autotrain_cycle_context import register, writer
from slm_training.autoresearch.experiment_campaign import (
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.schemas import CampaignSpec, ExperimentSpec
from slm_training.autoresearch.storage import CampaignStore, _sha as payload_sha
from slm_training.levers import MAX_RUN_SECONDS
from slm_training.versioning import build_version_stamp

from .cli import _write
from . import measurement_bundle as bundle
from .measurement_fixture import _inputs, source_identity
from .measurement_fixture_plan import _read, _sha, compile_fixture, supervised_matrix


def _design(store, plan):
    from slm_training.autoresearch.experiment_identity import (
        digest,
        replicate_identity,
        treatment_identity,
        InterventionContract,
    )
    from slm_training.autoresearch.preflight.compiled_treatment import persist_pair

    contract = InterventionContract(
        kind="mechanism",
        varied_fields=("checkpoint_bundle",),
        resource_basis="wall_seconds",
    )
    treatments, configurations = [], []
    for name in ("control", "candidate"):
        bundle = plan["inputs"]["arms"][name]
        directory = Path(bundle["checkpoint"]).parent
        metadata = _read(directory / "manifest.json")
        summary = _read(plan["arms"][name]["source_summary"])
        bindings = dict(
            architecture=_sha(directory / "last.meta.json"),
            tokenizer_layout=_sha(directory / "last.tokenizer.json"),
            training_snapshot=metadata["metadata"]["data_manifest_sha"],
            preprocessing=dict(
                source_summary_sha256=plan["arms"][name]["source_summary_sha256"]
            ),
            starting_checkpoint=bundle["bundle_digest"],
            starting_checkpoint_role="inference_only",
            endpoint=plan["primary"],
            resource_contract=dict(
                new_training_updates=0,
                selected_cases=6,
                total_wall_seconds=MAX_RUN_SECONDS,
            ),
        )
        config = dict(
            steps=summary["steps"],
            batch_size=summary["recipe"]["batch_size"],
            checkpoint_bundle=bundle["bundle_digest"],
            training_recipe_is_historical=True,
        )
        treatments.append(
            treatment_identity(config, bindings=bindings, intervention=contract)
        )
        configurations.append(dict(config=config, bindings=bindings))
    hypothesis = digest("supervised inference-only remeasurement of the retained pair")
    design_digest = digest(configurations)
    pair = dict(
        schema="diagnostic_remeasurement_design/v1",
        hypothesis_id=hypothesis,
        design_digest=design_digest,
        treatment_ids=treatments,
        arm_ids=["control", "candidate"],
        arm_config_sha256s=[
            arm.config_sha256
            for arm in store.load_experiment_campaign("candidate").manifest.arms
        ],
        campaign_manifest_id="candidate",
        design=configurations,
        independence="conditional_on_retained_training_states; no new independent training replicates",
        replicate_id=replicate_identity(
            hypothesis_id=hypothesis,
            design_digest=design_digest,
            unit_id="fixed-mask-seed-7301",
            seeds={"mask": 7301},
            ancestor_digest=digest(plan["inputs"]["arms"]),
            independence="conditional_on_retained_states",
        ),
    )
    persist_pair(store, pair)
    return pair


def prepare(store, loop_id, retained_plan):
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        primary_for_role,
    )
    from slm_training.autoresearch.engine import compile_commands

    if (store.root / "campaign.json").exists():
        raise ValueError(
            "supervised fixture requires a fresh campaign; never overwrite a lock"
        )
    previous = _read(retained_plan)
    inputs = _inputs(
        Path("outputs/runs/autonomy-mea-data/final-evidence.json"),
        previous["inputs"]["train_version"],
        previous["inputs"]["eval_version"],
    )
    if inputs != previous["inputs"]:
        raise ValueError("retained fixture input identities changed")
    primary = dict(primary_for_role(load_climb_policy(), "screening"))
    if primary != previous["primary"] or primary["claim_class"] != "diagnostic":
        raise ValueError(
            "new supervisor lock requires the same declared diagnostic endpoint"
        )
    stamp = build_version_stamp()
    campaign = CampaignSpec(
        campaign_id=store.campaign_id,
        loop_id=loop_id,
        cycle_index=1,
        **bundle.continuous_source_commits(stamp["code_commit"]),
        primary_metric=primary["metric"],
        objective="Finite supervised six-case inference-only remeasurement",
        budget=dict(max_experiments=2, max_gpu_hours=0, max_wall_minutes=3),
        notes="Controlled public fixture preparation; dirty authoring tree explicitly content-bound; no new training or promotion",
    )
    store.initialize(campaign)
    arms = {}
    for name in ("control", "candidate"):
        arm = compile_fixture(campaign, name, inputs, output_root=store.root.parent)
        summary_path = (
            Path(inputs["arms"][name]["checkpoint"]).parents[4] / "train_summary.json"
        )
        summary = _read(summary_path)
        experiment = ExperimentSpec.model_validate(
            {
                **arm["experiment"],
                "hypothesis": f"The fixed {name} checkpoint produces complete diagnostic evidence through the real supervisor.",
                "citations": [str(retained_plan)],
                "knobs": {
                    **arm["experiment"]["knobs"],
                    "lr": summary["recipe"]["learning_rate"],
                    "generate_batch_size": 2,
                },
            }
        )
        compiled = compile_commands(campaign, experiment, output_root=store.root.parent)
        arms[name] = {
            **arm,
            "experiment": experiment.model_dump(mode="json"),
            "compiled": compiled,
            "commands": bundle.inference_commands(
                compiled,
                inputs["arms"][name]["checkpoint"],
                inputs,
                Path(arm["run_dir"]),
            ),
            "source_summary": str(summary_path),
            "source_summary_sha256": _sha(summary_path),
        }
    source_store = CampaignStore(
        Path(retained_plan).parent.name, Path(retained_plan).parent.parent
    )
    template = source_store.load_experiment_campaign(
        "MEA-SIX-RESOLVED-ENDPOINT"
    ).manifest.model_dump(mode="json")
    template.update(
        campaign_id=store.campaign_id,
        source_commit=stamp["code_commit"],
        source_dirty=True,
        created_at=campaign.created_at,
        author="explicit supervised diagnostic fixture",
        stopping_rules=[
            "Same-campaign cursor; total driver budget 180 seconds, each arm invocation at most 95 seconds; no training/promotion"
        ],
    )
    from slm_training.evals.measurement_identity import content_digest

    template["arms"] = [
        dict(
            arm_id=name,
            role=name,
            config_sha256=content_digest(
                {"experiment": arms[name]["experiment"], "bundle": inputs["arms"][name]}
            ),
        )
        for name in arms
    ]
    for name, arm in arms.items():
        manifest = ExperimentCampaignV1.model_validate(
            {**template, "experiment_id": name}
        )
        store.lock_experiment_campaign(manifest)
        arm["manifest_sha256"] = campaign_manifest_sha256(manifest)
        _write(
            store.root / "manifests" / f"{name}.json", manifest.model_dump(mode="json")
        )
        _write(store.root / "experiments" / f"{name}.json", arm["experiment"])
    matrix = supervised_matrix(campaign, arms, str(retained_plan))
    artifact = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=artifact.stem)
    context = bundle.execution_context()
    identity = contract_digest(context)
    plan = dict(
        schema="supervised_measurement/v1",
        campaign_id=store.campaign_id,
        role="inference_only_diagnostic",
        new_training=False,
        promotion_allowed=False,
        ship_eligible=False,
        inputs=inputs,
        arms=arms,
        primary=primary,
        identity=source_identity(),
        execution_identity=identity,
        execution_context=context,
    )
    path = store.write_artifact("supervised_measurement", plan)
    store.append_event("supervised_measurement_locked", artifact_sha256=path.stem)
    _write(store.root / "supervised_measurement.json", plan)
    pair = _design(store, plan)
    _register_cycle(store, campaign, matrix, plan, pair, path)
    return plan


def _register_cycle(store, campaign, matrix, plan, pair, path):
    from scripts.autotrain_cycle_prepare import _FIELDS
    from slm_training.autoresearch.climb_policy import load_climb_policy

    value = dict.fromkeys(_FIELDS)
    root, cwd = store.root.parent, Path.cwd()
    ar = [sys.executable, "-m", "scripts.autoresearch", "--root", str(root)]
    value.update(
        schema_version="driver_cycle/v1",
        campaign_id=store.campaign_id,
        loop_id=campaign.loop_id,
        cycle=1,
        upstream=campaign.upstream_commit,
        integration=campaign.integration_commit,
        role="screening",
        cycle_intent="diagnostic_remeasurement",
        effective_primary=plan["primary"]["metric"],
        matrix=matrix.model_dump(mode="json"),
        control_eid="control",
        candidate_eid="candidate",
        order=["control", "candidate"],
        scheduled_order=["control", "candidate"],
        arm_seed=7301,
        arm_wall_minutes=95 / 60,
        by_id={a: str(store.root / "experiments" / f"{a}.json") for a in plan["arms"]},
        replay_manifest_paths={},
        replay_manifests={},
        screening_multi=False,
        screening_candidate_ids=["candidate"],
        role_primary=plan["primary"],
        multi_arm_skip=[],
        train_version=plan["inputs"]["train_version"],
        eval_version=plan["inputs"]["eval_version"],
        claim_for_role="fixture",
        ar=ar,
        locked_designs={"candidate": pair},
        replay=None,
        skip_slugs=[],
        root_arg=str(root),
        cwd=str(cwd.resolve()),
        total_seconds=float(MAX_RUN_SECONDS),
        initial_spent_seconds=0.0,
        policy_sha256=load_climb_policy().sha256,
        execution_identity=plan["execution_identity"],
        arms={},
        files={str(path): _sha(path)},
    )
    for name, arm in plan["arms"].items():
        experiment, manifest = (
            store.root / "experiments" / f"{name}.json",
            store.root / "manifests" / f"{name}.json",
        )
        value["arms"][name] = dict(
            cmd=[
                *ar,
                "run",
                "--campaign-id",
                store.campaign_id,
                "--experiment",
                str(experiment),
                "--campaign-manifest",
                str(manifest),
                "--execute",
                "--experiment-wall-seconds",
                "95",
                "--diagnostic-bundle-plan",
                str(path),
            ],
            commands=arm["commands"],
            manifest_digest=arm["manifest_sha256"],
            experiment_path=str(experiment),
        )
        value["files"].update(
            {
                str(p): _sha(p)
                for p in (experiment, manifest, Path(arm["source_summary"]))
            }
        )
    with writer(root, campaign.loop_id) as runtime:
        register(store, runtime, value)


def run(store):
    from scripts.autotrain_supervisor_operations import run_operation
    from scripts.autotrain_pending import drain_driver_pending
    from scripts.merge_verification_evidence import digest, environment_identity
    from scripts.run_autotrain_supervisor import _source_identity
    from .measurement_fixture_evidence import recording_runtime
    from .measurement_supervised_evidence import collect_supervised

    plan = bundle.load_contract(store, store.root / "supervised_measurement.json")
    loop_id = store.load_campaign().loop_id
    journal = CampaignStore("runtime", store.root.parent / "loops" / loop_id)
    # Finite explicit operation; never install/start a persistent service.
    request = dict(
        operation="driver",
        cwd=str(Path.cwd()),
        root=str(store.root.parent),
        loop_id=loop_id,
        source_digest=_source_identity(Path.cwd()),
        environment_digest=digest(environment_identity()),
        driver_argv=[
            "--root",
            str(store.root.parent),
            "--loop-id",
            loop_id,
            "--supervised",
            "--max-cycles",
            "1",
        ],
    )
    with recording_runtime(journal) as runtime:
        sequence = len(
            [
                e
                for e in journal.verify_event_chain()
                if e["event_type"] == "activity_controller_started"
            ]
        )
        result = run_operation(
            runtime,
            request,
            sequence=sequence,
            log_event=lambda event: print(event, flush=True),
        )
        if result and result.get("returncode") == 10:
            # A pending driver is a resumable activity, not a terminal stop.
            # Run the canonical probe/repair dispatcher before this bounded
            # invocation exits so the next invocation has a durable wake source.
            drain_driver_pending(
                runtime,
                request,
                sequence,
                lambda event: print(event, flush=True),
                run_operation,
            )
    if result is not None and result.get("returncode") == 0:
        collect_supervised(store, plan, result)
    return dict(payload=result, contract_sha256=payload_sha(plan))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--loop-id", default="autonomy-mea-supervised-20260908")
    parser.add_argument("--root", type=Path, default=Path("outputs/autoresearch"))
    parser.add_argument(
        "--retained-plan",
        type=Path,
        default=Path("outputs/autoresearch")
        / "autonomy-mea-resolved-endpoint-20260908-r2"
        / "measurement_fixture.json",
    )
    parser.add_argument(
        "--enable-fixture-experiment", action="store_true", required=True
    )
    args = parser.parse_args(argv)
    store = CampaignStore(args.run_id, args.root.resolve())
    result = (
        prepare(store, args.loop_id, args.retained_plan.resolve())
        if args.action == "prepare"
        else run(store)
    )
    print(
        {
            "action": args.action,
            "campaign_id": store.campaign_id,
            "promotion_allowed": False,
            "result": result if args.action == "run" else "locked",
        }
    )
    return 0 if args.action == "prepare" or result["payload"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
