"""LRN-CORRECTIVE-1: default-off, matched data intervention via model-build.

The controller supplies an already authorized manifest and admitted datasets.
This executes ordinary training/loss/decode owners; it owns no promotion rule.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from slm_training.evals.measurement_identity import content_digest

EXPERIMENT_ID = "LRN-CORRECTIVE-1"
DEFAULT_ENABLED = False


def fixture_config(root, train_dir, *, run_id, steps, test_dir=None, **changes):
    from slm_training.harnesses.model_build.config import ModelBuildConfig

    return ModelBuildConfig(
        train_dir=Path(train_dir),
        test_dir=Path(test_dir) if test_dir else None,
        run_root=root / "runs",
        run_id=run_id,
        steps=steps,
        seed=7301,
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
        freeze_context=False,
        batch_size=2,
        telemetry=False,
        sync_checkpoints=False,
        design_md_in_context=False,
        honest_slot_contract=True,
        gen_steps=2,
        grammar_ltr_max_tokens=32,
        eval_limit=2,
        decode_timeout_seconds=20.0,
        run_class="scratch_matrix",
        **changes,
    )


def train_once(store, config):
    """Reuse validated trial completion when only downstream measurement failed.

    This is an activity effect, not a second scheduler. Runtime still owns the
    outer lease. The local lock protects accidental concurrent CLI invocations.
    """
    import fcntl
    import hashlib
    import time
    from dataclasses import replace

    from slm_training.autoresearch.campaign_events import lock_learning_trial
    from slm_training.harnesses.model_build import train
    from slm_training.harness_core.checkpoint_bundle import validate_bundle
    from slm_training.harnesses.model_build.data import load_train_records
    from slm_training.versioning import build_version_stamp

    config.run_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "config": config_digest(config),
        "training": content_digest(
            [r.to_dict() for r in load_train_records(config.train_dir)]
        ),
        "parent": hashlib.sha256(
            Path(config.initialize_from or config.resume_from).read_bytes()
        ).hexdigest()
        if config.initialize_from or config.resume_from
        else None,
        "versions": build_version_stamp("harness.model_build.train", "model.twotower")[
            "components"
        ],
        "context_layout_contract": "frozen_ancestor/v1",
    }
    with (config.run_dir / ".learning-trial.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        artifact = lock_learning_trial(
            store, config.run_id, inputs, resumable=bool(config.resume_from)
        )
        summary_path = config.run_dir / "train_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            checkpoint = Path(summary["checkpoint"])
            validate_bundle(checkpoint.parent.parent.parent, checkpoint.parent.name)
            if summary["steps"] == config.steps:
                return summary
            if (
                summary["steps"] > config.steps
                or summary.get("stopped_on") != "wall_time_budget"
            ):
                raise ValueError(
                    "trial stopped outside its declared continuation contract"
                )
            if not (config.checkpoint_dir / "last_full_state.pt").is_file():
                raise ValueError("incomplete trial lacks exact continuation state")
        state = config.checkpoint_dir / "last_full_state.pt"
        resumed = (
            replace(config, resume_from=state, initialize_from=None)
            if state.exists()
            else config
        )
        started = time.monotonic()
        try:
            model = None
            if resumed.initialize_from:
                from slm_training.models.twotower import TwoTowerModel

                # Data arms must not buy extra context embeddings. Unknown input
                # tokens remain unknown under the shared ancestor's vocabulary.
                model = TwoTowerModel.from_checkpoint(
                    resumed.initialize_from, device=resumed.device
                )
            return train(resumed, model=model)
        finally:
            store.append_event(
                "learning_training_attempt_observed",
                detail={
                    "input_sha256": artifact.stem,
                    "run_id": config.run_id,
                    "spent_seconds": time.monotonic() - started,
                    "summary_present": summary_path.exists(),
                },
            )


def config_digest(config):
    return content_digest(json.loads(json.dumps(asdict(config), default=str)))


def validate_comparison(manifest, configurations):
    if manifest.experiment_id != EXPERIMENT_ID or manifest.claim_class != "fixture":
        raise ValueError("corrective comparison is registered fixture-only")
    if len(configurations) != 2 or len(manifest.arms) != 2:
        raise ValueError("exactly one original and one corrective-mixture arm required")
    if set(configurations) != {arm.arm_id for arm in manifest.arms}:
        raise ValueError("configured arms differ from the locked design")
    first, second = configurations.values()
    if not first.initialize_from or first.initialize_from != second.initialize_from:
        raise ValueError("same declared initialization checkpoint required")
    allowed = {"train_dir", "run_id"}
    left, right = asdict(first), asdict(second)
    mismatched = [key for key in left if key not in allowed and left[key] != right[key]]
    if mismatched:
        raise ValueError("non-treatment variables differ: " + ",".join(mismatched))
    if first.train_dir == second.train_dir:
        raise ValueError("data intervention requires distinct admitted snapshots")
    if (
        first.device != "cpu"
        or first.context_backend != "scratch"
        or first.sync_checkpoints
    ):
        raise ValueError(
            "local fixture grant allows CPU scratch and no external writes"
        )
    for arm in manifest.arms:
        if arm.config_sha256 != config_digest(configurations[arm.arm_id]):
            raise ValueError("arm configuration differs from preregistration")


def run_comparison(store, manifest, configurations):
    """Run both declared arms once under the caller's bounded invocation.

    A user-facing CLI registration delegates here; no new store or scheduler.
    Positive-control or oracle labels are never a model capability certificate.
    """
    from slm_training.data.readiness_lineage import leakage_findings
    from slm_training.evals.choice_proxy import evaluate_choice_proxy
    from slm_training.evals.denoising_nll import DenoisingNLLConfig
    from slm_training.evals.loss_suites import (
        evaluate_loss_suites,
        write_loss_suite_report,
    )
    from slm_training.harnesses.model_build import evaluate_suites
    from slm_training.harnesses.model_build.data import (
        load_suite_records,
        load_train_records,
    )
    from slm_training.models.twotower import TwoTowerModel
    from slm_training.versioning import build_version_stamp

    validate_comparison(manifest, configurations)
    lock = store.lock_experiment_campaign(manifest)
    rows = {}
    for arm_id, config in configurations.items():
        training = load_train_records(config.train_dir)
        scored = load_suite_records(config.test_dir, config.suite)
        scored = scored[: config.eval_limit]
        if (
            content_digest([r.to_dict() for r in scored])
            != manifest.locked_eval_manifest_sha256
        ):
            raise ValueError("selected evaluation changed after preregistration")
        if leakage_findings(training, {config.suite: scored}):
            raise ValueError("corrective comparison leaks into selected evaluation")
        summary = train_once(store, config)
        checkpoint = Path(summary["checkpoint"])
        model = TwoTowerModel.from_checkpoint(checkpoint, device=config.device)
        nll_config = DenoisingNLLConfig(
            mask_seed=config.seed, compute_legal_support=False
        )
        loss = evaluate_loss_suites(
            model,
            config.test_dir,
            base_suite=config.suite,
            limit=config.eval_limit,
            nll_config=nll_config,
        )
        loss["choice_proxy"] = evaluate_choice_proxy(model, scored, config=nll_config)
        write_loss_suite_report(config.run_dir / "loss_suites.json", loss)
        decoded = evaluate_suites(
            config,
            [config.suite],
            checkpoint=checkpoint,
            partial_scoreboard=True,
            resume_from=config.run_dir,
        )
        rows[arm_id] = {
            "training": summary,
            "loss": loss,
            "decoded": decoded,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
    control, candidate = rows.values()
    if control["trainable_parameters"] != candidate["trainable_parameters"]:
        raise ValueError("data arms are not parameter matched")
    from slm_training.autoresearch.paired_stats import (
        PairedSelection,
        paired_record_screening,
    )
    from slm_training.evals.loss_suites import per_record_nll_map
    from slm_training.evals.measurement_identity import selected_identity

    selection = selected_identity(scored)
    paired = paired_record_screening(
        per_record_nll_map(control["loss"]),
        per_record_nll_map(candidate["loss"]),
        selection=PairedSelection(
            record_ids=selection["selected_record_ids"],
            root_ids=dict(
                zip(
                    selection["selected_record_ids"],
                    selection["selected_root_ids"],
                    strict=True,
                )
            ),
        ),
    )
    report = {
        "experiment_id": EXPERIMENT_ID,
        "manifest_sha256": lock.manifest_sha256,
        "resource_basis": "matched_optimizer_updates_and_examples_per_update",
        "arms": rows,
        "promotion": False,
        "ship_eligible": False,
        "paired_diagnostic": paired,
        "decoded_probe_complete": all(
            row["decoded"].get("measurement_complete") is True for row in rows.values()
        ),
        "independence": "conditional_on_shared_initialization; fixture cases",
        "version_stamp": build_version_stamp(
            "harness.model_build.eval", "evals.loss_suite"
        ),
    }
    artifact = store.write_artifact("learner_correction_result", report)
    store.append_event(
        "learner_correction_completed",
        experiment_id=EXPERIMENT_ID,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={"manifest_sha256": lock.manifest_sha256, "promotion": False},
    )
    return report


def comparison_manifest(
    store, prepared, configurations, *, experiment_id=EXPERIMENT_ID
):
    from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
    from slm_training.versioning import build_version_stamp

    try:
        return store.load_experiment_campaign(experiment_id).manifest
    except FileNotFoundError:
        pass
    return ExperimentCampaignV1(
        campaign_id=store.campaign_id,
        experiment_id=experiment_id,
        hypothesis=(
            "Corrective train-only states change fixture denoising loss."
            if experiment_id == EXPERIMENT_ID
            else "Short-run rankings predict longer-run fixture loss and semantic outcomes."
        ),
        decision="Diagnostic only; no promotion or objective adoption.",
        endpoints=[
            dict(
                endpoint_id="loss",
                metric="masked_denoising_ce",
                role="primary",
                direction="decrease",
                minimum_effect=0,
            )
        ],
        arms=[
            dict(
                arm_id=name,
                role="control" if index == 0 else "candidate",
                config_sha256=config_digest(cfg),
            )
            for index, (name, cfg) in enumerate(configurations.items())
        ],
        seeds=[7301],
        budget=dict(
            max_experiments=len(configurations), max_gpu_hours=0, max_wall_minutes=3
        ),
        selection_rule="best_by_primary_then_smallest"
        if len(configurations) > 2
        else None,
        stopping_rules=[
            "Fixed declared updates per arm; separately bounded resumable invocations; missing decode remains incomplete"
        ],
        controls=[
            dict(
                control_id="original",
                kind="negative",
                description="Original data without corrective states",
            )
        ],
        negative_controls=["original"],
        multiplicity_families=[
            dict(family_id="fixture", hypothesis_ids=["loss"], alpha=0.05)
        ],
        promotion_gates=[
            dict(
                gate_id="unused_fixture", endpoint_id="loss", operator="ge", threshold=0
            )
        ],
        rollback_gates=[
            dict(
                gate_id="invalid_fixture",
                endpoint_id="loss",
                operator="lt",
                threshold=0,
            )
        ],
        artifact_requirements=[
            dict(kind="paired_examples"),
            dict(kind="agentv"),
            dict(kind="agentevals"),
        ],
        claim_class="fixture",
        created_at=prepared["bootstrap"]["finished_at"],
        source_commit=build_version_stamp()["code_commit"],
        source_dirty=True,
        author="local explicit fixture comparison",
        locked_eval_manifest_sha256=prepared["selection"]["selection_sha256"],
    )
