"""Typed measurement-only commands compiled by the canonical research owner."""

import sys
import hashlib
import json
from pathlib import Path

from slm_training.autoresearch.engine import compile_commands
from slm_training.autoresearch.schemas import ExperimentSpec

EXPERIMENT_ID = "MEA-SIX-RESOLVED-ENDPOINT"
COMPONENTS = (
    "harness.model_build.eval",
    "evals.scoring",
    "evals.loss_suite",
    "harness.experiments",
    "model.twotower",
)


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supervised_matrix(campaign, arms, citation):
    """A typed two-arm screen, not five administratively invented hypotheses."""
    from slm_training.autoresearch.schemas import HypothesisMatrix

    novelty = dict(
        transition_kind="fixed_regime_search",
        old_schema_elements=["direct measurement"],
        proposed_schema_elements=["supervised remeasurement"],
        transported_elements=["frozen public cases"],
        transport_analysis=["Same retained weights; no new training or independence"],
        residual_elements=["supervisor/cursor integration"],
        preservation_checks=["manifest and endpoint binding"],
        stress_tests=["partial decode resumes in the same campaign"],
        worthiness_criteria=["complete evidence without new promotion authority"],
    )
    return HypothesisMatrix(
        matrix_id="supervised-measurement",
        campaign_id=campaign.campaign_id,
        evidence_snapshot_id="retained-public-fixture",
        matrix_role="screen",
        hypotheses=[
            dict(
                experiment=arm["experiment"],
                novelty=novelty,
                evidence_uses=[
                    dict(
                        role="prior_trace",
                        citation=citation,
                        contribution="Original immutable weights and input identities; historical verdict not reused",
                    )
                ],
            )
            for arm in arms.values()
        ],
        recommended_experiment_id="candidate",
        selection_rationale="Predeclared two-arm integration fixture; no adaptive model selection",
    )


def compile_fixture(campaign, arm, inputs, *, output_root):
    experiment = ExperimentSpec(
        experiment_id=arm,
        campaign_id=campaign.campaign_id,
        hypothesis="The complete six-case diagnostic reaches the current disposition owner.",
        rationale="Remeasure fixed public cases; historical verdicts are not imported.",
        expected_effect="No model gain is assumed.",
        citations=(__file__,),
        falsification_criteria=(
            "Missing rows, incompatible identities or missing AgentV",
        ),
        stop_conditions=("Fixed two-arm eight-command budget; no promotion",),
        knobs=dict(
            train_version=inputs["train_version"],
            eval_version=inputs["eval_version"],
            seed=7301,
            steps=3,
            batch_size=2,
            context_backend="scratch",
            local_files_only=True,
            sync_checkpoints=False,
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            allow_unconstrained_fallback=False,
            eval_suites="smoke",
            eval_limit=6,
            eval_partial_scoreboard=True,
            eval_max_records_this_run=2,
        ),
    )
    compiled = compile_commands(campaign, experiment, output_root=output_root)
    decode = next(
        list(command) for command in compiled if "scripts.evaluate_model" in command
    )
    run_dir = Path(output_root) / campaign.campaign_id / "runs" / arm
    checkpoint = inputs["arms"][arm]["checkpoint"]
    decode.extend(
        [
            "--checkpoint",
            checkpoint,
            "--generate-batch-size",
            "2",
            "--resume-run",
            str(run_dir),
        ]
    )
    loss = [
        sys.executable,
        "-m",
        "scripts.evaluate_loss_suites",
        "--checkpoint",
        checkpoint,
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
    return {
        "experiment": experiment.model_dump(mode="json"),
        "compiled": compiled,
        "commands": {"loss": loss, "decode": decode},
        "run_dir": str(run_dir),
        "training_disposition": "not_executed_retained_validated_checkpoint",
    }
