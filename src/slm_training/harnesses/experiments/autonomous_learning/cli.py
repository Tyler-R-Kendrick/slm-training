"""Finite local SEARCH/LEARNING experiments through the canonical model harness.

Explicit CLI activation only. The fixed public AST-copy fixtures measure wiring,
not secret holdout generalization. Each invocation runs one resumable activity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from slm_training.autoresearch.storage import CampaignStore
from slm_training.evals.measurement_identity import content_digest, selected_identity

from .learner_correction import (
    fixture_config as config,
)


def _write(path, value):
    CampaignStore._replace_durable(
        path,
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str)
        + "\n",
    )


def prepare(root, store, *, fidelity_initialization="ancestor"):
    from slm_training.data.readiness_lineage import leakage_findings
    from slm_training.harnesses.model_build import train
    from slm_training.harnesses.train_data.learner_corrections import (
        CopyProgramOracle,
        LearnerState,
        copy_fixture_records,
        corrective_mixture,
        publish_fixture,
    )
    from slm_training.models.twotower import TwoTowerModel

    training, scored = copy_fixture_records()
    if leakage_findings(training, {"smoke": scored}):
        raise ValueError("fixed fixture violates canonical leakage checks")
    procedure = {
        "schema_version": "learning_fixture/v2",
        "seed": 7301,
        "training": [r.to_dict() for r in training],
        "evaluation": [r.to_dict() for r in scored],
        "bootstrap_updates": 1,
        "corrective_updates": 2,
        "fidelity_updates": [1, 4],
        "fidelity_initialization": fidelity_initialization,
        "learning_rates": {"fast": 0.03, "steady": 0.003, "slow": 0.0003},
        "total_optimizer_updates": 17,
        "choice_diagnostic": "exhaustive_single_hole_with_context_controls/v1",
        "context_layout_contract": "frozen_ancestor/v1",
        "promotion_eligible": False,
    }
    frozen = store.write_artifact("learning_fixture_plan", procedure)
    store.append_event(
        "learning_fixture_locked",
        artifact_sha256=frozen.stem,
        idempotency_key="learning-fixture-plan",
    )
    root.mkdir(parents=True, exist_ok=True)
    train_dir = publish_fixture(
        root, root.name + "-original", training, "train", write_json=_write
    )
    test_dir = publish_fixture(
        root, root.name + "-scored", scored, "eval", write_json=_write
    )
    base = train(config(root, train_dir, run_id="ancestor", steps=1))
    checkpoint = Path(base["checkpoint"])
    model = TwoTowerModel.from_checkpoint(checkpoint, device="cpu")
    model.eval()
    tok = model.tokenizer
    observed = []
    diagnostics = []
    for record in training:
        # Partial assignments, not a fictional autoregressive prefix. The task's
        # component position is ambiguous; all supplied context is train-only.
        canvas = tok.encode(record.openui, add_special=True)
        component = next(
            i
            for i, value in enumerate(canvas)
            if tok.id_to_token[value] in {"TextContent", "Button"}
        )
        canvas[component] = tok.mask_id
        unknown = tuple(value == tok.mask_id for value in canvas)
        state = LearnerState(
            tuple(canvas),
            unknown,
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            content_digest(tok.id_to_token),
            7301,
        )
        from slm_training.harnesses.distill.choice_diagnostics import rank_single_hole

        prediction, trace = rank_single_hole(
            model,
            record.prompt,
            canvas,
            component,
            context_controls=True,
            placeholders=record.placeholders,
        )
        counts = [
            token
            for row in training
            for token in tok.encode(row.openui, add_special=False)
        ]
        trace["train_only_ranker"] = max(trace["legal_token_ids"], key=counts.count)
        trace["train_only_ranker_data_digest"] = content_digest(
            [r.to_dict() for r in training]
        )
        trace["expert_upper_bound"] = tok.encode(record.openui, add_special=True)[
            component
        ]
        trace["expert_is_deployable"] = False
        diagnostics.append({"id": record.id, **trace})
        observed.append(
            (
                record,
                state,
                prediction,
                CopyProgramOracle(record.openui, record.meta["root_family_id"]),
            )
        )
    mixture, feedback = corrective_mixture(
        training, observed, scored_suites={"smoke": scored}, training_ancestors=training
    )
    _write(
        root / "learner_observations.json",
        [
            {"id": r.id, "state": s.to_dict(), "prediction": p, "target": o.program}
            for r, s, p, o in observed
        ],
    )
    _write(root / "choice_diagnostics.json", diagnostics)
    if not feedback["corrective_n"]:
        _write(root / "no_correction.json", feedback)
        raise ValueError(
            "fixed learner produced no legal incorrect states; no seed search authorized"
        )
    mixture_dir = publish_fixture(
        root,
        root.name + "-corrective",
        mixture,
        "train",
        write_json=_write,
        feedback=feedback,
    )
    result = {
        "procedure_sha256": frozen.stem,
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "mixture_dir": str(mixture_dir),
        "checkpoint": str(checkpoint),
        "bootstrap": base,
        "feedback": feedback,
        "selection": selected_identity(scored),
        "procedure": procedure,
    }
    _write(root / "prepared.json", result)
    return result


def corrective(root, store):
    from .learner_correction import (
        comparison_manifest,
        run_comparison,
    )

    prepared = json.loads((root / "prepared.json").read_text())
    configurations = {
        name: config(
            root,
            prepared[key],
            run_id=name,
            steps=2,
            test_dir=prepared["test_dir"],
            initialize_from=Path(prepared["checkpoint"]),
        )
        for name, key in (("control", "train_dir"), ("candidate", "mixture_dir"))
    }
    report = run_comparison(
        store, comparison_manifest(store, prepared, configurations), configurations
    )
    _write(root / "corrective_result.json", report)
    return report


def fidelity(root, store, treatment, updates, algorithm="rotation"):
    from slm_training.autoresearch.search.allocation import (
        AllocationPlan,
        FidelityObservation,
        contract_digest,
        fidelity_report,
        lock_allocation_plan,
        next_allocations,
        record_fidelity_observation,
    )
    from slm_training.evals.denoising_nll import DenoisingNLLConfig
    from slm_training.evals.loss_suites import (
        evaluate_loss_suites,
        write_loss_suite_report,
    )
    from slm_training.harnesses.model_build import evaluate_suites
    from slm_training.harnesses.model_build.eval_measurement import evaluator_identity
    from slm_training.models.twotower import TwoTowerModel
    from slm_training.versioning import build_version_stamp

    from .learner_correction import (
        comparison_manifest,
        train_once,
    )

    prepared = json.loads((root / "prepared.json").read_text())
    rates = prepared["procedure"]["learning_rates"]
    configurations = {
        name: config(
            root,
            prepared["train_dir"],
            run_id=f"{name}-4",
            steps=4,
            test_dir=prepared["test_dir"],
            lr=rate,
            initialize_from=Path(prepared["checkpoint"])
            if prepared["procedure"].get("fidelity_initialization", "ancestor")
            == "ancestor"
            else None,
        )
        for name, rate in rates.items()
    }
    store.lock_experiment_campaign(
        comparison_manifest(
            store, prepared, configurations, experiment_id="SEA-FIDELITY-1"
        )
    )
    ids = {
        name: content_digest({"lr": rate, "fixture": prepared["procedure_sha256"]})
        for name, rate in rates.items()
    }
    plan = AllocationPlan(
        algorithm=algorithm,
        enabled=True,
        treatment_ids=tuple(ids.values()),
        checkpoints=(1, 4),
        total_updates=12,
        random_seed=7301,
        direction="minimize",
        endpoint_identity=content_digest(
            {
                "metric": "masked_denoising_ce",
                "selection": prepared["selection"],
                "evaluator": evaluator_identity(
                    build_version_stamp(
                        "harness.model_build.eval", "evals.scoring", "evals.loss_suite"
                    )
                ),
            }
        ),
    )
    lock_allocation_plan(store, plan)
    # A scorer successor gets its own evidence namespace; existing trial state
    # remains reusable, but old/new measurements are never pooled.
    observations_dir = root / "fidelity" / contract_digest(plan)
    rows = [
        FidelityObservation.model_validate_json(path.read_text())
        for path in sorted(observations_dir.glob("*.json"))
    ]
    allocations = next_allocations(plan, rows)
    allocation = next(
        (
            a
            for a in allocations
            if a.treatment_id == ids[treatment] and a.to_updates == updates
        ),
        None,
    )
    if allocation is None:
        raise ValueError(
            "requested allocation is not pending under the locked resource plan"
        )
    cfg = config(
        root,
        prepared["train_dir"],
        run_id=f"{treatment}-{updates}",
        steps=updates,
        test_dir=prepared["test_dir"],
        lr=rates[treatment],
    )
    if allocation.from_updates:
        cfg = replace(
            cfg,
            resume_from=root
            / "runs"
            / f"{treatment}-{allocation.from_updates}"
            / "checkpoints/last_full_state.pt",
        )
    elif prepared["procedure"].get("fidelity_initialization", "ancestor") == "ancestor":
        cfg = replace(cfg, initialize_from=Path(prepared["checkpoint"]))
    summary = train_once(store, cfg)
    model = TwoTowerModel.from_checkpoint(Path(summary["checkpoint"]), device="cpu")
    loss = evaluate_loss_suites(
        model,
        cfg.test_dir,
        base_suite="smoke",
        limit=2,
        nll_config=DenoisingNLLConfig(mask_seed=7301, compute_legal_support=False),
    )
    write_loss_suite_report(cfg.run_dir / "loss_suites.json", loss)
    decoded = evaluate_suites(
        cfg, ["smoke"], checkpoint=Path(summary["checkpoint"]), partial_scoreboard=True
    )
    if decoded.get("measurement_complete") is not True:
        raise ValueError("fidelity decoded probe incomplete; no completed observation")
    state = Path(summary["checkpoint"]).parent / "last_full_state.pt"
    observation = FidelityObservation(
        plan_digest=contract_digest(plan),
        treatment_id=ids[treatment],
        updates=updates,
        value=loss["categories"]["broad"]["aggregate"]["mean_nll"],
        cursor_digest=hashlib.sha256(state.read_bytes()).hexdigest(),
        attempt_id=f"{treatment}-{updates}",
        endpoint_identity=plan.endpoint_identity,
    )
    record_fidelity_observation(store, observation)
    observations_dir.mkdir(parents=True, exist_ok=True)
    _write(observations_dir / f"{treatment}-{updates}.json", observation.model_dump())
    report = fidelity_report(plan, [*rows, observation])
    report["version_stamp"] = build_version_stamp()
    report["artifact_root"] = str(root)
    _write(root / "fidelity_result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "corrective", "fidelity"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--fidelity-initialization", choices=("ancestor", "scratch"), default="ancestor"
    )
    parser.add_argument("--enable-fixture-experiment", action="store_true")
    parser.add_argument("--treatment", choices=("fast", "steady", "slow"))
    parser.add_argument("--updates", type=int, choices=(1, 4))
    parser.add_argument(
        "--allocation",
        choices=("rotation", "random", "successive_halving"),
        default="rotation",
    )
    args = parser.parse_args(argv)
    if not args.enable_fixture_experiment:
        parser.error("default-off: explicit --enable-fixture-experiment required")
    from slm_training.data.store import DataStore

    DataStore.validate_id(args.run_id)
    root = (args.run_root / args.run_id).resolve()
    store = CampaignStore(args.run_id)
    if args.phase == "fidelity" and (args.treatment is None or args.updates is None):
        parser.error("fidelity requires --treatment and --updates")
    actions = {
        "prepare": lambda: prepare(
            root, store, fidelity_initialization=args.fidelity_initialization
        ),
        "corrective": lambda: corrective(root, store),
        "fidelity": lambda: fidelity(
            root, store, args.treatment, args.updates, args.allocation
        ),
    }
    result = actions[args.phase]()
    print(
        json.dumps(
            {
                "phase": args.phase,
                "artifact_root": str(root),
                "result_digest": content_digest(result),
                "promotion_eligible": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
