"""Finite six-case remeasurement through compiled CLI commands; never training.

Each `run` invocation executes one bounded workload. A new campaign locks the
current policy before execution; retained measurements supply no verdicts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from slm_training.autoresearch.climb_policy import load_climb_policy, primary_for_role
from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.schemas import CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.store import DataStore
from slm_training.evals.measurement_identity import content_digest, selected_identity
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.harness_core.checkpoint_bundle import validate_bundle
from slm_training.levers import (
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    MAX_RUN_SECONDS,
)
from slm_training.versioning import build_version_stamp

from .cli import _write
from .measurement_fixture_plan import (
    COMPONENTS,
    EXPERIMENT_ID,
    _read,
    _sha,
    compile_fixture,
)


def source_identity():
    from slm_training.harnesses.model_build.eval_measurement import evaluator_identity

    root = Path(__file__).resolve().parents[5]
    paths = [
        Path(__file__),
        Path(__file__).with_name("measurement_fixture_evidence.py"),
        Path(__file__).with_name("measurement_fixture_plan.py"),
        root / "scripts/run_autotrain_continuous.py",
        root / "scripts/autotrain_metrics.py",
        root / "scripts/autotrain_measurement.py",
        root / "scripts/autotrain_nll.py",
        root / "src/slm_training/evals/loss_suites.py",
        root / "src/slm_training/evals/denoising_nll.py",
        root / "src/slm_training/autoresearch/paired_analysis.py",
        root / "src/slm_training/autoresearch/search/evidence.py",
    ]
    stamp = build_version_stamp(*COMPONENTS)
    return {
        "components": stamp["components"],
        "evaluator_sha256": evaluator_identity(stamp),
        "source_files": {str(path.relative_to(root)): _sha(path) for path in paths},
        "policy_sha256": content_digest(load_climb_policy().payload),
    }


def require_release_versions(identity):
    for component, minimum in (("harness.model_build.eval", 109),):
        version = identity["components"][component]
        if not version.startswith("v") or int(version[1:]) < minimum:
            raise ValueError(
                f"release version pending: {component} needs v{minimum}, found {version}"
            )


def _inputs(evidence_path, train_version, eval_version):
    retained = _read(evidence_path)
    data = DataStore()
    train_dir = data.verify("train", train_version).path
    test_dir = data.verify("eval", eval_version).path
    selection = selected_identity(load_suite_records(test_dir, "smoke")[:6])
    if len(selection["selected_record_ids"]) != 6 or selection != retained["selection"]:
        raise ValueError(
            "retained checkpoint remeasurement requires the exact selected six cases"
        )
    arms = {}
    for arm in ("control", "candidate"):
        row = retained["arms"][arm]
        checkpoint = Path(row["checkpoint"]).resolve()
        validate_bundle(checkpoint.parent.parent.parent, checkpoint.parent.name)
        if _sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("retained checkpoint hash mismatch")
        arms[arm] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha(checkpoint),
            "bundle_digest": checkpoint.parent.name,
            "trainable_parameters": row["trainable_parameters"],
        }
    if (
        arms["control"]["trainable_parameters"]
        != arms["candidate"]["trainable_parameters"]
    ):
        raise ValueError("fixture arms must be size matched")
    return {
        "retained_evidence_sha256": _sha(evidence_path),
        "selection": selection,
        "arms": arms,
        "train_version": train_version,
        "eval_version": eval_version,
        "test_dir": str(test_dir.resolve()),
        "train_dir": str(train_dir.resolve()),
        "data_manifests": {
            str(path.resolve()): _sha(path)
            for path in (train_dir / "manifest.json", test_dir / "manifest.json")
        },
    }


def prepare(store, evidence_path, train_version, eval_version):
    identity = source_identity()
    require_release_versions(identity)
    inputs = _inputs(evidence_path, train_version, eval_version)
    primary = dict(primary_for_role(load_climb_policy(), "screening"))
    if primary["metric"] != "smoke.eval_nll" or primary["claim_class"] != "diagnostic":
        raise ValueError(
            "fixture only supports the policy diagnostic smoke.eval_nll endpoint"
        )
    campaign = CampaignSpec(
        campaign_id=store.campaign_id,
        primary_metric=primary["metric"],
        objective="Complete current six-case diagnostic evidence",
        budget=dict(max_experiments=2, max_gpu_hours=0, max_wall_minutes=3),
    )
    arms = {
        arm: compile_fixture(campaign, arm, inputs, output_root=store.root.parent)
        for arm in inputs["arms"]
    }
    manifest = ExperimentCampaignV1(
        campaign_id=store.campaign_id,
        experiment_id=EXPERIMENT_ID,
        hypothesis="Fixed checkpoint arms yield complete current six-case diagnostic evidence.",
        decision="Diagnostic remeasurement only; no confirmation, promotion, shipment or new training.",
        endpoints=[
            dict(
                endpoint_id="primary",
                role="primary",
                **{
                    key: primary[key]
                    for key in ("metric", "direction", "minimum_effect")
                },
            )
        ],
        arms=[
            dict(
                arm_id=arm,
                role=arm,
                config_sha256=content_digest({**value, **inputs["arms"][arm]}),
            )
            for arm, value in arms.items()
        ],
        seeds=[7301],
        budget=campaign.budget,
        stopping_rules=[
            "Exactly eight bounded child invocations; partial decode is pending, never a loss."
        ],
        controls=[
            dict(
                control_id="retained-control",
                kind="negative",
                description="Fixed retained control; repeated evaluation adds no training replicate",
            )
        ],
        negative_controls=["retained-control"],
        multiplicity_families=[
            dict(family_id="public-fixture", hypothesis_ids=["primary"], alpha=0.05)
        ],
        promotion_gates=[
            dict(
                gate_id="fixture_never_promotes",
                endpoint_id="primary",
                operator="lt",
                threshold=0,
            )
        ],
        rollback_gates=[
            dict(
                gate_id="invalid_negative_loss",
                endpoint_id="primary",
                operator="lt",
                threshold=0,
            )
        ],
        artifact_requirements=[
            dict(kind=kind) for kind in ("paired_examples", "agentv", "agentevals")
        ],
        claim_class="fixture",
        source_commit=build_version_stamp()["code_commit"],
        source_dirty=True,
        author="explicit local measurement acceptance",
        locked_eval_manifest_sha256=inputs["selection"]["selection_sha256"],
    )
    lock = store.lock_experiment_campaign(manifest)
    plan = {
        "schema": "measurement_fixture/v1",
        "inputs": inputs,
        "arms": arms,
        "identity": identity,
        "primary": primary,
        "manifest_sha256": lock.manifest_sha256,
        "maximum_child_attempts": 8,
        "maximum_child_seconds": 8 * MAX_RUN_SECONDS,
        "new_training": False,
        "promotion_allowed": False,
        "ship_eligible": False,
        "version_stamp": build_version_stamp(*COMPONENTS),
    }
    path = store.write_artifact("measurement_fixture_plan", plan)
    store.append_event(
        "measurement_fixture_locked",
        artifact_sha256=path.stem,
        experiment_id=EXPERIMENT_ID,
        idempotency_key="measurement-fixture-lock",
    )
    _write(store.root / "measurement_fixture.json", plan)
    return plan


def load_plan(store):
    plan = _read(store.root / "measurement_fixture.json")
    events = store.verify_event_chain()
    locks = [
        event for event in events if event["event_type"] == "measurement_fixture_locked"
    ]
    from slm_training.lineage.records import canonical_json

    digest = hashlib.sha256(canonical_json(plan).encode()).hexdigest()
    if len(locks) != 1 or locks[0]["artifact_sha256"] != digest:
        raise ValueError("fixture plan differs from its canonical event lock")
    if plan["identity"] != source_identity():
        raise ValueError(
            "source/policy/version changed; prepare a new measurement campaign"
        )
    for path, digest in plan["inputs"]["data_manifests"].items():
        if _sha(path) != digest:
            raise ValueError("dataset manifest changed")
    for arm in plan["inputs"]["arms"].values():
        checkpoint = Path(arm["checkpoint"])
        validate_bundle(checkpoint.parent.parent.parent, arm["bundle_digest"])
    selected = selected_identity(
        load_suite_records(Path(plan["inputs"]["test_dir"]), "smoke")[:6]
    )
    if selected != plan["inputs"]["selection"]:
        raise ValueError("selected cases changed")
    return plan


def run_activity(store, activity):
    plan = load_plan(store)
    arm, workload = activity.split("-", 1)
    allowed = [
        f"{a}-{w}"
        for a in plan["arms"]
        for w in ("loss", "decode1", "decode2", "decode3")
    ]
    if activity not in allowed:
        raise ValueError("unknown finite fixture activity")
    receipts = store.root / "measurement_receipts"
    receipts.mkdir(exist_ok=True)
    # Local accidental-concurrency exclusion, not a multi-host lease claim.
    with (receipts / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = receipts / f"{activity}.json"
        reservation = receipts / f"{activity}.started.json"
        if receipt.exists() or reservation.exists():
            raise ValueError(
                "activity already attempted; cannot silently reset its resource budget"
            )
        if workload.startswith("decode") and workload != "decode1":
            prior = _read(receipts / f"{arm}-decode{int(workload[-1]) - 1}.json")
            if prior["result"]["returncode"] != 10:
                raise ValueError(
                    "previous decode chunk did not yield pending measurement"
                )
        command = plan["arms"][arm]["commands"][
            "loss" if workload == "loss" else "decode"
        ]
        _write(reservation, {"activity": activity, "reserved_seconds": MAX_RUN_SECONDS})
        result = run_bounded_process(
            command,
            interrupt_after_seconds=INTERRUPT_AFTER_SECONDS,
            kill_grace_seconds=KILL_GRACE_SECONDS,
        )
        payload = {
            "activity": activity,
            "manifest_sha256": plan["manifest_sha256"],
            "result": asdict(result),
            "evidence_class": "actual_canonical_cli",
        }
        _write(receipt, payload)
        artifact = store.write_artifact("measurement_fixture_attempt", payload)
        store.append_event(
            "measurement_fixture_attempt",
            artifact_sha256=artifact.stem,
            experiment_id=EXPERIMENT_ID,
            detail={"activity": activity},
        )
        expected = (
            {0} if workload == "loss" else ({0, 8} if workload == "decode3" else {10})
        )
        if result.returncode not in expected or result.timed_out or result.interrupted:
            raise ValueError(
                f"{activity} failed: {result.outcome}, exit {result.returncode}"
            )
        return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "collect"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/autoresearch")
    )
    parser.add_argument(
        "--retained-evidence",
        type=Path,
        default=Path("outputs/runs/autonomy-mea-data/final-evidence.json"),
    )
    parser.add_argument("--train-version", default="autonomy-mea-train-admitted-v1")
    parser.add_argument("--eval-version", default="autonomy-mea-smoke-successor-v1")
    parser.add_argument(
        "--activity",
        choices=tuple(
            f"{a}-{w}"
            for a in ("control", "candidate")
            for w in ("loss", "decode1", "decode2", "decode3")
        ),
    )
    parser.add_argument("--enable-fixture-experiment", action="store_true")
    args = parser.parse_args(argv)
    if not args.enable_fixture_experiment:
        parser.error("explicit --enable-fixture-experiment required")
    DataStore.validate_id(args.run_id)
    store = CampaignStore(args.run_id, args.output_root)
    if args.action == "prepare":
        result = prepare(
            store, args.retained_evidence, args.train_version, args.eval_version
        )
    elif args.action == "run":
        if args.activity is None:
            parser.error("run requires --activity")
        result = run_activity(store, args.activity)
    else:
        from .measurement_fixture_evidence import collect

        result = collect(store, load_plan(store))
    print(
        json.dumps(
            {
                "action": args.action,
                "result_sha256": content_digest(result),
                "campaign": str(store.root),
                "promotion_allowed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
