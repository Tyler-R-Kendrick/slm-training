"""Verify and consume the finite measurement fixture's actual CLI artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.autotrain_measurement import measurement_is_complete
from scripts.autotrain_metrics import read_paired_nll
from scripts.autotrain_nll import run_arm_eval_nll
from slm_training.autoresearch.climb_policy import screening_nll_definition_hash
from slm_training.evals.loss_suites import per_record_nll_map, per_record_nll_rows
from slm_training.evals.measurement_identity import content_digest
from slm_training.harnesses.model_build.eval_measurement import (
    evaluator_identity,
    suite_result_cacheable,
)
from slm_training.versioning import build_version_stamp

from .cli import _write
from .measurement_fixture_plan import COMPONENTS, EXPERIMENT_ID, _read, _sha


def recording_runtime(store):
    """Observe the actual controller process, without substituting driver/results."""
    from dataclasses import asdict
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime

    class RecordingRuntime(ActivityRuntime):
        def run(self, lease, command, **kwargs):
            result = super().run(lease, command, **kwargs)
            artifact = self.store.write_artifact(
                "measurement_controller_process",
                {
                    "lease": lease.model_dump(mode="json"),
                    "command": command,
                    "result": asdict(result),
                    "evidence_class": "actual_supervisor_process",
                },
            )
            self.store.append_event(
                "measurement_controller_observed",
                experiment_id=lease.activity_id,
                artifact_sha256=artifact.stem,
            )
            return result

    return RecordingRuntime(store)


def verify_agentv(payload):
    """SDK execution success is distinct from passing the domain/ship criteria."""
    summary = payload.get("summary", {})
    errors = (
        payload.get("runner", {}).get("execution_errors"),
        summary.get("executionErrors"),
    )
    if (
        payload.get("sdk") != "@agentv/core"
        or any(type(value) is not int or value != 0 for value in errors)
        or type(summary.get("total")) is not int
        or summary["total"] <= 0
    ):
        raise ValueError("AgentV SDK execution missing or failed")
    if not {"runDir", "benchmarkPath", "indexPath", "timingPath"} <= set(
        payload.get("artifacts", {})
    ):
        raise ValueError("AgentV artifact manifest incomplete")
    paths = {"spec": payload["spec"], **payload["artifacts"]}
    artifacts = {}
    for name, value in paths.items():
        path = Path(value)
        if name == "runDir":
            if not path.is_dir():
                raise ValueError("AgentV run directory missing")
        elif not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"AgentV artifact missing: {name}")
        else:
            artifacts[str(path)] = _sha(path)
    for name in ("spec", "indexPath"):
        rows = [
            json.loads(line)
            for line in Path(paths[name]).read_text().splitlines()
            if line.strip()
        ]
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("AgentV contains no executed cases")
    return artifacts


def _current_stamp(payload, required):
    components = payload["version_stamp"]["components"]
    if (
        not set(required) <= set(components)
        or components != build_version_stamp(*components)["components"]
    ):
        raise ValueError("stale or incomplete evaluator version stamp")


def checked_arm(plan, arm):
    run_dir = Path(plan["arms"][arm]["run_dir"])
    scoreboard = _read(run_dir / "scoreboard.json")
    metrics = scoreboard["suites"]["smoke"]
    loss = _read(run_dir / "loss_suites.json")
    _current_stamp(
        metrics, ("harness.model_build.eval", "evals.scoring", "model.twotower")
    )
    _current_stamp(loss, ("evals.loss_suite", "model.twotower"))
    if metrics["evaluator_sha256"] != evaluator_identity(metrics["version_stamp"]):
        raise ValueError("decoded evaluator identity is stale")
    selection = plan["inputs"]["selection"]
    if (
        not suite_result_cacheable(metrics)
        or metrics["document_n"] != 6
        or scoreboard.get("measurement_complete") is not True
        or scoreboard.get("publication_complete") is not True
    ):
        raise ValueError("complete six-case decoded scoreboard required")
    if loss["selection"] != selection or any(
        metrics.get(key) != value for key, value in selection.items()
    ):
        raise ValueError("decoded/loss selection differs from the locked cases")
    if metrics["checkpoint_sha256"] != plan["inputs"]["arms"][arm]["checkpoint_sha256"]:
        raise ValueError("decoded checkpoint differs from locked arm")
    if Path(loss["checkpoint"]).resolve() != Path(
        plan["inputs"]["arms"][arm]["checkpoint"]
    ):
        raise ValueError("loss checkpoint differs from locked arm")
    # Reinvoke the real strict producer; duplicates cannot be hidden by maps.
    if per_record_nll_rows(loss["categories"]) != loss["per_record"]:
        raise ValueError(
            "flattened loss evidence differs from its category observations"
        )
    rows = per_record_nll_map(loss)
    broad = loss["categories"]["broad"]["per_record"]
    if set(rows) != set(selection["selected_record_ids"]) or len(broad) != 6:
        raise ValueError("broad loss does not cover exactly the selected six cases")
    by_id = {row["id"]: row for row in broad}
    for index, case in enumerate(selection["selected_record_ids"]):
        row = by_id[case]
        expected = {
            "case_id": case,
            "root_id": selection["selected_root_ids"][index],
            "input_sha256": selection["input_sha256s"][index],
            "selection_sha256": selection["selection_sha256"],
            "seed": 7301,
            "estimator_id": loss["estimator_id"],
            "evaluator_sha256": plan["identity"]["source_files"][
                "src/slm_training/evals/denoising_nll.py"
            ],
            "units": "nats_per_masked_token",
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("loss row identity differs from locked measurement")
    hashes = {**verify_agentv(scoreboard["evals"]), **verify_agentv(loss["agentv"])}
    return {
        "metrics": metrics,
        "loss": loss,
        "records": rows,
        "agentv_artifacts": hashes,
        "run_dir": run_dir,
    }


def _receipts(store, plan):
    from slm_training.lineage.records import canonical_json
    import hashlib

    events = store.verify_event_chain()
    receipts = {}
    for arm in plan["arms"]:
        for workload in ("loss", "decode1", "decode2", "decode3"):
            activity = f"{arm}-{workload}"
            path = store.root / "measurement_receipts" / f"{activity}.json"
            receipt = _read(path)
            digest = hashlib.sha256(canonical_json(receipt).encode()).hexdigest()
            if not any(
                event["event_type"] == "measurement_fixture_attempt"
                and event["artifact_sha256"] == digest
                for event in events
            ):
                raise ValueError("receipt lacks canonical attempt event")
            result = receipt["result"]
            expected = (
                {0}
                if workload == "loss"
                else ({0, 8} if workload == "decode3" else {10})
            )
            if result["returncode"] not in expected or result["outcome"] != "completed":
                raise ValueError(
                    "required workload did not complete its declared contract"
                )
            receipts[str(path)] = {
                "sha256": _sha(path),
                "seconds": result["duration_seconds"],
            }
    return receipts


def collect(store, plan):
    from scripts.run_autotrain_continuous import _classify_positive

    receipts = _receipts(store, plan)
    arms = {arm: checked_arm(plan, arm) for arm in plan["arms"]}
    definition = content_digest(
        {
            "screening_definition": screening_nll_definition_hash(),
            "loss_definition": arms["control"]["loss"]["definition"],
        }
    )
    if arms["control"]["loss"]["definition"] != arms["candidate"]["loss"]["definition"]:
        raise ValueError("paired loss definitions differ")
    for arm in arms.values():
        loss = arm["loss"]
        run_arm_eval_nll(
            arm["run_dir"],
            {
                "eval_nll": loss["categories"]["broad"]["aggregate"]["mean_nll"],
                "records": arm["records"],
                "selection": loss["selection"],
                "row_evidence": loss["per_record"],
                "estimator_id": loss["estimator_id"],
                "definition_hash": definition,
                "eval_version": plan["inputs"]["eval_version"],
            },
        )
    paired, info, failures = read_paired_nll(
        arms["control"]["run_dir"], arms["candidate"]["run_dir"]
    )
    if (
        failures
        or paired is None
        or len(paired["control"]) != 6
        or len(paired["candidate"]) != 6
    ):
        raise ValueError(
            "canonical NLL attachment did not produce complete paired evidence"
        )
    decision = _classify_positive(
        camp_dir=store.root,
        primary_metric=plan["primary"]["metric"],
        control_id="control",
        candidate_id="candidate",
        role="screening",
        baseline_trainable_params=plan["inputs"]["arms"]["control"][
            "trainable_parameters"
        ],
        candidate_trainable_params=plan["inputs"]["arms"]["candidate"][
            "trainable_parameters"
        ],
        observed_sd_path=store.root / "fixture_observed_sd.json",
    )
    if decision["primary_metric"] != plan["primary"][
        "metric"
    ] or not measurement_is_complete(decision):
        raise ValueError("current disposition refused locked completed evidence")
    result = {
        "schema": "measurement_fixture_result/v1",
        "manifest_sha256": plan["manifest_sha256"],
        "selection": plan["inputs"]["selection"],
        "decision": decision,
        "paired_counts": info,
        "command_receipts": receipts,
        "charged_child_seconds": sum(r["seconds"] for r in receipts.values()),
        "arms": {
            name: {
                "agentv_artifacts": arm["agentv_artifacts"],
                "meaningful_program_rate": arm["metrics"]["meaningful_program_rate"],
                "checkpoint_sha256": arm["metrics"]["checkpoint_sha256"],
                "trainable_parameters": plan["inputs"]["arms"][name][
                    "trainable_parameters"
                ],
                "scoreboard": str(arm["run_dir"] / "scoreboard.json"),
                "loss_report": str(arm["run_dir"] / "loss_suites.json"),
            }
            for name, arm in arms.items()
        },
        "evidence_class": "actual_checkpoint_remeasurement_and_canonical_disposition",
        "new_training": False,
        "diagnostic_complete": True,
        "decoded_probe_complete": True,
        "confirmation_complete": False,
        "promotion_allowed": False,
        "ship_eligible": False,
        "scope": "six public regression roots; conditional on retained initializations; no trained capability claim",
        "version_stamp": build_version_stamp(*COMPONENTS),
    }
    artifact = store.write_artifact("measurement_fixture_result", result)
    store.append_event(
        "measurement_fixture_consumed",
        artifact_sha256=artifact.stem,
        experiment_id=EXPERIMENT_ID,
        idempotency_key="measurement-fixture-consumed",
    )
    _write(store.root / "measurement_result.json", result)
    write_result_docs(store.campaign_id, result)
    return result


def project_paths(value):
    """Portable documentation projection; original controller evidence is untouched."""
    if isinstance(value, str):
        prefix = str(Path.cwd().resolve()) + "/"
        return value.removeprefix(prefix)
    if isinstance(value, list):
        return [project_paths(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key = project_paths(key)
            if key in result:
                raise ValueError("portable artifact paths collide")
            result[key] = project_paths(item)
        return result
    return value


def write_result_docs(campaign_id, result):
    """One content-bound closeout per explicitly named finite fixture run."""
    from slm_training.autoresearch.storage import CampaignStore

    prefix = Path("docs/design") / f"autonomy-measurement-{campaign_id}"
    json_path = prefix.with_suffix(".json")
    result = project_paths(result)
    if json_path.exists() and project_paths(_read(json_path)) != result:
        raise ValueError("refusing to overwrite another measured-result identity")
    _write(json_path, result)
    paired = result["decision"].get("paired_test", {})
    sizes = sorted({arm["trainable_parameters"] for arm in result["arms"].values()})
    text = (
        f"# Finite measurement acceptance: {campaign_id}\n\n"
        f"Machine evidence: [{json_path.name}]({json_path.name}).\n\n"
        f"CPU inference on two retained scratch TwoTower checkpoints, trainable parameter counts {sizes}; "
        "no new training. Six locked public regression cases per arm, "
        "mask seed 7301, constrained tree decode in three two-case chunks. "
        "Actual AgentV execution accompanies loss and completed decode.\n\n"
        f"Locked primary: `{result['decision']['primary_metric']}`; "
        f"diagnostic positive: `{result['decision'].get('positive')}`. "
        f"Paired diagnostic: `{json.dumps(paired, sort_keys=True)}`.\n\n"
        f"Charged child seconds: {result['charged_child_seconds']:.6f}. "
        "Each child obeys the canonical interrupt and kill-grace limits.\n\n"
        "This remeasurement does not accept the historical mismatched endpoint lock, "
        "add independent training replicates, confirm improvement, promote a champion, "
        "or authorize shipping. Full production ship suites were not run. "
        "No service or remote delivery was activated.\n"
    )
    CampaignStore._replace_durable(prefix.with_suffix(".md"), text)
