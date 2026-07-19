#!/usr/bin/env python3
"""Build the preregistered SDE4-01 (SLM-179) scaffold-distillation activation manifest.

Example::

    python -m scripts.build_scaffold_distillation_activation_manifest \
        --manifest-id scaffold-distillation-activation-20260719 \
        --teacher-checkpoint-id teacher/checkpoint \
        --teacher-run-id teacher/run \
        --trace-store-uri memory://traces \
        --scaffold-config-hash abcdef \
        --gate-slm-161 --gate-slm-162 --gate-slm-168 \
        --gate-scaffold-value --gate-latency --gate-budget \
        --total-dollars 1000 \
        --scaffold-decomposition value_demonstrated \
        --out-json docs/design/iter-scaffold-distillation-activation-20260719.json \
        --out-md docs/design/iter-scaffold-distillation-activation-20260719.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.scaffold_distillation_activation import (
    DEFAULT_ACTIVATION_GATES,
    DEFAULT_ARMS,
    ActivationGate,
    BudgetCap,
    ScaffoldDistillationActivationManifest,
    TeacherTraceContract,
    build_scaffold_distillation_activation_manifest,
    validate_scaffold_distillation_activation_manifest,
)
from slm_training.versioning import build_version_stamp


def _make_gates(args: argparse.Namespace) -> tuple[ActivationGate, ...]:
    flag_map = {
        "slm161_machine_readable_decomposition": args.gate_slm_161,
        "slm162_metric_gaming_suite": args.gate_slm_162,
        "slm168_public_structured_contract_pointer": args.gate_slm_168,
        "scaffold_value_demonstrated": args.gate_scaffold_value,
        "latency_or_complexity_worth_amortizing": args.gate_latency,
        "budget_approved": args.gate_budget,
    }
    return tuple(
        ActivationGate(
            gate_id=g.gate_id,
            depends_on_issue_id=g.depends_on_issue_id,
            required_status=g.required_status,
            available=flag_map.get(g.gate_id, False),
            evidence=g.evidence,
        )
        for g in DEFAULT_ACTIVATION_GATES
    )


def _markdown(manifest: ScaffoldDistillationActivationManifest) -> str:
    lines = [
        "# SDE4-01 (SLM-179) scaffold-distillation activation manifest",
        "",
        f"**manifest_id:** `{manifest.manifest_id}`  ",
        f"**schema_version:** `{manifest.schema_version}`  ",
        f"**hypothesis_id:** `{manifest.hypothesis_id}`  ",
        f"**activation_status:** `{manifest.activation_status}`  ",
        f"**activation_verdict:** `{manifest.activation_verdict}`  ",
        f"**campaign_verdict:** `{manifest.campaign_verdict}`  ",
        f"**manifest_hash:** `{manifest.manifest_hash}`  ",
        f"**primary_metric:** `{manifest.primary_metric}`  ",
        f"**max_attempts_for_teacher:** {manifest.max_attempts_for_teacher}  ",
        f"**seeds:** {list(manifest.seeds)}",
        "",
        "## Activation gates",
        "",
        "| gate_id | depends_on_issue_id | required_status | available | evidence |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for gate in manifest.activation_gates:
        lines.append(
            f"| {gate.gate_id} | {gate.depends_on_issue_id or ''} | "
            f"{gate.required_status or ''} | {gate.available} | {gate.evidence or ''} |"
        )

    lines += [
        "",
        "## Budget cap",
        "",
        "| teacher_trace_compute_dollars | student_training_dollars | "
        "student_training_gpu_hours | eval_dollars | total_dollars |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {manifest.budget.teacher_trace_compute_dollars} | "
            f"{manifest.budget.student_training_dollars} | "
            f"{manifest.budget.student_training_gpu_hours} | "
            f"{manifest.budget.eval_dollars} | "
            f"{manifest.budget.total_dollars} |"
        ),
        "",
        "## Teacher trace contract",
        "",
        "| teacher_checkpoint_id | teacher_run_id | trace_store_uri | "
        "trace_schema_version | min_traces | max_traces | scaffold_config_hash |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
        (
            f"| {manifest.trace_contract.teacher_checkpoint_id} | "
            f"{manifest.trace_contract.teacher_run_id} | "
            f"{manifest.trace_contract.trace_store_uri} | "
            f"{manifest.trace_contract.trace_schema_version} | "
            f"{manifest.trace_contract.min_traces} | "
            f"{manifest.trace_contract.max_traces} | "
            f"{manifest.trace_contract.scaffold_config_hash} |"
        ),
        "",
        "## Arms",
        "",
        "| arm_id | arm_kind | eligible | objectives | omission_reason |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for arm in manifest.arms:
        objectives = ", ".join(arm.objectives) if arm.objectives else ""
        lines.append(
            f"| {arm.arm_id} | {arm.arm_kind} | {arm.eligible} | {objectives} | "
            f"{arm.omission_reason or ''} |"
        )

    lines += [
        "",
        "## Note",
        "",
        manifest.note,
        "",
        "Full detail: "
        "`docs/design/iter-scaffold-distillation-activation-20260719.json`.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-id",
        default="scaffold-distillation-activation-20260719",
        help="Manifest identifier.",
    )
    parser.add_argument(
        "--teacher-checkpoint-id",
        default="teacher/checkpoint",
        help="Teacher checkpoint identifier.",
    )
    parser.add_argument(
        "--teacher-run-id", default="teacher/run", help="Teacher run identifier."
    )
    parser.add_argument(
        "--trace-store-uri", default="memory://traces", help="Trace store URI."
    )
    parser.add_argument(
        "--min-traces", type=int, default=0, help="Minimum planned trace count."
    )
    parser.add_argument(
        "--max-traces", type=int, default=0, help="Maximum planned trace count."
    )
    parser.add_argument(
        "--scaffold-config-hash", default="unknown", help="Hash of the scaffold config."
    )
    parser.add_argument(
        "--teacher-trace-compute-dollars", type=float, default=0.0, help="Budget cap."
    )
    parser.add_argument(
        "--student-training-dollars", type=float, default=0.0, help="Budget cap."
    )
    parser.add_argument(
        "--student-training-gpu-hours", type=float, default=0.0, help="Budget cap."
    )
    parser.add_argument("--eval-dollars", type=float, default=0.0, help="Budget cap.")
    parser.add_argument("--total-dollars", type=float, default=0.0, help="Budget cap.")
    parser.add_argument(
        "--scaffold-decomposition",
        default="unknown",
        choices=("unknown", "value_demonstrated", "no_value", "inventory_required"),
        help="Decomposition evidence state.",
    )
    parser.add_argument(
        "--gate-slm-161", action="store_true", help="Mark SLM-161 gate available."
    )
    parser.add_argument(
        "--gate-slm-162", action="store_true", help="Mark SLM-162 gate available."
    )
    parser.add_argument(
        "--gate-slm-168", action="store_true", help="Mark SLM-168 gate available."
    )
    parser.add_argument(
        "--gate-scaffold-value",
        action="store_true",
        help="Mark scaffold_value_demonstrated gate available.",
    )
    parser.add_argument(
        "--gate-latency",
        action="store_true",
        help="Mark latency/complexity amortization gate available.",
    )
    parser.add_argument(
        "--gate-budget", action="store_true", help="Mark budget approval gate available."
    )
    parser.add_argument(
        "--primary-metric",
        default="binding_aware_meaningful_program_rate",
        help="Primary preregistered comparison metric.",
    )
    parser.add_argument(
        "--max-attempts-for-teacher", type=int, default=1, help="Teacher attempts cap."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds for student runs.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Destination JSON manifest path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        required=True,
        help="Destination markdown summary path.",
    )
    parser.add_argument(
        "--note",
        default="SDE4-01 (SLM-179) scaffold-distillation activation manifest (wiring slice).",
        help="Free-form manifest note.",
    )
    args = parser.parse_args(argv)

    gates = _make_gates(args)
    budget = BudgetCap(
        teacher_trace_compute_dollars=args.teacher_trace_compute_dollars,
        student_training_dollars=args.student_training_dollars,
        student_training_gpu_hours=args.student_training_gpu_hours,
        eval_dollars=args.eval_dollars,
        total_dollars=args.total_dollars,
    )
    trace_contract = TeacherTraceContract(
        teacher_checkpoint_id=args.teacher_checkpoint_id,
        teacher_run_id=args.teacher_run_id,
        trace_store_uri=args.trace_store_uri,
        trace_schema_version="v1",
        min_traces=args.min_traces,
        max_traces=args.max_traces,
        scaffold_config_hash=args.scaffold_config_hash,
    )

    manifest = build_scaffold_distillation_activation_manifest(
        manifest_id=args.manifest_id,
        activation_gates=gates,
        teacher_trace_contract=trace_contract,
        budget=budget,
        arms=DEFAULT_ARMS,
        scaffold_decomposition=args.scaffold_decomposition,  # type: ignore[arg-type]
        primary_metric=args.primary_metric,
        seeds=args.seeds,
        max_attempts_for_teacher=args.max_attempts_for_teacher,
        note=args.note,
    )

    data = manifest.to_dict()
    errors = validate_scaffold_distillation_activation_manifest(data)
    if errors:
        for error in errors:
            print(f"manifest validation error: {error}")
        return 1

    data["version_stamp"] = build_version_stamp("harness.experiments")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_markdown(manifest), encoding="utf-8")

    print(f"wrote {args.out_json} and {args.out_md}")
    print(
        f"eligible={sum(1 for a in manifest.arms if a.eligible)} "
        f"omitted={sum(1 for a in manifest.arms if not a.eligible)} "
        f"activation={manifest.activation_status}:{manifest.activation_verdict} "
        f"manifest_hash={manifest.manifest_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
