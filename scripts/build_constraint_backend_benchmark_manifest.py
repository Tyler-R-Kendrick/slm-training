#!/usr/bin/env python3
"""Build the SDE3-04 constraint-backend benchmark manifest.

Example::

    python -m scripts.build_constraint_backend_benchmark_manifest \
        --manifest-id iter-constraint-backend-benchmark-20260719 \
        --microbenchmark-repetitions 10 \
        --end-to-end-repetitions 3 \
        --max-dollars 50.0 \
        --gpu-hours 2.0 \
        --gate-eval-cache \
        --gate-budget \
        --out-json docs/design/iter-constraint-backend-benchmark-20260719.json \
        --out-md docs/design/iter-constraint-backend-benchmark-20260719.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.constraint_backend_benchmark import (
    BackendAdapter,
    BudgetCap,
    ConstraintBackendBenchmarkManifest,
    build_constraint_backend_benchmark_manifest,
    validate_constraint_backend_benchmark_manifest,
)
from slm_training.versioning import build_version_stamp


DEFAULT_NOTE = (
    "SDE3-04 constraint-backend benchmark manifest (wiring slice). "
    "No decoder package is installed and no benchmark is run."
)


def _parse_backend(text: str) -> BackendAdapter:
    """Parse a backend spec as ``backend_id:package_name:package_version``.

    ``backend_id`` alone is accepted and uses ``unset`` for package metadata.
    """
    parts = text.split(":")
    backend_id = parts[0]
    if backend_id not in (
        "current",
        "syncode",
        "domino",
        "xgrammar",
        "unconstrained",
    ):
        raise ValueError(f"unknown backend_id: {backend_id!r}")
    package_name = parts[1] if len(parts) > 1 else backend_id
    package_version = parts[2] if len(parts) > 2 else "unset"
    return BackendAdapter(
        backend_id=backend_id,  # type: ignore[arg-type]
        package_name=package_name,
        package_version=package_version,
    )


def _render_markdown(manifest: ConstraintBackendBenchmarkManifest) -> str:
    lines = [
        "# SDE3-04 Constraint-Backend Benchmark Manifest",
        "",
        f"- **manifest_id**: `{manifest.manifest_id}`",
        f"- **schema_version**: `{manifest.schema_version}`",
        f"- **hypothesis_id**: `{manifest.hypothesis_id}`",
        f"- **activation_status**: `{manifest.activation_status}`",
        f"- **activation_verdict**: `{manifest.activation_verdict}`",
        f"- **campaign_verdict**: `{manifest.campaign_verdict}`",
        f"- **primary_metric**: `{manifest.primary_metric}`",
        f"- **seeds**: {list(manifest.seeds)}",
        f"- **null_threshold_percent**: {manifest.null_threshold_percent}",
        f"- **manifest_hash**: `{manifest.manifest_hash}`",
        "",
        "## Activation gates",
        "",
        "| gate_id | depends_on_issue_id | required_status | available | evidence |",
        "|---|---|---|---|---|",
    ]
    for gate in manifest.activation_gates:
        lines.append(
            f"| {gate.gate_id} | {gate.depends_on_issue_id or ''} | "
            f"{gate.required_status or ''} | {gate.available} | {gate.evidence or ''} |"
        )
    lines.extend(
        [
            "",
            "## Backends",
            "",
            "| backend_id | package_name | package_version | local_offline | supported_kinds |",
            "|---|---|---|---|---|",
        ]
    )
    for backend in manifest.backends:
        kinds = ", ".join(backend.supported_grammar_kinds) or "-"
        lines.append(
            f"| {backend.backend_id} | {backend.package_name} | "
            f"{backend.package_version} | {backend.local_offline_available} | {kinds} |"
        )
    lines.extend(
        [
            "",
            "## Budget caps",
            "",
            f"- microbenchmark_repetitions: {manifest.budget.microbenchmark_repetitions}",
            f"- end_to_end_repetitions: {manifest.budget.end_to_end_repetitions}",
            f"- max_dollars: {manifest.budget.max_dollars}",
            f"- gpu_hours: {manifest.budget.gpu_hours}",
            "",
            "## Arms",
            "",
            "| arm_id | backend_id | benchmark_layer | eligible | omission_reason |",
            "|---|---|---|---|---|",
        ]
    )
    for arm in manifest.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.backend_id} | {arm.benchmark_layer} | "
            f"{arm.eligible} | {arm.omission_reason or ''} |"
        )
    if manifest.note:
        lines.extend(["", "## Note", "", manifest.note])
    lines.extend(
        [
            "",
            "## Version stamp",
            "",
            "```json",
            json.dumps(build_version_stamp("harness.experiments", "matrix.perf"), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-id",
        default="iter-constraint-backend-benchmark-20260719",
        help="Manifest identifier.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help=(
            "Backend spec as backend_id:package_name:package_version "
            "(repeatable; omit to use the default backend set)."
        ),
    )
    parser.add_argument(
        "--microbenchmark-repetitions",
        type=int,
        default=0,
        help="Repetitions for static microbenchmark arms (default 0).",
    )
    parser.add_argument(
        "--end-to-end-repetitions",
        type=int,
        default=0,
        help="Repetitions for end-to-end surface arms (default 0).",
    )
    parser.add_argument(
        "--max-dollars",
        type=float,
        default=0.0,
        help="Maximum dollar budget (default 0.0).",
    )
    parser.add_argument(
        "--gpu-hours",
        type=float,
        default=0.0,
        help="Maximum GPU-hour budget (default 0.0).",
    )
    parser.add_argument(
        "--gate-eval-cache",
        action="store_true",
        help="Mark the eval-cache/cost gate as available.",
    )
    parser.add_argument(
        "--gate-budget",
        action="store_true",
        help="Mark the budget gate as available.",
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
        default=None,
        help="Destination markdown summary path (defaults to --out-json with .md).",
    )
    parser.add_argument(
        "--note",
        default=DEFAULT_NOTE,
        help="Free-form manifest note.",
    )
    args = parser.parse_args(argv)

    if args.out_md is None:
        args.out_md = args.out_json.with_suffix(".md")

    from slm_training.harnesses.experiments.constraint_backend_benchmark import (
        ActivationGate,
    )

    activation_gates = (
        ActivationGate(
            gate_id="eval_cache_or_cost_approved",
            depends_on_issue_id="SLM-175",
            required_status="Done",
            available=args.gate_eval_cache,
            evidence="",
        ),
        ActivationGate(
            gate_id="budget_approved",
            available=args.gate_budget,
            evidence="",
        ),
    )

    backends = None
    if args.backends:
        backends = tuple(_parse_backend(text) for text in args.backends)

    budget = BudgetCap(
        microbenchmark_repetitions=args.microbenchmark_repetitions,
        end_to_end_repetitions=args.end_to_end_repetitions,
        max_dollars=args.max_dollars,
        gpu_hours=args.gpu_hours,
    )

    manifest = build_constraint_backend_benchmark_manifest(
        manifest_id=args.manifest_id,
        activation_gates=activation_gates,
        backends=backends,
        budget=budget,
        primary_metric="binding_aware_meaningful_program_rate",
        note=args.note,
    )

    errors = validate_constraint_backend_benchmark_manifest(manifest.to_dict())
    if errors:
        for error in errors:
            print(f"manifest validation error: {error}")
        return 1

    payload = {
        "manifest": manifest.to_dict(),
        "version_stamp": build_version_stamp("harness.experiments", "matrix.perf"),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(manifest), encoding="utf-8")

    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(
        f"backends={len(manifest.backends)} arms={len(manifest.arms)} "
        f"activation_verdict={manifest.activation_verdict} "
        f"manifest_hash={manifest.manifest_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
