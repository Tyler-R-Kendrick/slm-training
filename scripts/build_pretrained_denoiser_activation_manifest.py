#!/usr/bin/env python3
"""Build the SDE4-04 (SLM-182) pretrained-denoiser activation manifest.

This is a plan-only wiring slice: it emits a frozen, versioned manifest that
describes the activation gates, candidate, budget, and experimental arms for the
pretrained-denoiser decision.  It does not download models or run training.

Example::

    python -m scripts.build_pretrained_denoiser_activation_manifest \
        --candidate-selection candidate_selected \
        --provider huggingface \
        --model bert-base-uncased \
        --gate-available slm161_data_contract_closed \
        --gate-available slm24_evaluation_ready \
        --gate-available slm175_connector_spec_closed \
        --gate-available small_baseline_stable \
        --gate-available budget_approved \
        --gate-available license_compatible \
        --budget-total-dollars 500 \
        --out-json outputs/pretrained_denoiser_activation.json \
        --out-md outputs/pretrained_denoiser_activation.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.pretrained_denoiser_activation import (
    DEFAULT_ACTIVATION_GATES,
    DEFAULT_ARMS,
    ActivationGate,
    BudgetCap,
    LicenseTerms,
    PretrainedDenoiserActivationManifest,
    PretrainedDenoiserArm,
    PretrainedDenoiserCandidate,
    build_pretrained_denoiser_activation_manifest,
    validate_pretrained_denoiser_activation_manifest,
)
from slm_training.versioning import build_version_stamp


def _make_candidate(args: argparse.Namespace) -> PretrainedDenoiserCandidate:
    return PretrainedDenoiserCandidate(
        candidate_id=args.candidate_id,
        provider=args.provider,
        repository=args.repository,
        model=args.model,
        revision=args.revision,
        file_hashes=dict(args.file_hash) if args.file_hash else {},
        license=LicenseTerms(
            spdx_id=args.license_spdx_id,
            commercial_use_allowed=args.license_commercial_use,
            redistribution_allowed=args.license_redistribution,
            modification_allowed=args.license_modification,
            attribution_required=args.license_attribution_required,
            notes=args.license_notes,
        ),
        architecture=args.architecture,
        pretraining_objective=args.pretraining_objective,
        parameter_count=args.parameter_count,
        hidden_width=args.hidden_width,
        num_layers=args.num_layers,
        context_length=args.context_length,
        tokenizer_id=args.tokenizer_id,
        conversion_method=args.conversion_method,
        supported_formats=tuple(args.supported_formats),
        estimated_train_memory_bytes=args.estimated_train_memory_bytes,
        estimated_inference_memory_bytes=args.estimated_inference_memory_bytes,
        estimated_flops_per_forward=args.estimated_flops_per_forward,
        expected_serialized_bytes=args.expected_serialized_bytes,
        expected_deployed_bytes=args.expected_deployed_bytes,
        local_offline_available=args.local_offline_available,
        unsupported_operations=tuple(args.unsupported_operations),
        hardware_requirements=tuple(args.hardware_requirements),
        selection_evidence=args.selection_evidence,
    )


def _make_gates(args: argparse.Namespace) -> tuple[ActivationGate, ...]:
    available = set(args.gate_available)
    return tuple(
        ActivationGate(
            gate_id=gate.gate_id,
            depends_on_issue_id=gate.depends_on_issue_id,
            required_status=gate.required_status,
            available=gate.gate_id in available,
            evidence=gate.evidence,
        )
        for gate in DEFAULT_ACTIVATION_GATES
    )


def _make_budget(args: argparse.Namespace) -> BudgetCap:
    return BudgetCap(
        model_acquisition_dollars=args.budget_model_acquisition,
        gpu_hours=args.budget_gpu_hours,
        storage_dollars=args.budget_storage_dollars,
        conversion_dollars=args.budget_conversion_dollars,
        eval_dollars=args.budget_eval_dollars,
        total_dollars=args.budget_total_dollars,
    )


def _render_markdown(manifest: PretrainedDenoiserActivationManifest) -> str:
    lines: list[str] = [
        f"# {manifest.manifest_id}",
        "",
        f"**Hypothesis:** {manifest.hypothesis_id}",
        f"**Activation status:** {manifest.activation_status}",
        f"**Activation verdict:** {manifest.activation_verdict}",
        f"**Campaign verdict:** {manifest.campaign_verdict}",
        f"**Primary metric:** {manifest.primary_metric}",
        f"**Manifest hash:** `{manifest.manifest_hash}`",
        "",
        "## Candidate",
        "",
        f"- **ID:** {manifest.candidate.candidate_id}",
        f"- **Provider:** {manifest.candidate.provider}",
        f"- **Repository:** {manifest.candidate.repository}",
        f"- **Model:** {manifest.candidate.model}",
        f"- **Revision:** {manifest.candidate.revision}",
        f"- **Architecture:** {manifest.candidate.architecture}",
        f"- **Parameters:** {manifest.candidate.parameter_count}",
        f"- **License:** {manifest.candidate.license.spdx_id}",
        f"- **Local offline available:** {manifest.candidate.local_offline_available}",
        "",
        "## Budget",
        "",
        f"- **Model acquisition:** {manifest.budget.model_acquisition_dollars}",
        f"- **GPU hours:** {manifest.budget.gpu_hours}",
        f"- **Storage:** {manifest.budget.storage_dollars}",
        f"- **Conversion:** {manifest.budget.conversion_dollars}",
        f"- **Eval:** {manifest.budget.eval_dollars}",
        f"- **Total:** {manifest.budget.total_dollars}",
        "",
        "## Activation gates",
        "",
        "| Gate | Depends on | Required | Available |",
        "|------|------------|----------|-----------|",
    ]
    for gate in manifest.activation_gates:
        lines.append(
            f"| {gate.gate_id} | {gate.depends_on_issue_id} | "
            f"{gate.required_status} | {gate.available} |"
        )
    lines.extend(["", "## Arms", ""])
    for arm in manifest.arms:
        status = "eligible" if arm.eligible else f"omitted ({arm.omission_reason})"
        lines.append(f"- **{arm.arm_id}** (`{arm.arm_kind}`) — {status}")
    if manifest.note:
        lines.extend(["", "## Note", "", manifest.note])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-id",
        default="pretrained_denoiser_activation/v1",
        help="Manifest identifier.",
    )
    parser.add_argument(
        "--candidate-selection",
        choices=("unknown", "candidate_selected", "no_candidate_meets_constraints"),
        default="unknown",
        help="Where the candidate selection process currently stands.",
    )
    parser.add_argument(
        "--primary-metric",
        default="binding_aware_meaningful_program_rate",
        help="Primary preregistered comparison metric.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(0, 1, 2),
        help="Random seeds for the experimental arms.",
    )
    parser.add_argument(
        "--max-deployed-bytes",
        type=int,
        default=1_000_000_000,
        help="Maximum deployed model size in bytes.",
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
        help="Optional destination markdown summary path.",
    )
    parser.add_argument(
        "--note",
        default="SDE4-04 pretrained-denoiser activation manifest (wiring slice).",
        help="Free-form manifest note.",
    )

    # Candidate fields
    parser.add_argument("--candidate-id", default="unset")
    parser.add_argument("--provider", default="unset")
    parser.add_argument("--repository", default="")
    parser.add_argument("--model", default="unset")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--file-hash",
        action="append",
        nargs=2,
        metavar=("FILENAME", "SHA256"),
        help="Add a file hash; may be given multiple times.",
    )
    parser.add_argument("--architecture", default="unset")
    parser.add_argument("--pretraining-objective", default="denoising_lm")
    parser.add_argument("--parameter-count", type=int, default=0)
    parser.add_argument("--hidden-width", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--tokenizer-id", default="unset")
    parser.add_argument("--conversion-method", default="unset")
    parser.add_argument(
        "--supported-formats",
        nargs="+",
        default=("safetensors",),
    )
    parser.add_argument("--estimated-train-memory-bytes", type=int, default=0)
    parser.add_argument("--estimated-inference-memory-bytes", type=int, default=0)
    parser.add_argument("--estimated-flops-per-forward", type=int, default=0)
    parser.add_argument("--expected-serialized-bytes", type=int, default=0)
    parser.add_argument("--expected-deployed-bytes", type=int, default=0)
    parser.add_argument(
        "--local-offline-available",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--unsupported-operations",
        nargs="*",
        default=(),
    )
    parser.add_argument(
        "--hardware-requirements",
        nargs="*",
        default=(),
    )
    parser.add_argument("--selection-evidence", default="candidate not yet selected")

    # License fields
    parser.add_argument("--license-spdx-id", default="unset")
    parser.add_argument(
        "--license-commercial-use",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--license-redistribution",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--license-modification",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--license-attribution-required",
        action="store_true",
        default=False,
    )
    parser.add_argument("--license-notes", default="")

    # Gate flags
    parser.add_argument(
        "--gate-available",
        action="append",
        default=[],
        help="Mark a gate as available; may be given multiple times.",
    )

    # Budget fields
    parser.add_argument("--budget-model-acquisition", type=float, default=None)
    parser.add_argument("--budget-gpu-hours", type=float, default=None)
    parser.add_argument("--budget-storage-dollars", type=float, default=None)
    parser.add_argument("--budget-conversion-dollars", type=float, default=None)
    parser.add_argument("--budget-eval-dollars", type=float, default=None)
    parser.add_argument("--budget-total-dollars", type=float, default=0.0)

    args = parser.parse_args(argv)

    arms: tuple[PretrainedDenoiserArm, ...] = DEFAULT_ARMS
    candidate = _make_candidate(args)
    gates = _make_gates(args)
    budget = _make_budget(args)

    manifest = build_pretrained_denoiser_activation_manifest(
        candidate=candidate,
        budget=budget,
        arms=arms,
        manifest_id=args.manifest_id,
        candidate_selection=args.candidate_selection,
        activation_gates=gates,
        primary_metric=args.primary_metric,
        seeds=args.seeds,
        max_deployed_bytes=args.max_deployed_bytes,
        note=args.note,
    )

    errors = validate_pretrained_denoiser_activation_manifest(manifest.to_dict())
    if errors:
        for error in errors:
            print(f"manifest validation error: {error}")
        return 1

    payload = {
        "version_stamp": build_version_stamp("harness.experiments"),
        "manifest": manifest.to_dict(),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_render_markdown(manifest), encoding="utf-8")

    print(f"wrote {args.out_json}")
    print(
        f"activation_verdict={manifest.activation_verdict} "
        f"status={manifest.activation_status} "
        f"eligible={sum(1 for a in manifest.arms if a.eligible)} "
        f"omitted={sum(1 for a in manifest.arms if not a.eligible)} "
        f"manifest_hash={manifest.manifest_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
