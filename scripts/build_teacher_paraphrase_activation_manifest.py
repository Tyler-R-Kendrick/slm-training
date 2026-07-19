#!/usr/bin/env python3
"""Build the SDE4-03 (SLM-181) teacher-paraphrase activation/budget manifest.

This CLI emits the machine-readable manifest that must exist before any teacher
spend is authorized. It is intentionally plan-only: no teacher calls, no
training, and no ship claim.

Example::

    python -m scripts.build_teacher_paraphrase_activation_manifest \
        --manifest-id sde4-03-v1 \
        --provider openai \
        --model gpt-4o \
        --revision 2024-05-13 \
        --max-dollars 100.0 \
        --max-input-tokens 10_000_000 \
        --max-output-tokens 2_000_000 \
        --slm171-outcome prompt_diversity_limited \
        --gate-slm-169 \
        --gate-slm-171 \
        --gate-slm-106 \
        --out-json docs/design/iter-teacher-paraphrase-activation-20260719.json \
        --out-md docs/design/iter-teacher-paraphrase-activation-20260719.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.teacher_paraphrase_activation import (
    ActivationGate,
    BudgetCap,
    TeacherParaphraseArm,
    TeacherProviderConfig,
    build_teacher_paraphrase_activation_manifest,
    validate_teacher_paraphrase_activation_manifest,
)
from slm_training.versioning import build_version_stamp


_DEFAULT_GATES: list[ActivationGate] = [
    ActivationGate(
        gate_id="canonical_ast_codec_binding",
        depends_on_issue_id="SLM-169",
        required_status="Done",
        available=False,
        evidence="canonical AST, codec round-trip, binding integrity, lineage, split-leakage gate",
    ),
    ActivationGate(
        gate_id="roottype_diversity_economics",
        depends_on_issue_id="SLM-171",
        required_status="Done",
        available=False,
        evidence="prompt/template diversity identified as a plausible bottleneck at fixed roots",
    ),
    ActivationGate(
        gate_id="independent_judge_path",
        depends_on_issue_id="SLM-106",
        required_status="Done",
        available=False,
        evidence="cross-family judge or blinded human rubric available and disjoint from teacher generator",
    ),
]

_DEFAULT_ARMS: list[TeacherParaphraseArm] = [
    TeacherParaphraseArm(
        arm_id="canonical_only",
        corpus_variant="canonical_only",
        eligible=True,
    ),
    TeacherParaphraseArm(
        arm_id="deterministic_templates",
        corpus_variant="deterministic_templates",
        eligible=True,
    ),
    TeacherParaphraseArm(
        arm_id="teacher_paraphrases",
        corpus_variant="teacher_paraphrases",
        eligible=True,
        styles=(
            "concise",
            "detailed",
            "business_user_story",
            "imperative",
            "multi_constraint",
        ),
    ),
    TeacherParaphraseArm(
        arm_id="mixed_50_50",
        corpus_variant="mixed_50_50",
        eligible=True,
    ),
    TeacherParaphraseArm(
        arm_id="teacher_shuffled_target",
        corpus_variant="teacher_shuffled_target",
        eligible=False,
        omission_reason="diagnostic control only; never an eligible training corpus beyond a bounded specificity test",
    ),
    TeacherParaphraseArm(
        arm_id="teacher_low_diversity",
        corpus_variant="teacher_low_diversity",
        eligible=True,
        styles=("concise",),
    ),
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-id", default="sde4-03-v1")
    parser.add_argument(
        "--slm171-outcome",
        default="unknown",
        choices=["unknown", "prompt_diversity_limited", "root_diversity_limited"],
        help="Result of SLM-171 that determines whether teacher paraphrases are prioritized.",
    )
    parser.add_argument(
        "--gate-slm-169",
        action="store_true",
        help="Mark the SLM-169 canonical-AST/binding gate as available.",
    )
    parser.add_argument(
        "--gate-slm-171",
        action="store_true",
        help="Mark the SLM-171 diversity-economics gate as available.",
    )
    parser.add_argument(
        "--gate-slm-106",
        action="store_true",
        help="Mark the SLM-106 independent-judge gate as available.",
    )
    parser.add_argument("--provider", default="unset")
    parser.add_argument("--model", default="unset")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-dollars", type=float, default=0.0)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument(
        "--primary-metric", default="binding_aware_meaningful_program_rate"
    )
    parser.add_argument(
        "--max-derivatives-per-root", type=int, default=5
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
        default="SDE4-03 teacher-paraphrase activation/budget manifest (wiring slice).",
    )
    return parser.parse_args(argv)


def _apply_gate_flags(gates: list[ActivationGate], args: argparse.Namespace) -> list[ActivationGate]:
    flag_map = {
        "SLM-169": args.gate_slm_169,
        "SLM-171": args.gate_slm_171,
        "SLM-106": args.gate_slm_106,
    }
    out: list[ActivationGate] = []
    for gate in gates:
        available = flag_map.get(gate.depends_on_issue_id, False)
        out.append(
            ActivationGate(
                gate_id=gate.gate_id,
                depends_on_issue_id=gate.depends_on_issue_id,
                required_status=gate.required_status,
                available=available,
                evidence=gate.evidence,
            )
        )
    return out


def _render_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# SDE4-03 teacher-paraphrase activation/budget manifest",
        "",
        f"- **manifest_id**: {manifest['manifest_id']}",
        f"- **schema_version**: {manifest['schema_version']}",
        f"- **hypothesis_id**: {manifest['hypothesis_id']}",
        f"- **activation_status**: {manifest['activation_status']}",
        f"- **activation_verdict**: {manifest['activation_verdict']}",
        f"- **campaign_verdict**: {manifest['campaign_verdict']}",
        f"- **primary_metric**: {manifest['primary_metric']}",
        f"- **max_derivatives_per_root**: {manifest['max_derivatives_per_root']}",
        f"- **manifest_hash**: {manifest['manifest_hash']}",
        "",
        "## Activation gates",
        "",
    ]
    for gate in manifest["activation_gates"]:  # type: ignore[index]
        status = "✅ available" if gate["available"] else "❌ not available"
        lines.append(
            f"- **{gate['gate_id']}** ({gate['depends_on_issue_id']} -> {gate['required_status']}): {status}"
        )
        if gate.get("evidence"):
            lines.append(f"  - evidence: {gate['evidence']}")
    lines.extend(
        [
            "",
            "## Provider",
            "",
            f"- provider: {manifest['provider']['provider']}",
            f"- model: {manifest['provider']['model']}",
            f"- revision: {manifest['provider'].get('revision', 'unset')}",
            "",
            "## Budget cap",
            "",
            f"- max_dollars: {manifest['budget'].get('max_dollars')}",
            f"- max_input_tokens: {manifest['budget'].get('max_input_tokens')}",
            f"- max_output_tokens: {manifest['budget'].get('max_output_tokens')}",
            "",
            "## Arms",
            "",
        ]
    )
    for arm in manifest["arms"]:  # type: ignore[index]
        eligibility = "eligible" if arm["eligible"] else "omitted"
        lines.append(f"- **{arm['arm_id']}** ({arm['corpus_variant']}) — {eligibility}")
        if arm.get("styles"):
            lines.append(f"  - styles: {', '.join(arm['styles'])}")
        if arm.get("omission_reason"):
            lines.append(f"  - omission_reason: {arm['omission_reason']}")
    lines.extend(
        [
            "",
            "## Honest caveats",
            "",
            "This manifest is a wiring-only artifact. No teacher API calls, no model training, and no ship claim have occurred. Teacher spend must not begin until `activation_verdict` is `ready_to_spend`. The default output is intentionally blocked/budgeted to avoid accidental spend.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    gates = _apply_gate_flags(_DEFAULT_GATES, args)

    provider = TeacherProviderConfig(
        provider=args.provider,
        model=args.model,
        revision=args.revision,
        system_prompt_template_hash=None,
        user_prompt_template_hash=None,
        sampling_parameters={"temperature": 0.7, "top_p": 0.95},
        max_tokens=4096,
        retry_policy={"max_retries": 3, "backoff": "exponential"},
        cost_per_1k_input_usd=None,
        cost_per_1k_output_usd=None,
    )
    budget = BudgetCap(
        max_dollars=args.max_dollars,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
    )

    slm171_outcome = (
        None if args.slm171_outcome == "unknown" else args.slm171_outcome
    )
    manifest_obj = build_teacher_paraphrase_activation_manifest(
        manifest_id=args.manifest_id,
        activation_gates=gates,
        provider=provider,
        budget=budget,
        arms=_DEFAULT_ARMS,
        slm171_outcome=slm171_outcome,
        primary_metric=args.primary_metric,
        max_derivatives_per_root=args.max_derivatives_per_root,
        note=args.note,
    )
    manifest: dict[str, object] = manifest_obj.to_dict()
    manifest["version_stamp"] = build_version_stamp("harness.experiments")

    errors = validate_teacher_paraphrase_activation_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest validation error: {error}")
        return 1

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out_json}")
    print(
        f"activation_status={manifest['activation_status']} "
        f"activation_verdict={manifest['activation_verdict']} "
        f"campaign_verdict={manifest['campaign_verdict']} "
        f"manifest_hash={manifest['manifest_hash']}"
    )

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_render_markdown(manifest), encoding="utf-8")
        print(f"wrote {args.out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
