#!/usr/bin/env python3
"""Run the CAP2-08 selective-hybrid continuous/discrete program-factor
latent audit (AP-033 / SLM-333; milestone "Selective Hybrid Objective").

For each of the six named arms (discrete_only, continuous_only,
selective_hybrid, all_continuous, random_continuous, oracle_continuous),
trains a :class:`~slm_training.harnesses.experiments.cap2_selective_hybrid_latent.SelectiveHybridModel`
on the SLM-144 fixture corpus's train split, then scores it on the held-out
split: overall/soft/precision-critical strict-semantics-proxy rates, per-
family MSE, encoder-collapse checks, and the factor-swap falsification test
(perturb only the soft-factor input, confirm precision-critical decode is
unchanged). Reports paired confidence intervals against the ``discrete_only``
reference and a run-level promotion disposition.

Example:

    python -m scripts.run_cap2_selective_hybrid_latent \
      --fixture-count 32 --out-dir outputs/runs/cap2_selective_hybrid_latent

This is fixture/wiring evidence only. The "strict semantics" rate scored
here is an MSE-threshold proxy (AP-030's own DISTORTION_THRESHOLD), not the
real binding_aware_meaningful_v2 scorer -- see the harness module's scope
decision. No production/ship claim is made here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.cap2_selective_hybrid_latent import (
    SelectiveHybridArmConfig,
    SelectiveHybridReport,
    default_arm_configs,
    run_selective_hybrid,
)
from slm_training.models.selective_latent_connector import SelectiveLatentArm


def _markdown_report(report: SelectiveHybridReport) -> str:
    lines = [
        "# CAP2-08 selective-hybrid latent audit",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Timestamp:* {report.timestamp}  ",
        f"*Train records:* {report.num_train}  ",
        f"*Held-out records:* {report.num_holdout}  ",
        f"*Disposition:* **{report.disposition}**",
        "",
        "## Honest caveat",
        "",
        "This is a fixture/wiring CPU run at a tiny holdout sample size. The",
        "strict-semantics-proxy metric is an MSE-threshold proxy over",
        "reconstructed SemanticPlanV1 program factors (AP-030's own",
        "DISTORTION_THRESHOLD), not the real binding_aware_meaningful_v2",
        "scorer. No production/ship claim is made here.",
        "",
        "## Disposition reasons",
        "",
    ]
    for reason in report.disposition_reasons:
        lines.append(f"- {reason}")
    lines.append("")

    lines.append("## Arms")
    lines.append("")
    lines.append(
        "| arm | overall_strict | soft_strict | precision_strict | binder_ref_mse | "
        "soft_mse | soft_collapsed | precise_collapsed | swaps_bit_exact |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |")
    for arm in report.arms:
        swaps_ok = all(s.precision_recon_bit_exact for s in arm.factor_swaps)
        lines.append(
            f"| {arm.arm_id} | {arm.overall_strict_rate:.3f} | {arm.soft_strict_rate:.3f} | "
            f"{arm.precision_strict_rate:.3f} | {arm.precision_mse_per_family.get('binder_reference', float('nan')):.4f} | "
            f"{arm.soft_mse_overall:.4f} | {arm.soft_codec_collapsed} | {arm.precise_codec_collapsed} | {swaps_ok} |"
        )
    lines.append("")

    lines.append("## Paired comparisons vs. `discrete_only`")
    lines.append("")
    lines.append("| arm | overall_strict_delta | overall_rate_ci | soft_strict_delta | soft_rate_ci | binder_ref_mse_delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for c in report.comparisons:
        overall_ci = f"[{c.overall_strict_rate_ci.get('low', float('nan')):.3f}, {c.overall_strict_rate_ci.get('high', float('nan')):.3f}]"
        soft_ci = f"[{c.soft_strict_rate_ci.get('low', float('nan')):.3f}, {c.soft_strict_rate_ci.get('high', float('nan')):.3f}]"
        lines.append(
            f"| {c.arm_id} | {c.overall_strict_rate_delta:.3f} | {overall_ci} | "
            f"{c.soft_strict_rate_delta:.3f} | {soft_ci} | {c.binder_reference_mse_delta:.4f} |"
        )
    lines.append("")

    lines.append("## Factor-swap falsification test (per arm)")
    lines.append("")
    for arm in report.arms:
        lines.append(f"### `{arm.arm_id}`")
        lines.append("")
        lines.append("| perturbation | precision_recon_bit_exact | max_abs_delta |")
        lines.append("| --- | :---: | ---: |")
        for swap in arm.factor_swaps:
            lines.append(f"| {swap.perturbation} | {swap.precision_recon_bit_exact} | {swap.max_abs_precision_delta:.2e} |")
        lines.append("")
        for note in arm.notes:
            lines.append(f"> {note}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-count", type=int, default=32)
    parser.add_argument("--fixture-seed", type=int, default=0)
    parser.add_argument(
        "--arms",
        default="",
        help="Comma-separated arm ids to keep from the default set. Empty means all defaults.",
    )
    parser.add_argument("--train-steps", type=int, default=0, help="Override train_steps for every default arm (0 = keep per-arm default).")
    parser.add_argument("--ci-resamples", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs/cap2_selective_hybrid_latent"))
    parser.add_argument("--dry-run", action="store_true", help="Describe the arms without running them.")
    args = parser.parse_args(argv)

    arms: list[SelectiveHybridArmConfig] = default_arm_configs(seed=args.fixture_seed)
    if args.arms:
        wanted = {x.strip() for x in args.arms.split(",") if x.strip()}
        arms = [a for a in arms if a.arm_id in wanted]
        if not arms:
            parser.error(f"--arms matched no default arm id (got {args.arms!r})")
        if not any(a.arm is SelectiveLatentArm.DISCRETE_ONLY for a in arms):
            parser.error("--arms must include discrete_only (the paired-comparison reference)")
    if args.train_steps > 0:
        arms = [SelectiveHybridArmConfig(**{**a.__dict__, "train_steps": args.train_steps}) for a in arms]

    if args.dry_run:
        print(f"CAP2-08 dry-run: {len(arms)} arm(s)")
        for arm in arms:
            print(f"  {arm.arm_id}: arm={arm.arm.value} train_steps={arm.train_steps}")
        return 0

    report = run_selective_hybrid(
        fixture_count=args.fixture_count,
        fixture_seed=args.fixture_seed,
        arms=arms,
        ci_resamples=args.ci_resamples,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cap2_selective_hybrid_latent_{report.run_id}.json"
    json_path.write_text(report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"cap2_selective_hybrid_latent_{report.run_id}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    swap_failures = [
        (a.arm_id, s.perturbation)
        for a in report.arms
        for s in a.factor_swaps
        if not s.precision_recon_bit_exact
    ]
    print(f"Wrote {json_path} and {md_path}")
    print(f"Disposition: {report.disposition}")
    print(f"Factor-swap falsification failures: {len(swap_failures)}")
    return 1 if swap_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
