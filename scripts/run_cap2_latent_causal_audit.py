#!/usr/bin/env python3
"""Run the CAP2-07 oracle-latent causal-necessity/support/robustness/collapse
audit (AP-031 / SLM-330; milestone "Oracle Bottleneck").

For each evaluated (codec, capacity) arm, trains the AP-030 frozen
factor-reconstruction decoder once on a train split of the SLM-144 fixture
corpus, then -- holding that decoder frozen -- scores it on a held-out split
under the retained latent, a zero/prompt-only latent, a fresh random latent,
a between-example shuffled latent, a slot-permuted latent, a role-swapped
latent (where the codec has typed slots), a quantization-only sanity
control, and a noise-perturbed latent at several radii. Reports paired
confidence intervals (correct vs. each destructive control), latent
effective rank/variance (collapse check), and a robust semantic radius.

Example:

    python -m scripts.run_cap2_latent_causal_audit \
      --fixture-count 32 --out-dir outputs/runs/cap2_latent_causal_audit

This is fixture/wiring evidence only. The "strict semantics proxy" scored
here is an MSE-threshold proxy (AP-030's own DISTORTION_THRESHOLD), not the
real binding_aware_meaningful_v2 scorer -- no decoder from oracle latents to
full OpenUI program text exists yet in this repo (see the harness module's
scope decision). No production/ship claim is made here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.cap2_latent_causal_audit import (
    CausalAuditArm,
    CausalAuditReport,
    default_causal_audit_arms,
    run_causal_audit,
)


def _markdown_report(report: CausalAuditReport) -> str:
    lines = [
        "# CAP2-07 oracle-latent causal-necessity audit",
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
        "scorer -- no decoder from oracle latents to full OpenUI program text",
        "exists yet in this repo. No production/ship claim is made here.",
        "",
        "## Disposition reasons",
        "",
    ]
    for reason in report.disposition_reasons:
        lines.append(f"- {reason}")
    lines.append("")

    for arm in report.arms:
        lines.append(f"## Arm `{arm.arm_id}` (codec={arm.codec})")
        lines.append("")
        lines.append(
            f"- no_bypass_ok: **{arm.no_bypass_ok}** | causally_necessary: "
            f"**{arm.causally_necessary}** | role_swap_applicable: {arm.role_swap_applicable}"
        )
        lines.append(
            f"- effective_rank={arm.effective_rank:.4f} stable_rank={arm.stable_rank:.4f} "
            f"total_variance={arm.total_variance:.4f} (latent_dim={arm.latent_dim}, "
            f"num_holdout={arm.num_holdout})"
        )
        radius = "censored (>= max swept)" if arm.robust_semantic_radius_censored else arm.robust_semantic_radius
        lines.append(f"- robust_semantic_radius (sigma multiplier): {radius}")
        lines.append(f"- weaker_decoder_note: {arm.weaker_decoder_note}")
        lines.append("")
        lines.append("| condition | n | strict_rate | mean_mse | rate_ci_low | rate_ci_high |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for c in arm.conditions:
            lines.append(
                f"| {c.condition} | {c.n} | {c.strict_semantics_proxy_rate:.4f} | "
                f"{c.mean_mse_overall:.4f} | {c.rate_ci_vs_correct.get('low', float('nan')):.4f} | "
                f"{c.rate_ci_vs_correct.get('high', float('nan')):.4f} |"
            )
        lines.append("")
        for note in arm.notes:
            lines.append(f"> {note}")
        lines.append("")

    lines.append("## Hard gates")
    lines.append("")
    no_bypass_failures = [a.arm_id for a in report.arms if not a.no_bypass_ok]
    lines.append(f"- No-bypass audit failures: {len(no_bypass_failures)}")
    if no_bypass_failures:
        lines.append("- **Collapsed arms (untrustworthy):** " + ", ".join(no_bypass_failures))
    else:
        lines.append("- **PASS** every arm's decoder input is codec-only and non-collapsed.")
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
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs/cap2_latent_causal_audit"))
    parser.add_argument("--dry-run", action="store_true", help="Describe the arms without running them.")
    args = parser.parse_args(argv)

    arms: list[CausalAuditArm] = default_causal_audit_arms(seed=args.fixture_seed)
    if args.arms:
        wanted = {x.strip() for x in args.arms.split(",") if x.strip()}
        arms = [a for a in arms if a.arm_id in wanted]
        if not arms:
            parser.error(f"--arms matched no default arm id (got {args.arms!r})")
    if args.train_steps > 0:
        arms = [CausalAuditArm(**{**a.__dict__, "train_steps": args.train_steps}) for a in arms]

    if args.dry_run:
        print(f"CAP2-07 dry-run: {len(arms)} arm(s)")
        for arm in arms:
            print(f"  {arm.arm_id}: codec={arm.codec} train_steps={arm.train_steps}")
        return 0

    report = run_causal_audit(
        fixture_count=args.fixture_count,
        fixture_seed=args.fixture_seed,
        arms=arms,
        ci_resamples=args.ci_resamples,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cap2_latent_causal_audit_{report.run_id}.json"
    json_path.write_text(report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"cap2_latent_causal_audit_{report.run_id}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    no_bypass_failures = [a for a in report.arms if not a.no_bypass_ok]
    print(f"Wrote {json_path} and {md_path}")
    print(f"Disposition: {report.disposition}")
    print(f"No-bypass audit failures: {len(no_bypass_failures)}/{len(report.arms)}")
    return 1 if len(no_bypass_failures) == len(report.arms) else 0


if __name__ == "__main__":
    raise SystemExit(main())
