"""SIE-002 (SLM-476): run EXP-SR-1 fixture-scale oracle localization.

    python -m scripts.run_sie002_oracle_localization --mode plan-only
    python -m scripts.run_sie002_oracle_localization --mode fixture \
        --out-dir outputs/runs/sie002_oracle_localization \
        --docs-out docs/design/iter-slm476-sie-002-oracle-localization-20260810.json

``--mode plan-only`` previews the SIE-001 ``exp-sr-1`` catalogue identity and
the fixture arm surface without generating evidence. ``--mode fixture`` runs
the governed campaign (composing VCE-009 arm generators) and writes durable
version-stamped results under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.sie002_oracle_localization import (
    Sie002CampaignV1,
    plan_only_preview,
    run_campaign,
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    auth = payload.get("authorized_factors", [])
    analysis = payload.get("analysis", {})
    per_factor = analysis.get("per_factor", {})
    lines = [
        "# SLM-476 (SIE-002): Factor-wise semantic oracle localization (EXP-SR-1)",
        "",
        "**Claim class:** `fixture` only",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-1')}`",
        "",
        f"**Primary metric (`oracle_factor_localization_precision`):** "
        f"{payload.get('oracle_factor_localization_precision')}",
        "",
        f"**Authorized factors for later learned-predictor work:** "
        f"`{auth if auth else '[] (none — fixture falsifier / empty authorization)'}`",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| leakage_audit_passed | {payload.get('leakage_audit', {}).get('passed')} |",
        f"| no_op_control_holds | {analysis.get('no_op_control_holds')} |",
        f"| declared_factor_fidelity_rate | {analysis.get('declared_factor_fidelity_rate')} |",
        f"| n_active_baselines (seed-aggregated) | {analysis.get('n_active_baselines')} |",
        f"| seeds | {payload.get('seeds')} |",
        f"| promotion | {payload.get('promotion')} |",
        "",
        "## Per-factor",
        "",
        "| Factor | change_rate | ceiling_recovery_rate | fidelity | authorized |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    reasons = payload.get("authorization", {}).get("authorization_reasons", {})
    auth_set = set(auth)
    for factor, stats in per_factor.items():
        lines.append(
            f"| {factor} | {stats.get('change_rate')} | "
            f"{stats.get('ceiling_recovery_rate')} | "
            f"{stats.get('mechanical_one_factor_fidelity')} | "
            f"{'yes' if factor in auth_set else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Authorization notes",
            "",
            f"- falsifier_holds: `{payload.get('authorization', {}).get('falsifier_holds')}`",
            f"- {payload.get('authorization', {}).get('falsifier_note', '')}",
            "",
            "Per-factor reasons:",
            "",
        ]
    )
    for factor, reason in reasons.items():
        lines.append(f"- `{factor}`: {reason}")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            payload.get("scope_disclaimer", ""),
            "",
            "Command: `python -m scripts.run_sie002_oracle_localization --mode fixture`",
            "",
            f"Full detail: `{docs_json.as_posix()}`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/sie002_oracle_localization"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/design/iter-slm476-sie-002-oracle-localization-20260810.json"),
    )
    parser.add_argument("--n-baseline-plans", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=40)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Matched seeds (default: catalogue seeds 0 1 2)",
    )
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = Sie002CampaignV1(
        n_baseline_plans=args.n_baseline_plans,
        pool_size=args.pool_size,
        seeds=tuple(args.seeds),
    )
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "sie002_oracle_localization_fixture/v1",
        "claim_class": result["claim_class"],
        "promotion": False,
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_sie002_oracle_localization",
                "--mode",
                "fixture",
                "--n-baseline-plans",
                str(args.n_baseline_plans),
                "--pool-size",
                str(args.pool_size),
                "--seeds",
                *[str(s) for s in args.seeds],
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "oracle_diagnostic",
        },
        "catalogue_id": result["catalogue_id"],
        "catalogue_manifest_sha256": result["catalogue_manifest_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "dataset_id": result["dataset_id"],
        "leakage_audit": result["leakage_audit"],
        "n_baseline_plans": result["n_baseline_plans"],
        "pool_size": result["pool_size"],
        "seeds": result["seeds"],
        "oracle_factor_localization_precision": result[
            "oracle_factor_localization_precision"
        ],
        "analysis": result["analysis"],
        "authorization": result["authorization"],
        "authorized_factors": result["authorized_factors"],
        "seed_runs": [
            {
                "seed": run["seed"],
                "analysis": run["analysis"],
                # Omit full arm dumps from durable docs (large); CampaignStore
                # artifact under out-dir retains them.
                "baselines_attempted": len(run["baseline_reports"]),
                "baselines_skipped": sum(
                    1 for r in run["baseline_reports"] if r["skipped"]
                ),
            }
            for run in result["seed_runs"]
        ],
        "scope_disclaimer": result["scope_disclaimer"],
        "version_stamp": build_version_stamp("harness.experiments"),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload)
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(f"authorized_factors={payload['authorized_factors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
