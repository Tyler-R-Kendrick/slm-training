"""SIE-004 (SLM-482): run EXP-SR-2 fixture-scale prompt-requirement predictor campaign.

    python -m scripts.run_sie004_predictor_campaign --mode plan-only
    python -m scripts.run_sie004_predictor_campaign --mode fixture \\
        --out-dir outputs/runs/sie004_predictor_campaign \\
        --docs-out docs/design/iter-slm482-sie-004-predictor-campaign-20260810.json

``--mode plan-only`` previews the SIE-001 ``exp-sr-2`` catalogue identity and
fixture arms. ``--mode fixture`` runs the governed multi-seed campaign and
writes durable version-stamped results under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.sie004_predictor_campaign import (
    Sie004CampaignV1,
    plan_only_preview,
    run_campaign,
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    arms = payload.get("aggregate_arms", {})
    falsifier = payload.get("falsifier", {})
    parity = payload.get("legal_support_parity", {})
    auth = payload.get("authorized_factors_source", {})
    paired = payload.get("paired_vs_frequency", {})
    lines = [
        "# SLM-482 (SIE-004): Prompt-requirement predictor campaign (EXP-SR-2)",
        "",
        "**Claim class:** `fixture` only (catalogue identity claim_class=`screening`; "
        "execution is fixture — no promotion)",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-2')}`",
        "",
        f"**Primary metric (`requirement_predictor_f1`, learned arm mean):** "
        f"{payload.get('requirement_predictor_f1')}",
        "",
        f"**Effect vs frequency:** `{payload.get('effect_vs_frequency')}` "
        f"(minimum_effect=`{payload.get('minimum_effect')}`)",
        "",
        f"**Ceiling recovery vs gold oracle:** `{payload.get('ceiling_recovery_vs_gold')}`",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| legal_support_parity_exact | {parity.get('legal_support_parity_exact')} |",
        f"| falsifier_holds | {falsifier.get('falsifier_holds')} |",
        f"| clears_frequency_baseline | {falsifier.get('clears_frequency_baseline')} |",
        f"| clears_none_baseline | {falsifier.get('clears_none_baseline')} |",
        f"| dominated_by_simpler_control | {falsifier.get('dominated_by_simpler_control')} |",
        f"| paired Δ vs frequency (mean ± SE) | {paired.get('mean')} ± {paired.get('se')} "
        f"(n={paired.get('n_seeds')}) |",
        f"| SIE-002 authorized_factors | `{auth.get('authorized_factors')}` |",
        f"| mapped program families | `{auth.get('authorized_program_families')}` |",
        f"| seeds | {payload.get('seeds')} |",
        f"| promotion | {payload.get('promotion')} |",
        "",
        "## Arms (mean F1 across seeds)",
        "",
        "| Arm | requirement_predictor_f1_mean | per-seed |",
        "| --- | ---: | --- |",
    ]
    for arm, stats in arms.items():
        lines.append(
            f"| {arm} | {stats.get('requirement_predictor_f1_mean')} | "
            f"{stats.get('requirement_predictor_f1_per_seed')} |"
        )
    lines.extend(
        [
            "",
            "## Falsifier / authorization notes",
            "",
            f"- {falsifier.get('note', '')}",
            f"- SIE-002 auth note: {auth.get('note', '')}",
            "",
            "## Legal support parity",
            "",
            f"- exact: `{parity.get('legal_support_parity_exact')}`",
            f"- {parity.get('note', '')}",
            "",
            "## Scope",
            "",
            payload.get("scope_disclaimer", ""),
            "",
            "Command: `python -m scripts.run_sie004_predictor_campaign --mode fixture`",
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
        default=Path("outputs/runs/sie004_predictor_campaign"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm482-sie-004-predictor-campaign-20260810.json"
        ),
    )
    parser.add_argument("--max-examples", type=int, default=400)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Matched seeds (default: 0 1 2)",
    )
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = Sie004CampaignV1(
        max_examples=args.max_examples,
        train_fraction=args.train_fraction,
        seeds=tuple(args.seeds),
    )
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "sie004_predictor_campaign_fixture/v1",
        "claim_class": result["claim_class"],
        "catalogue_claim_class": result["catalogue_claim_class"],
        "promotion": False,
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_sie004_predictor_campaign",
                "--mode",
                "fixture",
                "--max-examples",
                str(args.max_examples),
                "--train-fraction",
                str(args.train_fraction),
                "--seeds",
                *[str(s) for s in args.seeds],
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "fixture_diagnostic",
        },
        "catalogue_id": result["catalogue_id"],
        "catalogue_manifest_sha256": result["catalogue_manifest_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "dataset_id": result["dataset_id"],
        "factor_family": result["factor_family"],
        "max_examples": result["max_examples"],
        "seeds": result["seeds"],
        "authorized_factors_source": result["authorized_factors_source"],
        "legal_support_parity": result["legal_support_parity"],
        "requirement_predictor_f1": result["requirement_predictor_f1"],
        "aggregate_arms": result["aggregate_arms"],
        "effect_vs_frequency": result["effect_vs_frequency"],
        "ceiling_recovery_vs_gold": result["ceiling_recovery_vs_gold"],
        "minimum_effect": result["minimum_effect"],
        "paired_vs_frequency": result["paired_vs_frequency"],
        "falsifier": result["falsifier"],
        "seed_runs": [
            {
                "seed": run["seed"],
                "n_train": run["n_train"],
                "n_held_out": run["n_held_out"],
                "n_train_programs": run["n_train_programs"],
                "n_held_programs": run["n_held_programs"],
                "majority_label": run["majority_label"],
                "confidence_threshold": run["confidence_threshold"],
                "calibration": run["calibration"],
                "arms": {
                    arm: {
                        "requirement_predictor_f1": stats["requirement_predictor_f1"],
                        "coverage": stats["coverage"],
                        "selective_risk": stats["selective_risk"],
                        "abstention_rate": stats["abstention_rate"],
                        "requirement_facts_emitted_under_sie002_auth": stats[
                            "requirement_facts_emitted_under_sie002_auth"
                        ],
                    }
                    for arm, stats in run["arms"].items()
                },
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
    print(f"requirement_predictor_f1={payload['requirement_predictor_f1']}")
    print(f"falsifier_holds={payload['falsifier']['falsifier_holds']}")
    print(
        f"legal_support_parity_exact="
        f"{payload['legal_support_parity']['legal_support_parity_exact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
