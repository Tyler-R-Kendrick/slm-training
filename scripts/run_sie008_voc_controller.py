"""SIE-008 (SLM-484): run EXP-SR-5 fixture-scale value-of-compute controller campaign.

    python -m scripts.run_sie008_voc_controller --mode plan-only
    python -m scripts.run_sie008_voc_controller --mode fixture \\
        --out-dir outputs/runs/sie008_voc_controller \\
        --docs-out docs/design/iter-slm484-sie-008-voc-controller-20260810.json

``--mode plan-only`` previews the SIE-001 ``exp-sr-5`` catalogue identity and
fixture arms. ``--mode fixture`` runs the governed multi-seed campaign and
writes durable version-stamped results under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.sie008_voc_controller import (
    Sie008CampaignV1,
    plan_only_preview,
    run_campaign,
)

_VERSION_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm484_sie008_voc",
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    arms = payload.get("aggregate_arms", {})
    falsifier = payload.get("falsifier", {})
    parity = payload.get("legal_support_parity", {})
    paired = payload.get("paired_vs_hand_threshold", {})
    rec = payload.get("recommendation", {})
    lines = [
        "# SLM-484 (SIE-008): Value-of-compute controller campaign (EXP-SR-5)",
        "",
        "**Claim class:** `fixture` only (catalogue identity claim_class=`diagnostic`; "
        "execution is fixture — no promotion)",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-5')}`",
        "",
        f"**Primary metric (`compute_value_regret`, symbolic arm mean):** "
        f"{payload.get('compute_value_regret')}",
        "",
        f"**Effect vs hand-threshold (control − symbolic):** "
        f"`{payload.get('effect_vs_hand_threshold')}` "
        f"(minimum_effect=`{payload.get('minimum_effect')}`)",
        "",
        f"**Ceiling recovery vs gold oracle:** `{payload.get('ceiling_recovery_vs_gold')}`",
        "",
        f"**Recommendation:** `{rec.get('disposition')}` — {rec.get('rationale', '')}",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| legal_support_parity_exact | {parity.get('legal_support_parity_exact')} |",
        f"| falsifier_holds | {falsifier.get('falsifier_holds')} |",
        f"| clears_hand_threshold_control | {falsifier.get('clears_hand_threshold_control')} |",
        f"| clears_minimum_effect | {falsifier.get('clears_minimum_effect')} |",
        f"| paired Δ vs hand-threshold (mean ± SE) | {paired.get('mean')} ± {paired.get('se')} "
        f"(n={paired.get('n_seeds')}) |",
        f"| seeds | {payload.get('seeds')} |",
        f"| promotion | {payload.get('promotion')} |",
        "",
        "## Arms (mean regret across seeds; lower is better)",
        "",
        "| Arm | compute_value_regret_mean | per-seed |",
        "| --- | ---: | --- |",
    ]
    for arm, stats in arms.items():
        lines.append(
            f"| {arm} | {stats.get('compute_value_regret_mean')} | "
            f"{stats.get('compute_value_regret_per_seed')} |"
        )
    lines.extend(
        [
            "",
            "## Falsifier / safety notes",
            "",
            f"- {falsifier.get('note', '')}",
            f"- {parity.get('note', '')}",
            "",
            "## Scope",
            "",
            payload.get("scope_disclaimer", ""),
            "",
            "Command: `python -m scripts.run_sie008_voc_controller --mode fixture`",
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
        default=Path("outputs/runs/sie008_voc_controller"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm484-sie-008-voc-controller-20260810.json"
        ),
    )
    parser.add_argument("--n-records", type=int, default=120)
    parser.add_argument("--discovery-fraction", type=float, default=0.7)
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

    campaign = Sie008CampaignV1(
        n_records=args.n_records,
        discovery_fraction=args.discovery_fraction,
        seeds=tuple(args.seeds),
    )
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "sie008_voc_controller_fixture/v1",
        "claim_class": result["claim_class"],
        "catalogue_claim_class": result["catalogue_claim_class"],
        "promotion": False,
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_sie008_voc_controller",
                "--mode",
                "fixture",
                "--n-records",
                str(args.n_records),
                "--discovery-fraction",
                str(args.discovery_fraction),
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
        "n_records": result["n_records"],
        "discovery_fraction": result["discovery_fraction"],
        "seeds": result["seeds"],
        "symbolic_rule": result["symbolic_rule"],
        "legal_support_parity": result["legal_support_parity"],
        "compute_value_regret": result["compute_value_regret"],
        "aggregate_arms": result["aggregate_arms"],
        "effect_vs_hand_threshold": result["effect_vs_hand_threshold"],
        "ceiling_recovery_vs_gold": result["ceiling_recovery_vs_gold"],
        "minimum_effect": result["minimum_effect"],
        "paired_vs_hand_threshold": result["paired_vs_hand_threshold"],
        "falsifier": result["falsifier"],
        "recommendation": result["recommendation"],
        "seed_runs": [
            {
                "seed": run["seed"],
                "n_records": run["n_records"],
                "n_discovery": run["n_discovery"],
                "n_eval": run["n_eval"],
                "singleton_bypass": run["singleton_bypass"],
                "arms": {
                    arm: {
                        "compute_value_regret": stats["compute_value_regret"],
                        "agreements": stats["agreements"],
                        "abstentions": stats["abstentions"],
                        "coverage": stats["coverage"],
                        "expressions_evaluated": stats["expressions_evaluated"],
                    }
                    for arm, stats in run["arms"].items()
                },
            }
            for run in result["seed_runs"]
        ],
        "scope_disclaimer": result["scope_disclaimer"],
        "version_stamp": build_version_stamp(*_VERSION_COMPONENTS),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload)
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(f"compute_value_regret={payload['compute_value_regret']}")
    print(f"falsifier_holds={payload['falsifier']['falsifier_holds']}")
    print(f"disposition={payload['recommendation']['disposition']}")
    print(
        f"legal_support_parity_exact="
        f"{payload['legal_support_parity']['legal_support_parity_exact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
