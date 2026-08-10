"""RSP-001 (SLM-488): run EXP-SR-4 witness-guided CEGIS repair campaign.

    python -m scripts.run_rsp001_cegis_repair --mode plan-only
    python -m scripts.run_rsp001_cegis_repair --mode fixture \\
        --out-dir outputs/runs/rsp001_cegis_repair \\
        --docs-out docs/design/iter-slm488-rsp-001-cegis-repair-20260810.json

``--mode plan-only`` previews the SIE-001 ``exp-sr-4`` catalogue identity and
fixture arms. ``--mode fixture`` runs the governed multi-seed campaign and
writes durable version-stamped results under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.rsp001_cegis_repair import (
    Rsp001CampaignV1,
    plan_only_preview,
    run_campaign,
)

_VERSION_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm488_rsp001_cegis",
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    falsifier = payload.get("falsifier", {})
    yields = payload.get("aggregate_yields", {})
    invariants = payload.get("invariants", {})
    lines = [
        "# SLM-488 (RSP-001): Witness-guided CEGIS repair (EXP-SR-4)",
        "",
        "**Claim class:** `fixture` only (catalogue identity claim_class=`screening`; "
        "execution is fixture — no promotion)",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-4')}`",
        "",
        f"**Primary metric (`verified_repair_yield`, witness_cegis arm):** "
        f"{payload.get('verified_repair_yield')}",
        "",
        f"**Effect vs regenerate:** `{payload.get('effect_vs_regenerate')}` "
        f"(minimum_effect=`{payload.get('minimum_effect')}`)",
        "",
        f"**Recommendation:** `{payload.get('recommendation')}`",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| falsifier_holds | {falsifier.get('falsifier_holds')} |",
        f"| clears_regenerate_control | {falsifier.get('clears_regenerate_control')} |",
        f"| clears_minimum_effect | {falsifier.get('clears_minimum_effect')} |",
        f"| default_off | {invariants.get('default_off')} |",
        f"| learned_scorer_status | {payload.get('learned_scorer_status')} |",
        f"| seeds | {payload.get('seeds')} |",
        f"| promotion | {payload.get('promotion')} |",
        "",
        "## Arms (`verified_repair_yield`)",
        "",
        "| Arm | yield |",
        "| --- | ---: |",
    ]
    for arm, yield_val in sorted(yields.items()):
        lines.append(f"| {arm} | {yield_val} |")
    lines.extend(
        [
            "",
            "## Falsifier notes",
            "",
            f"- {falsifier.get('note', '')}",
            "",
            "## Scope",
            "",
            payload.get("scope_disclaimer", ""),
            "",
            "Command: `python -m scripts.run_rsp001_cegis_repair --mode fixture`",
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
        default=Path("outputs/runs/rsp001_cegis_repair"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm488-rsp-001-cegis-repair-20260810.json"
        ),
    )
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-cycles", type=int, default=3)
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

    campaign = Rsp001CampaignV1(
        max_records=args.max_records,
        max_cycles=args.max_cycles,
        seeds=tuple(args.seeds),
    )
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "rsp001_cegis_repair_fixture/v1",
        "claim_class": result["claim_class"],
        "catalogue_claim_class": result["catalogue_claim_class"],
        "promotion": False,
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_rsp001_cegis_repair",
                "--mode",
                "fixture",
                "--max-records",
                str(args.max_records),
                "--max-cycles",
                str(args.max_cycles),
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
        "max_records": result["max_records"],
        "max_cycles": result["max_cycles"],
        "seeds": result["seeds"],
        "learned_scorer_status": result["learned_scorer_status"],
        "verified_repair_yield": result["verified_repair_yield"],
        "aggregate_yields": result["aggregate_yields"],
        "effect_vs_regenerate": result["effect_vs_regenerate"],
        "minimum_effect": result["minimum_effect"],
        "recommendation": result["recommendation"],
        "falsifier": result["falsifier"],
        "invariants": result["invariants"],
        "topology_policy_outcomes": result["topology_policy_outcomes"],
        "seed_runs": result["seed_runs"],
        "scope_disclaimer": result["scope_disclaimer"],
        "version_stamp": build_version_stamp(*_VERSION_COMPONENTS),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload)
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(f"verified_repair_yield={payload['verified_repair_yield']}")
    print(f"falsifier_holds={payload['falsifier']['falsifier_holds']}")
    print(f"recommendation={payload['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
