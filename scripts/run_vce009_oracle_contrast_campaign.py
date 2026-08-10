"""VCE-009 (SLM-468): run the fixture oracle/contrast evidence campaign.

    python -m scripts.run_vce009_oracle_contrast_campaign --mode plan-only
    python -m scripts.run_vce009_oracle_contrast_campaign --mode fixture \
        --out-dir outputs/runs/vce009_oracle_contrast \
        --docs-out docs/design/vce009-oracle-contrast-results.json

``--mode plan-only`` builds and validates the manifest without generating any
evidence. ``--mode fixture`` runs the real campaign
(``vce009_oracle_contrast_campaign.run_campaign``): every oracle-intervention
arm over a cleaned pool of real baseline/oracle plan pairs, and writes a
durable, version-stamped results JSON under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.vce009_oracle_contrast_campaign import (
    Vce009CampaignV1,
    run_campaign,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/runs/vce009_oracle_contrast")
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/design/vce009-oracle-contrast-results.json"),
    )
    parser.add_argument("--n-baseline-plans", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=40)
    args = parser.parse_args(argv)

    campaign = Vce009CampaignV1(n_baseline_plans=args.n_baseline_plans, pool_size=args.pool_size)

    if args.mode == "plan-only":
        manifest = campaign.manifest()
        payload = {
            "status": "plan_only",
            "claim_class": manifest.claim_class,
            "campaign_id": manifest.campaign_id,
            "arms": [arm.arm_id for arm in manifest.arms],
        }
        print(json.dumps(payload, indent=2))
        return 0

    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "vce009_oracle_contrast_fixture/v1",
        "claim_class": result["claim_class"],
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_vce009_oracle_contrast_campaign",
                "--mode",
                "fixture",
                "--n-baseline-plans",
                str(args.n_baseline_plans),
                "--pool-size",
                str(args.pool_size),
            ],
            "docs_path": str(args.docs_out),
        },
        "baselines_attempted": result["baselines_attempted"],
        "baselines_skipped": result["baselines_skipped"],
        "declared_factor_fidelity_rate": result["declared_factor_fidelity_rate"],
        "no_op_control_holds": result["no_op_control_holds"],
        "records_total": result["records_total"],
        "records_contaminated": result["records_contaminated"],
        "records_safe_for_export": result["records_safe_for_export"],
        "contaminated_arm_ids": result["contaminated_arm_ids"],
        "expected_contaminated_arm_ids": result["expected_contaminated_arm_ids"],
        "baseline_reports": result["baseline_reports"],
        "scope_disclaimer": result["scope_disclaimer"],
        "manifest_sha256": result["manifest_sha256"],
        "version_stamp": build_version_stamp("harness.experiments"),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.docs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
