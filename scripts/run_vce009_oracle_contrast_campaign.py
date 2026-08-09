"""VCE-009 (SLM-468): run the oracle/contrast fixture campaign.

    python -m scripts.run_vce009_oracle_contrast_campaign --mode plan-only
    python -m scripts.run_vce009_oracle_contrast_campaign --mode fixture \
        --out-dir outputs/runs/vce009_oracle_contrast_campaign \
        --docs-out docs/design/vce009-oracle-contrast-results.json

``--mode plan-only`` builds and validates the manifest without running any
arm or writing evidence. ``--mode fixture`` runs the real campaign
(``vce009_oracle_contrast_campaign.run_campaign``): every oracle
intervention arm (VCE-005) and semantic-contrast/metamorphic generator
(VCE-006/VCE-007) against a frozen fixture slice, and writes a durable,
version-stamped results JSON under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
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
        "--out-dir", type=Path, default=Path("outputs/runs/vce009_oracle_contrast_campaign")
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/design/vce009-oracle-contrast-results.json"),
    )
    parser.add_argument("--source-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)

    campaign = Vce009CampaignV1(source_count=args.source_count, seed=args.seed)

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
        "kind": "vce009_oracle_contrast_campaign/v1",
        "claim_class": result["claim_class"],
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_vce009_oracle_contrast_campaign",
                "--mode",
                "fixture",
                "--source-count",
                str(args.source_count),
                "--seed",
                str(args.seed),
            ],
            "docs_path": str(args.docs_out),
        },
        "arms": result["arms"],
        "arm_contract_match_rate": result["arm_contract_match_rate"],
        "inconclusive_count": result["inconclusive_count"],
        "contaminated_arm_ids": result["contaminated_arm_ids"],
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
