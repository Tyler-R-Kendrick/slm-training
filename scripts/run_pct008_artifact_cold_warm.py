"""PCT-008 (SLM-464): run the fixture cold/warm artifact-parity campaign.

    python -m scripts.run_pct008_artifact_cold_warm --mode plan-only
    python -m scripts.run_pct008_artifact_cold_warm --mode fixture \
        --out-dir outputs/runs/pct008_artifact_cold_warm \
        --docs-out docs/design/pct008-artifact-cold-warm-results.json

``--mode plan-only`` builds and validates the manifest without spawning any
subprocess or writing evidence. ``--mode fixture`` runs the real campaign
(``pct008_artifact_cold_warm.run_campaign``): a domain-parity gate, then
fresh-subprocess cold trials for every arm plus one in-process warm trial,
and writes a durable, version-stamped results JSON under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.pct008_artifact_cold_warm import (
    Pct008CampaignV1,
    run_campaign,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/runs/pct008_artifact_cold_warm")
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/design/pct008-artifact-cold-warm-results.json"),
    )
    parser.add_argument("--cold-trials-per-arm", type=int, default=3)
    parser.add_argument("--warm-trials", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args(argv)

    campaign = Pct008CampaignV1(
        cold_trials_per_arm=args.cold_trials_per_arm, warm_trials=args.warm_trials
    )

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

    result = run_campaign(campaign, root=args.out_dir, timeout_s=args.timeout_s)
    payload = {
        "kind": "pct008_artifact_cold_warm/v1",
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
                "scripts.run_pct008_artifact_cold_warm",
                "--mode",
                "fixture",
                "--cold-trials-per-arm",
                str(args.cold_trials_per_arm),
                "--warm-trials",
                str(args.warm_trials),
            ],
            "docs_path": str(args.docs_out),
        },
        "domain_parity": result["domain_parity"],
        "summary": result["summary"],
        "phase_breakdown_by_arm": result["phase_breakdown_by_arm"],
        "trials": result["trials"],
        "noise_floor_ms": result["noise_floor_ms"],
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
