"""PGS-H01 (SLM-510): run goal-support domain-adequacy fixture campaign.

    python -m scripts.run_goal_support_domain_adequacy --mode plan-only
    python -m scripts.run_goal_support_domain_adequacy --mode fixture \\
        --out-dir outputs/runs/pgs_h01_goal_support_domain_adequacy \\
        --docs-out docs/design/iter-slm510-pgs-h01-goal-support-domain-adequacy-20260811.json

``--mode plan-only`` previews the preregistered manifest and fixture matrix.
``--mode fixture`` runs the bounded diagnostic campaign and writes durable
version-stamped results under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.goal_support_domain_adequacy import (
    GoalSupportDomainAdequacyCampaignV1,
    RESULT_SCHEMA,
    plan_only_preview,
    render_markdown,
    run_campaign,
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/pgs_h01_goal_support_domain_adequacy"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm510-pgs-h01-goal-support-domain-adequacy-20260811.json"
        ),
    )
    parser.add_argument(
        "--claim-class",
        choices=("diagnostic", "fixture"),
        default="diagnostic",
    )
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = GoalSupportDomainAdequacyCampaignV1(claim_class=args.claim_class)
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": RESULT_SCHEMA,
        "claim_class": result["claim_class"],
        "promotion": False,
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_goal_support_domain_adequacy",
                "--mode",
                "fixture",
                "--claim-class",
                args.claim_class,
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "fixture_diagnostic",
        },
        **result,
        "version_stamp": build_version_stamp("harness.experiments"),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload)
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(f"result_digest={payload['result_digest']}")
    print(f"falsifier_holds={payload['falsifier']['falsifier_holds']}")
    print(f"false_hard_prune_count={payload['falsifier']['false_hard_prune_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
