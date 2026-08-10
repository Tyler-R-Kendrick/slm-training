"""SIE-006 (SLM-483): run EXP-SR-3 diagnostic evaluator construct-validity
and human calibration campaign.

    python -m scripts.run_sie006_calibration_campaign --mode plan-only
    python -m scripts.run_sie006_calibration_campaign --mode diagnostic \\
        --out-dir outputs/runs/sie006_calibration_campaign \\
        --docs-out docs/design/iter-slm483-sie-006-calibration-campaign-20260810.json

``--mode plan-only`` previews the SIE-001 ``exp-sr-3`` catalogue identity.
``--mode diagnostic`` runs VCE-010's evaluator-calibration protocol through
that locked identity and writes durable version-stamped results under
``docs/design/``. Every result carries an explicit scope_disclaimer: the
external judge and human raters are labeled mocks (this environment has no
real external-judge credentials or human raters), so 'authorized' here is
never a substitute for a genuine construct-validity claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.sie006_calibration_campaign import (
    Sie006CampaignV1,
    plan_only_preview,
    run_campaign,
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    lines = [
        "# SLM-483 (SIE-006): Evaluator construct-validity and human calibration campaign (EXP-SR-3)",
        "",
        "**Claim class:** `diagnostic` (matches the locked exp-sr-3 catalogue entry)",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-3')}`",
        "",
        f"**Primary metric (`evaluator_human_agreement_kappa`):** {payload.get('evaluator_human_agreement_kappa')}",
        "",
        f"**minimum_effect:** `{payload.get('minimum_effect')}` -- kill_threshold=`{payload.get('kill_threshold')}` (le)",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| meets_minimum_effect | {payload.get('meets_minimum_effect')} |",
        f"| killed_by_catalogue_gate | {payload.get('killed_by_catalogue_gate')} |",
        f"| independence_ok | {payload.get('independence_ok')} |",
        f"| authorized | {payload.get('authorized')} |",
        f"| pair_n | {payload.get('pair_n')} |",
        f"| vce010_disposition | {payload.get('vce010_disposition')} |",
        f"| external_human_calibration_error | {payload.get('external_human_calibration_error')} |",
        f"| admission_divergence_rate | {payload.get('admission_divergence_rate')} |",
        f"| promotion | {payload.get('promotion')} |",
        "",
        "## Scope",
        "",
        payload.get("scope_disclaimer", ""),
        "",
        "Command: `python -m scripts.run_sie006_calibration_campaign --mode diagnostic`",
        "",
        f"Full detail: `{docs_json.as_posix()}`.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "diagnostic"), default="plan-only")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/runs/sie006_calibration_campaign")
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/design/iter-slm483-sie-006-calibration-campaign-20260810.json"),
    )
    parser.add_argument("--protocol-seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = Sie006CampaignV1(protocol_seed=args.protocol_seed)
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "sie006_calibration_campaign_diagnostic/v1",
        **{k: v for k, v in result.items() if k != "schema"},
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_sie006_calibration_campaign",
                "--mode",
                "diagnostic",
                "--protocol-seed",
                str(args.protocol_seed),
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "diagnostic_mock_evidence",
        },
        "version_stamp": build_version_stamp("harness.experiments"),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload)
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(f"evaluator_human_agreement_kappa={payload['evaluator_human_agreement_kappa']}")
    print(f"authorized={payload['authorized']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
