#!/usr/bin/env python3
"""RESEARCH-08 (SLM-540): Myhill–Nerode static residual quotient runner.

    PYTHONPATH=src uv run python -m scripts.run_research08_myhill_nerode --mode plan-only
    PYTHONPATH=src uv run python -m scripts.run_research08_myhill_nerode --mode fixture \\
        --out-dir outputs/runs/research08_myhill_nerode \\
        --docs-out docs/design/iter-revmath-research-08-preregistered.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.research08_myhill_nerode import (
    Research08CampaignV1,
    plan_only_preview,
    run_campaign,
)

_VERSION_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm540_research08_myhill_nerode",
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    pilot = payload.get("pilot") or {}
    primary = pilot.get("primary_metric") or {}
    control = pilot.get("control") or {}
    treatment = pilot.get("treatment") or {}
    lines = [
        "# RESEARCH-08 (SLM-540): Myhill–Nerode minimization of the static residual",
        "",
        "**Claim class:** `fixture` / research-only (default_off; never production)",
        "",
        f"**Decision:** `{payload.get('recommendation')}`",
        "",
        f"**Zero semantic disagreement:** `{pilot.get('zero_semantic_disagreement')}`",
        "",
        f"**Promotion:** `{payload.get('promotion')}`",
        "",
        "## Gate delta",
        "",
        "| Metric | Control | Treatment | Delta |",
        "| --- | --- | --- | --- |",
        f"| cold_load_p50_ms | {primary.get('control_p50_ms')} | "
        f"{primary.get('treatment_p50_ms')} | {primary.get('delta_ms')} |",
        f"| state_count | {control.get('n_states')} | {treatment.get('n_states')} | "
        f"{treatment.get('state_reduction')} |",
        f"| artifact_bytes | {control.get('bytes')} | {treatment.get('bytes')} | "
        f"{(control.get('bytes') or 0) - (treatment.get('bytes') or 0)} |",
        f"| language_equivalence_failures | — | "
        f"{(pilot.get('language_equivalence') or {}).get('disagreement_count')} | — |",
        f"| complete_domain_parity | — | "
        f"{(pilot.get('complete_domain_parity') or {}).get('parity_ok')} | — |",
        "",
        "## Scope",
        "",
        "Terminal-continuation singleton-stack DFA abstraction of the certified "
        "static LALR residual adapter. Request-local overlays excluded. "
        "Default-off; no production decode authority.",
        "",
        "Command: `PYTHONPATH=src uv run python -m scripts.run_research08_myhill_nerode "
        "--mode fixture`",
        "",
        f"Full detail: `{docs_json.as_posix()}`.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/research08_myhill_nerode"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/design/iter-revmath-research-08-preregistered.json"),
    )
    parser.add_argument("--cold-trials", type=int, default=31)
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        preview = plan_only_preview(Research08CampaignV1(cold_trials=args.cold_trials))
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    campaign = Research08CampaignV1(cold_trials=args.cold_trials, enabled=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run_campaign(campaign, root=args.out_dir)
    stamp = build_version_stamp(*_VERSION_COMPONENTS)
    payload = {
        **result,
        "version_stamp": stamp,
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "scope_disclaimer": (
            "Fixture-scale RESEARCH-08 evidence for Myhill–Nerode quotienting of "
            "the terminal-continuation abstraction of the static residual graph. "
            "claim_class=fixture; default_off; never promotion_candidate/ship_gate."
        ),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = _write_markdown(args.docs_out, payload)
    (args.out_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "docs_json": str(args.docs_out),
                "docs_md": str(md),
                "decision": payload.get("recommendation"),
                "manifest_sha256": payload.get("manifest_sha256"),
                "zero_semantic_disagreement": (payload.get("pilot") or {}).get(
                    "zero_semantic_disagreement"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
