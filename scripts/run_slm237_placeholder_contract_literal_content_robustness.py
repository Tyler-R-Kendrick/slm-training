#!/usr/bin/env python3
"""Run the SLM-237 (PCR0-01) placeholder-contract literal-content robustness
probe.

Exercises the real, unmodified ``extract_placeholders`` (``dsl/
placeholders.py``), ``_contract_precision`` / ``_contract_recall`` /
``_placeholder_fidelity`` (``harnesses/model_build/eval_runner.py``), and
``score_openui`` RL-reward contract (``integrations/openui_rl.py``) against
synthetic, grammar-validated OpenUI document pairs that share identical DSL
structure and differ only in leaf-literal text, to check whether ordinary
literal content can silently contaminate placeholder-contract metrics --
including awarding full false credit for an omitted, unfilled placeholder.

Examples:
  python -m scripts.run_slm237_placeholder_contract_literal_content_robustness --mode plan-only
  python -m scripts.run_slm237_placeholder_contract_literal_content_robustness --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm237_placeholder_contract_literal_content_robustness import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    PlaceholderContractRobustnessReport,
    render_markdown,
    run_robustness_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm237-pcr0-01-placeholder-contract-literal-content-robustness-20260721.json"
_DESIGN_MD = "docs/design/iter-slm237-pcr0-01-placeholder-contract-literal-content-robustness-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "PlaceholderContractRobustnessReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "extract_placeholders scans the raw source text with no "
            "string-literal-boundary awareness, so a colon-prefixed dotted "
            "token inside ordinary literal text is scored identically to a "
            "real placeholder reference, which can both spuriously lower "
            "contract precision and award false credit for an omitted "
            "placeholder."
        ),
        "falsifier": (
            "An unrelated placeholder-shaped literal mention never lowers "
            "contract precision when the real contract is fully filled; or "
            "an omitted real placeholder is always correctly penalized "
            "regardless of incidental literal-text mentions."
        ),
        "rows": [],
        "reward_probe_rows": [],
        "reward_probe_shape": "single_leaf_card",
        "gate_hash": "",
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m "
            "scripts.run_slm237_placeholder_contract_literal_content_robustness "
            "--mode fixture` to execute."
        ),
        "honest_caveats": [
            "Plan-only: no row was scored.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm237_placeholder_contract_literal_content_robustness",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-237 PCR0-01 placeholder-contract literal-content robustness probe",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="fixture",
        help="Run mode (default: fixture).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--write-design-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write design docs in fixture mode (default: True).",
    )
    parser.add_argument("--design-json", type=Path, default=None)
    parser.add_argument("--design-md", type=Path, default=None)

    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    args.output_dir = args.output_dir or Path(
        f"outputs/runs/slm237-placeholder-contract-literal-content-robustness-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload()
    else:
        report = run_robustness_fixture(run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}")
        payload = report.to_dict()

    run_json = (
        args.output_dir
        / "slm237_placeholder_contract_literal_content_robustness_report.json"
    )
    run_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if args.mode == "fixture" and args.write_design_docs:
        root = Path(__file__).resolve().parents[1]
        json_path = args.design_json or root / _DESIGN_JSON
        md_path = args.design_md or root / _DESIGN_MD
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(
            render_markdown(PlaceholderContractRobustnessReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
