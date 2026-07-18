#!/usr/bin/env python3
"""Run the VSS4-03 verified-scope-solver campaign (SLM-76).

CPU-runnable phases execute and produce durable JSON/Markdown evidence; every
frontier phase/row that requires a missing checkpoint, benchmark family, or
training CLI is marked blocked with an explicit reason.

    python -m scripts.run_vss4_campaign --describe
    python -m scripts.run_vss4_campaign --out-dir outputs/runs/vss4_03_campaign
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.vss4_campaign import (
    describe_vss4_campaign,
    render_markdown,
    run_vss4_campaign,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Resolve phase statuses and artifact lock without running benchmarks.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for campaign.json and campaign.md.",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    report = describe_vss4_campaign() if args.describe else run_vss4_campaign()

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "campaign.json"
        md_path = out_dir / "campaign.md"
        json_path.write_text(report.to_json(indent=args.indent), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote {json_path} and {md_path}")
    else:
        print(report.to_json(indent=args.indent))

    print(
        f"campaign={report.campaign_id} run_id={report.run_id} "
        f"phases={len(report.phases)} blocked={len(report.blocked_phases)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
