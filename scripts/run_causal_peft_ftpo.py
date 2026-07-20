#!/usr/bin/env python3
"""Run the SLM-121 LDI1-02 causal PEFT FTPO experiment.

Example (plan only, no model load):
  python -m scripts.run_causal_peft_ftpo --mode plan-only \
      --output-dir outputs/runs/slm121_causal_peft_ftpo

Example (fixture wiring check):
  python -m scripts.run_causal_peft_ftpo --mode fixture \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/causal-base/ref.json \
      --output-dir outputs/runs/slm121_causal_peft_fixture

Frontier execution (GPU + durable checkpoint required):
  python -m scripts.run_causal_peft_ftpo --mode frontier \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/causal-base/ref.json \
      --checkpoint-bucket hf://buckets/TKendrick/OpenUI \
      --output-dir outputs/runs/slm121_causal_peft_frontier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.harnesses.experiments.causal_peft_ftpo import (
    FTPO_OBJECTIVES,
    build_causal_peft_ftpo_manifest,
    render_markdown,
    run_fixture_ftpo,
    validate_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-121 causal PEFT FTPO experiment"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture", "frontier"),
        default="plan-only",
        help=(
            "plan-only emits the manifest without loading models; fixture runs a "
            "torch-free wiring check; frontier dispatches the real GPU training (not "
            "implemented in this wiring slice)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm121_causal_peft_ftpo"),
    )
    parser.add_argument(
        "--parent-checkpoint-uri",
        default=None,
        help="Durable URI of the causal base checkpoint",
    )
    parser.add_argument(
        "--checkpoint-bucket",
        default="hf://buckets/TKendrick/OpenUI",
        help="HF bucket for durable adapter checkpoints",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds for the arms",
    )
    parser.add_argument(
        "--objectives",
        default=",".join(FTPO_OBJECTIVES),
        help="Comma-separated FTPO objectives",
    )
    parser.add_argument(
        "--adapter-methods",
        default="lora",
        help="Comma-separated PEFT adapter methods",
    )
    args = parser.parse_args(argv)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    objectives = tuple(o.strip() for o in args.objectives.split(",") if o.strip())
    adapter_methods = tuple(
        m.strip() for m in args.adapter_methods.split(",") if m.strip()
    )

    manifest = build_causal_peft_ftpo_manifest(
        parent_checkpoint_uri=args.parent_checkpoint_uri,
        checkpoint_bucket=args.checkpoint_bucket,
        seeds=seeds,
        objectives=objectives,
        adapter_methods=adapter_methods,
    )
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_json(args.output_dir / "causal_peft_ftpo_manifest.json")

    if args.mode == "plan-only":
        report = run_fixture_ftpo(
            manifest, run_id="slm121_plan", output_dir=args.output_dir
        )
    elif args.mode == "fixture":
        report = run_fixture_ftpo(
            manifest, run_id="slm121_fixture", output_dir=args.output_dir
        )
    else:
        print(
            "frontier mode requires GPU host and durable checkpoints; "
            "emitting fixture plan only",
            file=sys.stderr,
        )
        report = run_fixture_ftpo(
            manifest, run_id="slm121_frontier_partial", output_dir=args.output_dir
        )

    markdown = render_markdown(report)
    (args.output_dir / "causal_peft_ftpo_report.md").write_text(
        markdown, encoding="utf-8"
    )
    report.to_json(args.output_dir / "causal_peft_ftpo_report.json")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
