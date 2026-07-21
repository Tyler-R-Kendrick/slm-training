"""CLI for the SLM-261 bounded memorization probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.harnesses.experiments.slm261_memorization_probe import (
    ARM_NAMES,
    run_memorization_probe_fixture,
)


def _parse_arm_list(value: str) -> tuple[str, ...]:
    arms = tuple(a.strip() for a in value.split(",") if a.strip())
    unknown = set(arms) - set(ARM_NAMES)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    return arms


def _parse_seed_list(value: str) -> tuple[int, ...]:
    return tuple(int(s.strip()) for s in value.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-261 bounded memorization probe fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to the source corpus records.jsonl (required unless --describe).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/slm261_memorization_probe"),
        help="Directory for the fixture report and corruption suite.",
    )
    parser.add_argument(
        "--arms",
        type=_parse_arm_list,
        default=",".join(ARM_NAMES),
        help=f"Comma-separated arms (default: {','.join(ARM_NAMES)}).",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default="0",
        help="Comma-separated seeds (default: 0).",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=5,
        help="Number of records to select for the fixture corpus.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Training steps per arm (kept small for the fixture).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the tiny fast model config.",
    )
    parser.add_argument(
        "--write-design-docs",
        action="store_true",
        help="Write docs/design/iter-slm261-memorization-probe-*.json and .md.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the varied fields and exit without running training.",
    )

    args = parser.parse_args(argv)

    if not args.describe and args.corpus is None:
        parser.error("--corpus is required unless --describe")

    if args.describe:
        print("SLM-261 memorization probe varied fields:")
        print(f"  arms={args.arms}")
        print(f"  seeds={args.seeds}")
        print(f"  n_records={args.n_records}")
        print(f"  steps={args.steps}")
        print(f"  lr={args.lr}")
        print(f"  fast={args.fast}")
        print("All other ModelBuildConfig fields are pinned by the harness.")
        return 0

    manifest = run_memorization_probe_fixture(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        arms=args.arms,
        seeds=args.seeds,
        n_records=args.n_records,
        steps=args.steps,
        lr=args.lr,
        fast=args.fast,
        write_design_docs=args.write_design_docs,
        version_components=("harness.experiments", "model.twotower"),
    )

    print(f"Run ID: {manifest.run_id}")
    print(f"Disposition: {manifest.disposition}")
    print(f"Report: {args.output_dir / 'report.json'}")
    for arm in manifest.arms:
        print(
            f"  {arm.arm_name} seed={arm.seed}: "
            f"loss={arm.final_reported_loss}, "
            f"ledger_error={arm.final_loss_ledger_reconciliation_error}, "
            f"exact_acc={arm.exact_target_accuracy}, "
            f"recon={arm.canonical_reconstruction_rate}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
