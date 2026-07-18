#!/usr/bin/env python3
"""Run the VSS4-02 verified-scope-solver matrix (SLM-75).

Adds solver / capsule / energy / topology / surface metrics and fail-closed
correctness gates to the matched experiment-matrix system. This runner is the
``verified-solver`` matrix set; ``scripts/run_quality_matrix.py --matrix-set
verified-solver`` delegates to the same functions.

    # resolve every row config + capability without running any model/benchmark
    python -m scripts.run_verified_solver_matrix --describe

    # CPU fixture run over the committed VSS4-01 benchmark, writing JSON + Markdown
    python -m scripts.run_verified_solver_matrix --fixture --out-dir outputs/runs/vss4_02

Fixture rows R0-R1 are computed with independent ground truth from ``solver_bench``;
rows R2-R6 are fully specified but ``not_run`` (frontier execution is VSS4-03). A
non-zero exit code signals a fail-closed hard-gate violation (a false certified
prune, an unknown-preservation violation, a certificate replay failure, an
unverified solved result, a candidate-set parity failure, a semantic-IR mutation, or
a structured/observable slot routed to the AR realizer).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.verified_solver_matrix import (
    MATRIX_SET,
    describe_matrix,
    render_markdown,
    run_fixture_matrix,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-set",
        default=MATRIX_SET,
        help=f"Matrix set/tag (default: {MATRIX_SET}).",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Resolve row configs/capabilities without running any model or benchmark.",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run the CPU fixture matrix over the committed VSS4-01 benchmark.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    if args.matrix_set != MATRIX_SET:
        parser.error(f"unknown matrix set {args.matrix_set!r}; expected {MATRIX_SET!r}")

    # --describe never runs models/data; default (no flag) also describes.
    report = run_fixture_matrix() if args.fixture and not args.describe else describe_matrix()

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"vss4_02_{report.mode}_{report.run_id}.json"
        md_path = out_dir / f"vss4_02_{report.mode}_{report.run_id}.md"
        json_path.write_text(report.to_json(indent=args.indent), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote {json_path} and {md_path}")
    else:
        print(report.to_json(indent=args.indent))

    print(
        f"mode={report.mode} rows={len(report.rows)} "
        f"gate_failures={len(report.gate_failures)} passed={report.passed}"
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
