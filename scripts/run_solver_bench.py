"""VSS4-01 solver benchmark CLI (SLM-74).

Thin wrapper over ``slm_training.harnesses.solver_bench``. Runs the committed v1
finite fixture, cross-checking the reference ``EnumerativeSupportOracle`` against
an independent brute-force enumerator, and prints a JSON report. No model, no
network. A non-zero exit code signals a hard benchmark failure (a false certified
prune, an unknown-preservation violation, a certificate replay failure, or a
disagreement with independent ground truth).

    python -m scripts.run_solver_bench --describe
    python -m scripts.run_solver_bench --all
"""

from __future__ import annotations

import argparse
import json
import sys

from slm_training.harnesses.solver_bench import build_reference_fixture, run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verified-scope solver benchmark (VSS4-01).")
    parser.add_argument("--describe", action="store_true", help="list cases without running")
    parser.add_argument("--all", action="store_true", help="run the full v1 suite")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    fx = build_reference_fixture()

    if args.describe or not args.all:
        payload = {
            "suite": "verified_scope_solver/v1",
            "pack_id": fx.expander.pack_id,
            "constraint_version": fx.expander.constraint_version,
            "cases": [
                {"case_id": c.case_id, "family": c.family, "expected": c.expected_verdict}
                for c in fx.cases
            ],
        }
        print(json.dumps(payload, indent=args.indent, sort_keys=True))
        return 0

    report = run_suite(fx.oracle, fx.expander, fx.verifier, fx.state, fx.hole_id, fx.cases)
    from slm_training.versioning import build_version_stamp

    payload = report.to_dict()
    payload["version_stamp"] = build_version_stamp("harness.solver_bench")
    print(json.dumps(payload, indent=args.indent, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
