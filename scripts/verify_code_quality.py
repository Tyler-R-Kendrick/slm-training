#!/usr/bin/env python3
"""CI certificate for module size, cyclomatic complexity, and package structure.

Enforces three things a 700k-line codebase cannot hold by convention alone:

1. **Module size.** No module over ``MAX_MODULE_LINES`` (400) physical lines.
2. **Complexity / SRP.** Ruff's mccabe (``C901``) and pylint function-shape
   rules (``PLR0911/0912/0913/0915``).
3. **Package structure.** Robert C. Martin's ADP (no dependency cycles), SDP
   (depend toward stability) and SAP (stable components must be abstract), plus
   a per-component size budget for the Common Closure Principle.

The repository does not satisfy these today, so the gate is a **ratchet**: the
current debt is frozen in ``src/slm_training/resources/code_quality_baseline.json``
and may only shrink. A change that makes any number worse fails; a change that
makes one better must write the improvement back with ``--update``, permanently
lowering the ceiling. The Boy Scout rule, mechanised.

Usage::

    python -m scripts.verify_code_quality            # gate (CI)
    python -m scripts.verify_code_quality --report   # metrics, no gating
    python -m scripts.verify_code_quality --update   # re-freeze the baseline

Contract: ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from slm_training.quality import baseline, gate, report  # noqa: E402
from slm_training.quality.complexity import RuffUnavailableError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="gate against the baseline (the default when no mode is given)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="re-freeze the baseline from current measurements",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the full metrics report without gating",
    )
    parser.add_argument(
        "--skip-complexity",
        action="store_true",
        help="skip the ruff pass (for environments without ruff on PATH)",
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root to analyse"
    )
    return parser


def _analyse(args: argparse.Namespace) -> gate.QualityReport:
    try:
        return gate.analyse(root=args.root, skip_complexity=args.skip_complexity)
    except RuffUnavailableError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def _emit(outcome: baseline.RatchetOutcome) -> None:
    for entry in outcome.failures:
        print(f"  FAIL  {entry}")
    for entry in outcome.drift:
        print(f"  DRIFT {entry}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analysis = _analyse(args)

    if args.report:
        print(report.render(analysis))
        return 0

    recorded = baseline.load(root=args.root)

    if args.update:
        written = baseline.save(
            gate.merged_dimensions(analysis, recorded), root=args.root
        )
        print(report.render(analysis))
        print(f"\nbaseline written: {written.relative_to(args.root)}")
        return 0

    if not recorded:
        print(
            "error: no baseline recorded. Run "
            "`python -m scripts.verify_code_quality --update` to freeze current debt.",
            file=sys.stderr,
        )
        return 2

    outcomes = gate.evaluate(analysis, recorded)
    failures = sum(len(item.failures) for item in outcomes)
    drift = sum(len(item.drift) for item in outcomes)

    for outcome in outcomes:
        if outcome.clean:
            continue
        print(f"\n{outcome.dimension}:")
        _emit(outcome)

    if failures:
        print(
            f"\n{failures} regression(s): code quality got worse. "
            "Split the module, simplify the function, or break the dependency "
            "cycle -- do not raise the baseline by hand.",
            file=sys.stderr,
        )
        return 1
    if drift:
        print(
            f"\n{drift} improvement(s) not yet recorded. Run "
            "`python -m scripts.verify_code_quality --update` to lock them in.",
            file=sys.stderr,
        )
        return 1

    print("code quality: no regressions against the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
