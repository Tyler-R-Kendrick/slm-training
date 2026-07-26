#!/usr/bin/env python3
"""Evaluate a LOT2+ issue's downstream activation gate in plan-only mode.

No training, factorial, intervention, or systems code is loaded or run.
Reads the real committed upstream contract artifacts and emits a per-issue
``LotDownstreamGateV1`` disposition.

Example:
  python -m scripts.evaluate_lot_downstream_gate --issue SLM-252 \
      --out outputs/runs/slm252_downstream_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.lot_downstream_gate import (
    DOWNSTREAM_ISSUES,
    evaluate_downstream_gate,
    render_markdown,
)
from slm_training.harnesses.experiments.lot1_01_activation_gate import (
    load_upstream_contract,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LOT2+ downstream launch-gate evaluator (plan-only, no campaign code)"
    )
    parser.add_argument(
        "--issue",
        required=True,
        choices=sorted(DOWNSTREAM_ISSUES),
        help="Linear issue id to evaluate (registry: DOWNSTREAM_ISSUES)",
    )
    parser.add_argument(
        "--fidelity-contract",
        type=Path,
        default=Path("docs/design/lotus-openui-fidelity-contract-v1.json"),
        help="SLM-248 LotusOpenUIFidelityContractV1 JSON artifact",
    )
    parser.add_argument(
        "--trace-gate-contract",
        type=Path,
        default=Path("docs/design/compiler-reasoning-trace-v1.json"),
        help="SLM-249 CompilerReasoningTraceGateV1 JSON artifact",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default outputs/runs/<issue>_downstream_gate)",
    )
    args = parser.parse_args(argv)

    spec = DOWNSTREAM_ISSUES[args.issue]
    out = args.out or Path("outputs/runs") / f"{args.issue.lower()}_downstream_gate"

    contract = evaluate_downstream_gate(
        spec,
        load_upstream_contract(args.fidelity_contract),
        load_upstream_contract(args.trace_gate_contract),
    )

    out.mkdir(parents=True, exist_ok=True)
    stem = spec.slug.replace("-", "_")
    (out / f"{stem}_gate.json").write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(contract)
    (out / f"{stem}_gate.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
