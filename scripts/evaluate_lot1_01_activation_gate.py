#!/usr/bin/env python3
"""Evaluate SLM-250 (LOT1-01)'s hard activation gates in plan-only mode.

No model, training, or K x c workspace code is loaded or executed. This
reads the two real committed upstream contract artifacts and emits the
required ``LotusOpenUIModelContractV1`` disposition.

Example:
  python -m scripts.evaluate_lot1_01_activation_gate \
      --fidelity-contract docs/design/lotus-openui-fidelity-contract-v1.json \
      --trace-gate-contract docs/design/compiler-reasoning-trace-v1.json \
      --out outputs/runs/slm250_activation_gate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.lot1_01_activation_gate import (
    evaluate_activation_gates,
    load_upstream_contract,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-250 LOT1-01 hard-activation-gate evaluator (plan-only, no model code)"
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
        default=Path("outputs/runs/slm250_activation_gate"),
    )
    args = parser.parse_args(argv)

    fidelity_contract = load_upstream_contract(args.fidelity_contract)
    trace_gate_contract = load_upstream_contract(args.trace_gate_contract)

    contract = evaluate_activation_gates(fidelity_contract, trace_gate_contract)

    args.out.mkdir(parents=True, exist_ok=True)
    contract.to_json(args.out / "lotus_openui_model_contract.json")
    markdown = render_markdown(contract)
    (args.out / "lotus_openui_model_contract.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
