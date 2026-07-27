#!/usr/bin/env python3
"""Evaluate SLM-251 (LOT1-02)'s launch prerequisites in plan-only mode.

No training, curriculum, or model code is loaded or executed. This reads
the real committed upstream contract artifacts and emits the
``Lot102LaunchGateV1`` disposition.

Example:
  python -m scripts.evaluate_lot1_02_activation_gate \
      --fidelity-contract docs/design/lotus-openui-fidelity-contract-v1.json \
      --trace-gate-contract docs/design/compiler-reasoning-trace-v1.json \
      --out outputs/runs/slm251_launch_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.lot1_01_activation_gate import (
    load_upstream_contract,
)
from slm_training.harnesses.experiments.lot1_02_activation_gate import (
    evaluate_lot1_02_launch_gate,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-251 LOT1-02 launch-gate evaluator (plan-only, no training code)"
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
        default=Path("outputs/runs/slm251_launch_gate"),
    )
    args = parser.parse_args(argv)

    fidelity_contract = load_upstream_contract(args.fidelity_contract)
    trace_gate_contract = load_upstream_contract(args.trace_gate_contract)

    contract = evaluate_lot1_02_launch_gate(fidelity_contract, trace_gate_contract)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lot1_02_launch_gate.json").write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(contract)
    (args.out / "lot1_02_launch_gate.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
