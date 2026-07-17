"""G4 (SLM-36) CLI: sketch-vs-direct reasoning bench with checkable answers.

Fixture example:
    python -m scripts.run_reasoning_bench --campaign-id g4_fixture \
        --n-train 96 --n-test 24 --steps 60 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.reasoning import ReasoningBenchConfig, run_reasoning_bench


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="g4_bench")
    parser.add_argument("--n-train", type=int, default=96)
    parser.add_argument("--n-test", type=int, default=24)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/experiments/reasoning_bench")
    )
    args = parser.parse_args()

    summary = run_reasoning_bench(
        ReasoningBenchConfig(
            campaign_id=args.campaign_id,
            n_train=args.n_train,
            n_test=args.n_test,
            steps=args.steps,
            seed=args.seed,
            device=args.device,
            d_model=args.d_model,
            output_root=args.output_root,
        )
    )
    print(
        json.dumps(
            {
                "campaign_id": summary["campaign_id"],
                "n_test": summary["n_test"],
                "sketch_answer_accuracy": summary["sketch"]["answer_accuracy"],
                "sketch_trace_validity": summary["sketch"]["trace_validity_rate"],
                "direct_answer_accuracy": summary["direct"]["answer_accuracy"],
                "note": summary["note"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
