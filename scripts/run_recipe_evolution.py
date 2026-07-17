"""G2 (SLM-35) CLI: evolve training recipes under frozen honest gates.

Fixture example:
    python -m scripts.run_recipe_evolution \
        --campaign-id g2_fixture --train-dir outputs/data/train/v1 \
        --test-dir outputs/data/eval/v1 --steps 20 --generations 2 \
        --population 3 --device cpu

`--dry-run` builds and persists the seeded population plan without training.
HF mirroring of the campaign tree stays external and optional
(`autoresearch.persistence.sync_campaign`, dry by default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.recipe_evolution import (
    CandidateResult,
    EvolutionConfig,
    RecipeGene,
    run_evolution,
    train_eval_evaluator,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/experiments")
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=3)
    parser.add_argument("--elite-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--suites",
        default="smoke,held_out",
        help="fitness suites; gate check still fails closed on missing suites",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="rank the seeded population without training (wiring check)",
    )
    args = parser.parse_args()

    config = EvolutionConfig(
        campaign_id=args.campaign_id,
        population_size=args.population,
        generations=args.generations,
        elite_k=min(args.elite_k, args.population),
        seed=args.seed,
        output_root=args.output_root,
    )
    if args.dry_run:
        def evaluator(gene: RecipeGene, generation: int, slot: int) -> CandidateResult:
            return CandidateResult(
                gene=gene, fitness=None, gates_pass=False,
                gate_failures=("dry_run:not_evaluated",),
            )
    else:
        evaluator = train_eval_evaluator(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            run_root=args.run_root,
            campaign_id=args.campaign_id,
            steps=args.steps,
            device=args.device,
            suites=tuple(s.strip() for s in args.suites.split(",") if s.strip()),
            seed=args.seed,
        )
    summary = run_evolution(config, evaluator)
    print(json.dumps({
        "campaign_id": summary["campaign_id"],
        "promotable": summary["promotable"],
        "best": summary["best"],
        "note": summary["note"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
