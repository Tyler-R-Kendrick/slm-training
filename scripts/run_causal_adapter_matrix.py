"""CLI for the LDI1-03 causal adapter/objective campaign matrix (SLM-122).

Thin wrapper over :mod:`slm_training.harnesses.preference.causal_adapter_matrix`.
By default it describes the arm matrix (dry-run) and classifies each arm against a
declared corpus-support profile — it runs no training. Ship-grade execution
requires GPU, a pinned causal checkpoint, and an admitted DecisionEventV2 corpus;
without those every trainable arm resolves to ``expired`` rather than a fabricated
result. No quality claim is produced here.

Examples::

    python scripts/run_causal_adapter_matrix.py --describe
    python scripts/run_causal_adapter_matrix.py --stage 0 --corpus-admitted \\
        --has-pairs --out outputs/data/ldi_causal_matrix/plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.harnesses.preference.causal_adapter_matrix import (
    CampaignConfig,
    CorpusSupport,
    build_stage0,
    build_stage1,
    build_stage2,
    describe_campaign,
    run_arm,
)


def _build_arms(args: argparse.Namespace, config: CampaignConfig) -> list[Any]:
    arms = list(build_stage0(config))
    if args.stage in (None, 1, 2):
        arms += build_stage1(config, best_objective=args.best_objective)
    if args.stage in (None, 2):
        arms += build_stage2(
            config, best_objective=args.best_objective, best_rank=args.best_rank
        )
    if args.stage is not None:
        arms = [a for a in arms if a.stage == args.stage]
    return arms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="dry-run: emit the matrix only")
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--best-objective", default="ftpo_single")
    parser.add_argument("--best-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-model-id", default="")
    parser.add_argument("--base-model-revision", default="")
    parser.add_argument("--corpus-admitted", action="store_true")
    parser.add_argument("--has-pairs", action="store_true")
    parser.add_argument("--has-set-valued", action="store_true")
    parser.add_argument("--allow-experimental-methods", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = parser.parse_args(argv)

    config = CampaignConfig(
        base_model_id=args.base_model_id,
        base_model_revision=args.base_model_revision,
        allow_experimental_methods=args.allow_experimental_methods,
    )
    arms = _build_arms(args, config)
    corpus = CorpusSupport(
        admitted=args.corpus_admitted,
        has_pairs=args.has_pairs,
        has_set_valued=args.has_set_valued,
    )

    payload: dict[str, Any] = describe_campaign(arms)
    if not args.describe:
        # Classify (and, in this environment, expire) every arm without training.
        results = [
            run_arm(
                arm,
                corpus=corpus,
                seed=args.seed,
                allow_experimental_methods=args.allow_experimental_methods,
            ).as_dict()
            for arm in arms
        ]
        status_counts: dict[str, int] = {}
        for res in results:
            status_counts[res["status"]] = status_counts.get(res["status"], 0) + 1
        payload["results"] = results
        payload["status_counts"] = {k: status_counts[k] for k in sorted(status_counts)}

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
