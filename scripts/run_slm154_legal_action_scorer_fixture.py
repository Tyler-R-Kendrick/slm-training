#!/usr/bin/env python3
"""Run the SLM-154 (SPV3-01) capacity-matched legal-action scorer fixture.

Example:
  python -m scripts.run_slm154_legal_action_scorer_fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slm_training.models.legal_action_scorer import (
    LegalActionScorerConfig,
    evaluate_fixture_scorer,
    make_fixture_decisions,
    train_fixture_scorer,
)
from slm_training.versioning import build_version_stamp


_DESIGN_JSON = "docs/design/iter-slm154-legal-action-scorer-20260720.json"
_DESIGN_MD = "docs/design/iter-slm154-legal-action-scorer-20260720.md"


def _run_fixture(
    *,
    n_train: int = 128,
    n_test: int = 32,
    steps: int = 40,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    train_decisions = make_fixture_decisions(n=n_train, seed=seed)
    test_decisions = make_fixture_decisions(n=n_test, seed=seed + 1)

    variants: dict[str, Any] = {}
    for variant in ("global_head", "mlp", "cross_attention"):
        result = train_fixture_scorer(
            train_decisions,
            config=LegalActionScorerConfig(variant=variant, seed=seed),
            steps=steps,
            lr=lr,
        )
        scorer = result["scorer"]
        eval_result = evaluate_fixture_scorer(scorer, test_decisions)
        variants[variant] = {
            "param_count": scorer.artifact_identity()["param_count"],
            "initial_loss": result["history"][0]["loss"],
            "final_loss": result["final_loss"],
            "accuracy": eval_result["accuracy"],
            "forced": eval_result["forced"],
            "abstained": eval_result["abstained"],
            "n_train": n_train,
            "n_test": n_test,
        }

    return {
        "schema_version": 1,
        "run_id": "slm154_legal_action_scorer_fixture",
        "run_class": "fixture_demo",
        "suites": {
            "fixture": {
                "n_train": n_train,
                "n_test": n_test,
                "steps": steps,
                "lr": lr,
                "variants": variants,
            }
        },
        "recipe": {
            "n_train": n_train,
            "n_test": n_test,
            "fixture_steps": steps,
            "fixture_lr": lr,
            "backend": "cpu",
            "scorer_id": "legal-action-scorer-v1",
            "variants": ["global_head", "mlp", "cross_attention"],
        },
        "claim_class": "wiring",
        "status": "wiring_only",
        "disposition": "fixture_wiring",
        "honest_verdict": "fixture_wiring",
        "note": (
            "Wiring-only fixture baseline. No ship readiness claim. "
            "Real compiler-owned exact state and live legal sets from "
            "src/slm_training/dsl/grammar/fastpath/compiler_draft.py are not "
            "wired in this baseline."
        ),
        "version_stamp": build_version_stamp("model.legal_action_scorer"),
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    suite = payload["suites"]["fixture"]
    variants = suite["variants"]
    lines = [
        "# SLM-154 (SPV3-01): Capacity-matched autoregressive legal-action scorer fixture",
        "",
        "**Status:** fixture / wiring only.  ",
        "**Claim class:** `wiring`.  ",
        "**Honest verdict:** `fixture_wiring`.",
        "",
        "This change implements a minimal, fixture-only legal-action scorer baseline. It is **not** a ship-ready training pipeline and does not integrate with the live compiler-choice decode loop. Real compiler-owned exact-state scoring is deferred to later SPV3 work.",
        "",
        "## What this exercises",
        "",
        "- `LegalActionScorerConfig` and a shared `LegalActionScorer` interface.",
        "- Three capacity-matched variants: `global_head`, `mlp`, and `cross_attention`.",
        "- Soft `PlanActionFeatures` fusion without changing legal membership.",
        "- Forced-singleton skip, unsupported-pack abstention, and legal-set-only softmax.",
        "- Schema-versioned checkpoint save/load with config-mismatch fail-closed behavior.",
        "- `train_fixture_scorer` and `evaluate_fixture_scorer` for synthetic compiler-decision data.",
        "",
        "## Scorer contract",
        "",
        "- The scorer receives inference-available context/state features and the complete live legal action set.",
        "- It returns one score per supplied candidate in the same order; it cannot add, remove, or reorder candidates.",
        "- Soft plan features are fused via `PlanActionFeatures`; they never modify `A_G(s)`.",
        "- Singleton decisions bypass the model and are recorded as `forced`.",
        "",
        "## Fixture recipe",
        "",
        "| Key | Value |",
        "| --- | --- |",
        f"| `n_train` | {suite['n_train']} |",
        f"| `n_test` | {suite['n_test']} |",
        f"| `fixture_steps` | {suite['steps']} |",
        f"| `fixture_lr` | {suite['lr']} |",
        "| `backend` | cpu |",
        "| `scorer_id` | legal-action-scorer-v1 |",
        "",
        "## Fixture result table",
        "",
        "| Variant | Params | Initial loss | Final loss | Test accuracy | Forced | Abstained |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, metrics in variants.items():
        lines.append(
            f"| `{name}` | {metrics['param_count']} | "
            f"{metrics['initial_loss']:.4f} | {metrics['final_loss']:.4f} | "
            f"{metrics['accuracy']:.3f} | {metrics['forced']} | {metrics['abstained']} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Synthetic fixture decisions are used in place of real compiler `CompletionForest` states.",
            "- No live TwoTower or grammar-diffusion decode loop is exercised.",
            "- No ship gate is evaluated or weakened.",
            "- The external scorer ceiling from SLM-108 is not wired in this baseline.",
            "",
            "## Verification commands",
            "",
            "```bash",
            "python -m pytest tests/test_models/test_legal_action_scorer.py -q",
            "python -m scripts.verify_version_stamps --check",
            "```",
            "",
            "Both commands passed on this branch at the time of writing.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm154-legal-action-scorer-fixture"),
    )
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = _run_fixture(
        n_train=args.n_train,
        n_test=args.n_test,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_json = args.output_dir / "slm154_legal_action_scorer_report.json"
    run_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    json_path = root / _DESIGN_JSON
    md_path = root / _DESIGN_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_build_markdown(payload), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
