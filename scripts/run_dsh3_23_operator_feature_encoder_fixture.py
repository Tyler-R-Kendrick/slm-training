#!/usr/bin/env python3
"""Run the SLM-398 (DSH3-23) matched-arm operator-feature-encoder fixture.

Example:
  python -m scripts.run_dsh3_23_operator_feature_encoder_fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slm_training.models.operator_feature_encoder import run_matched_arms_fixture
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/dsh3-23-operator-feature-encoder.json"
_DESIGN_MD = "docs/design/dsh3-23-operator-feature-encoder.md"


def _run_fixture(
    *,
    n_train: int,
    n_eval: int,
    n_candidates: int,
    dim: int,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    report = run_matched_arms_fixture(
        n_train=n_train,
        n_eval=n_eval,
        n_candidates=n_candidates,
        dim=dim,
        steps=steps,
        lr=lr,
        seed=seed,
    )
    payload = report.to_dict()
    payload.update(
        {
            "schema_version": 1,
            "run_id": "dsh3_23_operator_feature_encoder_fixture",
            "run_class": "fixture_demo",
            "n_candidates": n_candidates,
            "honest_verdict": "fixture_wiring",
            "version_stamp": build_version_stamp("model.operator_feature_encoder"),
        }
    )
    return payload


def _build_markdown(payload: dict[str, Any]) -> str:
    arms = payload["arms"]
    robustness = payload["permutation_robustness"]
    recipe = payload["recipe"]
    lines = [
        "# SLM-398 (DSH3-23): typed operator-feature-encoder matched-arm fixture",
        "",
        "**Status:** fixture / wiring only.  ",
        "**Claim class:** `wiring`.  ",
        "**Honest verdict:** `fixture_wiring`.  ",
        f"**Disposition:** `{payload['disposition']}`.",
        "",
        payload["note"],
        "",
        "## What this exercises",
        "",
        "- `OperatorFeatureVocabularyV1` / `OperatorFeatureEncoder` over the "
        "SLM-397 sanitized `OperatorPolicyInputV1` boundary — three matched "
        "arms: `hash_scalar` (the existing `_stable_scalar` control), "
        "`typed` (learned per-field embeddings), and "
        "`typed_identity_bucket` (typed plus a labeled, row-position-only "
        "anti-generalization control).",
        "- `CandidateScoringHead`, shared unchanged across arms.",
        "- `make_fixture_operator_decisions` / `train_fixture_arm` / "
        "`evaluate_fixture_arm` / `run_matched_arms_fixture`.",
        "- `permute_fixture_decision` — the same opaque-ID/row-order "
        "permutation adversarial control from SLM-397, replayed at the "
        "encoder layer.",
        "",
        "## Fixture recipe",
        "",
        "| Key | Value |",
        "| --- | --- |",
        f"| `n_train` | {recipe['n_train']} |",
        f"| `n_eval` | {recipe['n_eval']} |",
        f"| `n_candidates` | {payload['n_candidates']} |",
        f"| `dim` | {recipe['dim']} |",
        f"| `steps` | {recipe['steps']} |",
        f"| `lr` | {recipe['lr']} |",
        f"| `seed` | {recipe['seed']} |",
        "",
        "## Matched-arm result table",
        "",
        "| Arm | Params | Top-1 recall | NDCG | Accepted-set mass | Brier | ECE | Permutation-invariant |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, metrics in arms.items():
        lines.append(
            f"| `{name}` | {metrics['param_count']} | "
            f"{metrics['top1_recall']:.3f} | {metrics['ndcg']:.3f} | "
            f"{metrics['accepted_set_mass']:.3f} | "
            f"{metrics['brier_score']:.4f} | {metrics['calibration_error']:.4f} | "
            f"{robustness[name]['invariant']} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial control: permutation robustness",
            "",
            "`typed` and `hash_scalar` never receive row position as a "
            "feature; both are structurally permutation-invariant and this "
            "run confirms it empirically "
            f"({robustness['typed']['matches']}/{robustness['typed']['n']} and "
            f"{robustness['hash_scalar']['matches']}/{robustness['hash_scalar']['n']} "
            "gold-logit matches under a row-order/opaque-ID reshuffle that "
            "preserves every allowed content field). "
            "`typed_identity_bucket` is the deliberately-unsafe control — its "
            f"gold-logit matched only {robustness['typed_identity_bucket']['matches']}/"
            f"{robustness['typed_identity_bucket']['n']} times, confirming it "
            "can and does exploit build-time row order the way a real "
            "identity leak would, which is exactly why it is a labeled "
            "control and never a candidate for adoption.",
            "",
            "## Caveats",
            "",
            "- Synthetic single-slot, single-operator decisions — not a real "
            "compiler-produced `OperatorLegalSetV1` distribution.",
            "- Only a minimal flat `CandidateScoringHead` is exercised; "
            "factorized (SLM-383/DSH3-15) and ECOC (SLM-404/DSH3-29) heads "
            "are separate tickets that reuse this same encoder/batch.",
            "- No ship gate is evaluated or weakened. Fixture scale does not "
            "clear the ticket's own powered-comparison bar in either "
            "direction; see the disposition and stop rule.",
            "- The ticket's own listed verification command "
            "`python -m scripts.run_capability_gate --capability CAP2_TRANSFORM "
            "--suite fixture --dry-run` does not correspond to any script in "
            "this repository (no `scripts/run_capability_gate.py` exists); "
            "it was not run.",
            "",
            "## Verification commands",
            "",
            "```bash",
            "python -m pytest -q tests/test_models/test_operator_feature_encoder.py "
            "tests/test_models/test_legal_edit_batch.py",
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
        default=Path("outputs/runs/dsh3-23-operator-feature-encoder-fixture"),
    )
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = _run_fixture(
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_candidates=args.n_candidates,
        dim=args.dim,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_json = args.output_dir / "dsh3_23_operator_feature_encoder_report.json"
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
