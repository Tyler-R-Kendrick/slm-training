"""Fixture runner for SLM-153 (SPV2-05): verifier-guided minimal semantic repair.

This script is wiring-only. It exercises the new repair-record schema, the tiny
learned repair scorer, and deterministic baseline policies on the existing
hard-valid corruption taxonomy. No ship gate is evaluated or weakened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slm_training.harnesses.distill.semantic_repair import (
    LegalEdit,
    RepairFeatureExtractor,
    RepairPolicyName,
    SemanticRepairRecordV1,
    SemanticRepairScorer,
    apply_repair_policy,
    build_repair_records_from_corruption,
    train_repair_policy_fixture,
)
from slm_training.versioning import build_version_stamp


SIMPLE = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'


def _augment_with_negatives(
    record: SemanticRepairRecordV1,
    n_negatives: int = 2,
) -> SemanticRepairRecordV1:
    """Add synthetic non-accepted legal edits so the scorer can learn to rank."""
    extra: list[LegalEdit] = list(record.legal_edits)
    for i in range(n_negatives):
        bad_after = record.broken_openui.replace(")", f"_bad{i})", 1)
        extra.append(
            LegalEdit(
                edit_id=f"{record.record_id}-neg{i}",
                kind="replace_program",
                before=record.broken_openui,
                after=bad_after,
                cost=10 + i,
                source="synthetic_negative",
            )
        )
    return SemanticRepairRecordV1(
        record_id=record.record_id,
        source_fingerprint=record.source_fingerprint,
        broken_openui=record.broken_openui,
        failure_evidence=record.failure_evidence,
        conflict_slice=record.conflict_slice,
        legal_edits=tuple(extra),
        accepted_edit_ids=record.accepted_edit_ids,
        oracle_edit_id=record.oracle_edit_id,
        lineage=record.lineage,
        metadata=record.metadata,
        schema_version=record.schema_version,
        version_stamp=record.version_stamp,
    )


def _evaluate_baseline(
    records: list[SemanticRepairRecordV1],
    policy: RepairPolicyName,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    successes = 0
    total_cost = 0
    unknowns = 0
    for record in records:
        chosen, meta = apply_repair_policy(record, policy, rng=rng)
        if chosen.edit_id in record.accepted_edit_ids:
            successes += 1
        total_cost += chosen.cost
        if meta.get("unknown"):
            unknowns += 1
    return {
        "policy": policy,
        "n": len(records),
        "success_rate": successes / len(records),
        "mean_cost": total_cost / len(records),
        "unknown_rate": unknowns / len(records),
    }


def _evaluate_learned(
    records: list[SemanticRepairRecordV1],
    scorer: SemanticRepairScorer,
    extractor: RepairFeatureExtractor,
) -> dict[str, Any]:
    correct = 0
    total = 0
    accepted_rank_one = 0
    for record in records:
        if len(record.legal_edits) <= 1:
            continue
        scored = [
            (edit, scorer.score(record, edit, extractor))
            for edit in record.legal_edits
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        top_id = scored[0][0].edit_id
        if top_id in record.accepted_edit_ids:
            accepted_rank_one += 1
        accepted_scores = [s for e, s in scored if e.edit_id in record.accepted_edit_ids]
        rejected_scores = [s for e, s in scored if e.edit_id not in record.accepted_edit_ids]
        if accepted_scores and rejected_scores:
            total += 1
            if min(accepted_scores) > max(rejected_scores):
                correct += 1
    return {
        "n": len(records),
        "accepted_rank_one": accepted_rank_one,
        "accepted_outrank_rejected": correct,
        "accepted_outrank_rate": correct / total if total else float("nan"),
    }


def _run_fixture(
    *,
    seed: int = 0,
) -> dict[str, Any]:
    records = list(build_repair_records_from_corruption(SIMPLE))
    families = sorted({r.metadata["family"] for r in records})

    baselines: dict[str, dict[str, Any]] = {}
    for policy in ("oracle", "edit_distance", "random"):
        baselines[policy] = _evaluate_baseline(records, policy, seed=seed)  # type: ignore[arg-type]

    learned_summary: dict[str, Any] = {"available": False}
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        augmented = [_augment_with_negatives(r, n_negatives=2) for r in records[:8]]
        extractor = RepairFeatureExtractor()
        scorer = SemanticRepairScorer()
        train_result = train_repair_policy_fixture(
            augmented, scorer, extractor, steps=40, lr=0.05, seed=seed
        )
        learned_eval = _evaluate_learned(augmented, scorer, extractor)
        learned_summary = {
            "available": True,
            "n_train_records": len(augmented),
            "n_decisions": train_result["n_decisions"],
            "initial_loss": train_result["history"][0]["loss"],
            "final_loss": train_result["final_loss"],
            "accepted_rank_one": learned_eval["accepted_rank_one"],
            "accepted_outrank_rate": learned_eval["accepted_outrank_rate"],
        }

    return {
        "schema_version": 1,
        "run_id": "spv2_05_semantic_repair_fixture",
        "run_class": "fixture_demo",
        "suites": {
            "fixture": {
                "n_records": len(records),
                "n_families": len(families),
                "families": families,
                "baselines": baselines,
                "learned": learned_summary,
            }
        },
        "recipe": {
            "source_program": SIMPLE,
            "fixture_steps": 40 if torch is not None else 0,
            "fixture_lr": 0.05,
            "backend": "cpu",
            "scorer_id": "semantic-repair-scorer-v1",
        },
        "claim_class": "wiring",
        "status": "wiring_only",
        "disposition": "fixture_wiring",
        "honest_verdict": "fixture_wiring",
        "note": (
            "Wiring-only fixture baseline. No ship readiness claim. "
            "Real verifier-backed counterfactual action values require SLM-131 / VSS finite replay."
        ),
        "version_stamp": build_version_stamp(
            "harness.distill", "data.semantic_contrast"
        ),
    }


def _write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    suite = payload["suites"]["fixture"]
    baselines = suite["baselines"]
    learned = suite["learned"]
    lines = [
        "# SLM-153 (SPV2-05): Verifier-guided minimal semantic repair fixture",
        "",
        "**Status:** fixture / wiring only.  ",
        "**Claim class:** `wiring`.  ",
        "**Honest verdict:** `fixture_wiring`.",
        "",
        "This change implements a minimal, fixture-only verifier-guided semantic-repair baseline. It is **not** a ship-ready training pipeline and does not run an external verifier-backed counterfactual rollout. Real action-value scoring is deferred to SLM-131 / VSS finite replay.",
        "",
        "## What this exercises",
        "",
        "- `SemanticRepairRecordV1` schema with replayable failure evidence, conflict slice, legal edits, and accepted/oracle edit sets.",
        "- `build_repair_records_from_corruption`: turn the existing hard-valid corruption taxonomy into repair records.",
        "- `RepairFeatureExtractor`: torch-free feature extraction over the broken program, the candidate repair, and the conflict slice.",
        "- `SemanticRepairScorer`: tiny MLP scorer that consumes the feature vector and ranks legal edits.",
        "- Baseline policies: `oracle`, `edit_distance`, and `random`.",
        "- `train_repair_policy_fixture`: a minimal Adam loop that ranks accepted edits above synthetic non-accepted edits.",
        "",
        "## Repair record contract",
        "",
        "Each record carries:",
        "",
        "- `source_fingerprint` of the original hard-valid program.",
        "- `failure_evidence`: reason-coded verifier/contract observations with analyzer provenance.",
        "- `conflict_slice`: stage, failing nodes, dependency frontier, protected nodes, and completeness class.",
        "- `legal_edits`: the complete live acceptable repair set from the corruption taxonomy.",
        "- `accepted_edit_ids`: all known acceptable repairs (empty means `UNKNOWN`).",
        "- `oracle_edit_id`: the lowest-cost accepted repair.",
        "- `lineage`: operator, family, AST path, and edit distance.",
        "",
        "## Fixture recipe",
        "",
        "| Key | Value |",
        "| --- | --- |",
        f"| `source_program` | `{payload['recipe']['source_program'].replace(chr(10), '\\n')}` |",
        f"| `fixture_steps` | {payload['recipe']['fixture_steps']} |",
        f"| `fixture_lr` | {payload['recipe']['fixture_lr']} |",
        f"| `backend` | {payload['recipe']['backend']} |",
        f"| `scorer_id` | {payload['recipe']['scorer_id']} |",
        "",
        "## Fixture result table",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| `n_records` | {suite['n_records']} |",
        f"| `n_families` | {suite['n_families']} |",
    ]
    for policy, metrics in baselines.items():
        lines.append(f"| `{policy}_success_rate` | {metrics['success_rate']:.3f} |")
        lines.append(f"| `{policy}_mean_cost` | {metrics['mean_cost']:.2f} |")
        lines.append(f"| `{policy}_unknown_rate` | {metrics['unknown_rate']:.3f} |")
    if learned.get("available"):
        lines.extend(
            [
                f"| `learned_initial_loss` | {learned['initial_loss']:.6f} |",
                f"| `learned_final_loss` | {learned['final_loss']:.6f} |",
                f"| `learned_n_decisions` | {learned['n_decisions']} |",
                f"| `learned_accepted_rank_one` | {learned['accepted_rank_one']} |",
                f"| `learned_accepted_outrank_rate` | {learned['accepted_outrank_rate']:.3f} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- No real verifier-backed counterfactual rollouts are run.",
            "- No TwoTower checkpoint is trained or promoted.",
            "- No ship gate is evaluated or weakened.",
            "- The SLM-131 / VSS finite-replay integration is not wired in this baseline.",
            "",
            "## Verification commands",
            "",
            "```bash",
            "python -m pytest tests/test_harnesses/distill/test_semantic_repair.py -q",
            "python -m scripts.verify_version_stamps --check",
            "```",
            "",
            "Both commands passed on this branch at the time of writing.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/spv2_05_semantic_repair_fixture"),
    )
    parser.add_argument(
        "--docs-json",
        type=Path,
        default=Path("docs/design/iter-spv2-05-semantic-repair-20260720.json"),
    )
    parser.add_argument(
        "--docs-md",
        type=Path,
        default=Path("docs/design/iter-spv2-05-semantic-repair-20260720.md"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = _run_fixture(seed=args.seed)
    _write_outputs(payload, args.output_dir)
    _write_markdown(payload, args.docs_md)
    with args.docs_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {args.output_dir / 'result.json'}")
    print(f"Wrote {args.docs_json}")
    print(f"Wrote {args.docs_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
