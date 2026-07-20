"""Promotion-protocol evaluation (P1c)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from slm_training.harness_core.promotion_engine import (
    PromotionCriteria,
    check_rank_stability,
)
from slm_training.harness_core.promotion_engine import (
    check_category_regression as _check_category_regression,
)
from slm_training.harness_core.promotion_engine import (
    evaluate_promotion as _evaluate_promotion,
)
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)

__all__ = [
    "HARD_CATEGORIES",
    "PromotionCriteria",
    "check_category_regression",
    "check_data_integrity",
    "check_rank_stability",
    "evaluate_promotion",
    "register_promoted_checkpoint",
]

HARD_CATEGORIES = ("binding", "structural", "repair")


def _openui_gate_evaluator(
    suites: dict[str, dict[str, Any]],
    policy: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    return evaluate_ship_gates(suites, thresholds=policy or DEFAULT_SHIP_GATES)


def check_data_integrity(
    train_dir: Path | str,
    test_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Lightweight integrity: train manifest exists + optional leakage scan."""
    from slm_training.data.leakage import find_leakage, load_train_fingerprints
    from slm_training.dsl.schema import load_jsonl

    train_dir = Path(train_dir)
    manifest = train_dir / "manifest.json"
    records_path = train_dir / "records.jsonl"
    failures: list[str] = []
    if not records_path.exists():
        failures.append("missing_train_records")
    if not manifest.exists():
        failures.append("missing_train_manifest")
    leakage_hits = 0
    if test_dir is not None and manifest.exists():
        fps = load_train_fingerprints(manifest)
        suites_root = Path(test_dir) / "suites"
        if suites_root.exists():
            for suite_dir in sorted(suites_root.iterdir()):
                rec_path = suite_dir / "records.jsonl"
                if not rec_path.exists():
                    continue
                for record in load_jsonl(rec_path):
                    hits = find_leakage(record, fps)
                    leakage_hits += len(hits)
        if leakage_hits:
            failures.append(f"leakage_hits:{leakage_hits}")
    return {
        "pass": not failures,
        "failures": failures,
        "leakage_hits": leakage_hits,
    }


def check_category_regression(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """No hard category may regress more than ``tolerance`` relatively."""
    return _check_category_regression(
        baseline,
        candidate,
        categories=HARD_CATEGORIES,
        tolerance=tolerance,
    )


def evaluate_promotion(
    *,
    integrity: dict[str, Any] | None = None,
    baseline_loss_report: dict[str, Any] | None = None,
    candidate_loss_report: dict[str, Any] | None = None,
    rankings: dict[str, list[str]] | None = None,
    eg_time_by_seed: Sequence[float] | None = None,
    ship_suites: dict[str, dict[str, Any]] | None = None,
    criteria: PromotionCriteria | None = None,
) -> dict[str, Any]:
    """Return ``{promotable, checks, failures}`` mirroring ship-gates shape."""
    return _evaluate_promotion(
        integrity=integrity,
        baseline_loss_report=baseline_loss_report,
        candidate_loss_report=candidate_loss_report,
        rankings=rankings,
        eg_time_by_seed=eg_time_by_seed,
        ship_suites=ship_suites,
        criteria=criteria,
        hard_categories=HARD_CATEGORIES,
        gate_evaluator=_openui_gate_evaluator,
    )


def register_promoted_checkpoint(
    checkpoint_dir: Path | str,
    *,
    source: Path | str | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Copy/link the mid-trained anchor to ``promoted.pt`` (P1d)."""
    import shutil

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir / "promoted.pt"
    if source is not None:
        source = Path(source)
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
    meta_path = checkpoint_dir / "promoted.json"
    payload = {"kind": "promoted_anchor", **(meta or {})}
    meta_path.write_text(
        __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return dest
