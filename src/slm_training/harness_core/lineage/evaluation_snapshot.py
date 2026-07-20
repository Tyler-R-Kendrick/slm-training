"""Fail-closed construction of the frozen production evaluation snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from slm_training.harness_core.lineage.data_cycle import snapshot_directory
from slm_training.harness_core.lineage.records import DataSnapshot

REQUIRED_SUITES = ("smoke", "held_out", "adversarial", "ood", "rico_held")


def build_evaluation_snapshot(
    snapshot_id: str,
    suites: Mapping[str, Path | str],
    human_feedback_holdout: Path | str,
    *,
    training_ids: set[str] | None = None,
) -> DataSnapshot:
    missing = [name for name in REQUIRED_SUITES if name not in suites]
    if missing:
        raise ValueError(f"evaluation snapshot missing suites: {', '.join(missing)}")
    counts = {name: _ids(Path(suites[name])) for name in REQUIRED_SUITES}
    if len(counts["rico_held"]) < 1500:
        raise ValueError("production evaluation snapshot requires rico_held n>=1500")
    feedback_ids = _ids(Path(human_feedback_holdout))
    if not feedback_ids:
        raise ValueError("human-feedback holdout is empty")
    overlap = feedback_ids & set(training_ids or ())
    if overlap:
        raise ValueError(
            f"human-feedback holdout overlaps training ids: {sorted(overlap)[:3]}"
        )
    sources = [Path(suites[name]) for name in REQUIRED_SUITES] + [
        Path(human_feedback_holdout)
    ]
    return snapshot_directory(
        snapshot_id,
        sources,
        metadata={
            "kind": "frozen_production_evaluation",
            "suite_sizes": {name: len(ids) for name, ids in counts.items()},
            "human_feedback_holdout_n": len(feedback_ids),
            "screening_only_remediated_n": 19,
        },
    )


def _ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: set[str] = set()
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            result.add(str(row["id"]))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid evaluation row") from exc
    return result
