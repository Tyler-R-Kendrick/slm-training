"""Consume immutable trainer exposure; never infer examples from update counts."""

from __future__ import annotations

import math
from pathlib import Path

from .checkpoint_bundle import validate_bundle


def exposure_epochs(rows: list[dict]) -> float:
    """Each historical snapshot retains its own denominator, including replay."""
    if not isinstance(rows, list):
        raise ValueError("champion_exposure:invalid_rows")
    total = 0.0
    for row in rows:
        if not isinstance(row, dict) or not row.get("snapshot_sha256"):
            raise ValueError("champion_exposure:missing_snapshot")
        count, examples = row.get("record_count"), row.get("consumed_examples")
        if (
            type(count) is not int
            or count <= 0
            or type(examples) is not int
            or examples < 0
        ):
            raise ValueError("champion_exposure:invalid_counts")
        epochs = examples / count
        reported = row.get("example_exposure_epochs")
        if (
            type(reported) not in (int, float)
            or not math.isfinite(reported)
            or reported != epochs
        ):
            raise ValueError("champion_exposure:inconsistent_epochs")
        tokens = row.get("tokens")
        if not isinstance(tokens, dict) or set(tokens) != {"prompt", "target"}:
            raise ValueError("champion_exposure:missing_token_counts")
        if any(type(value) is not int or value < 0 for value in tokens.values()):
            raise ValueError("champion_exposure:invalid_token_counts")
        total += epochs
    if not math.isfinite(total):
        raise ValueError("champion_exposure:nonfinite_total")
    return total


def checkpoint_exposure(checkpoint: Path) -> list[dict] | None:
    """Return complete ancestor + local exposure, or explicitly unknown legacy.

    Exact resumes carry cumulative local counters, so adding the previous
    champion would double-charge them. Warm starts already carry ancestor rows.
    Bundle integrity is not worker authenticity: this runs in the controller.
    """
    directory = checkpoint.parent
    if not (directory / "manifest.json").is_file():
        return None
    _, manifest = validate_bundle(directory.parent.parent, directory.name)
    meta = manifest["metadata"]
    if meta.get("role") == "trial_cursor":
        ancestor = meta.get("ancestor_exposure")
        current = meta.get("exposure")
        if ancestor is None or current is None:
            return None
        rows = [*ancestor, *current]
    else:
        rows = meta.get("exposure_history")
    if rows is not None:
        exposure_epochs(rows)
    return rows


def bind_exposure(checkpoint: Path, payload: dict) -> dict:
    """Bind a new sidecar to its actual bundle without strengthening old history."""
    rows = checkpoint_exposure(checkpoint)
    origin = None
    if (checkpoint.parent / "manifest.json").is_file():
        _, manifest = validate_bundle(
            checkpoint.parent.parent.parent, checkpoint.parent.name
        )
        meta = manifest["metadata"]
        origin = (
            meta.get("training_bundle_digest")
            if meta.get("schema") == "climb_champion/v3"
            else checkpoint.parent.name
        )
    payload = dict(
        payload,
        schema="climb_champion/v3",
        exposure_history=rows,
        training_bundle_digest=origin,
    )
    if rows is not None:
        payload["cumulative_epochs"] = exposure_epochs(rows)
    return payload


def cumulative_epochs(payload: dict) -> float:
    rows = payload.get("exposure_history")
    if rows is not None:
        return exposure_epochs(rows)
    # Historical estimates remain historical. A new corpus cannot erase them.
    value = payload.get("cumulative_epochs")
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("champion_exposure:invalid_legacy_estimate")
    return float(value)
