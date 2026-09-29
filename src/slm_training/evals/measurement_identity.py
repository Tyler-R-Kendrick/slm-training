"""Content and declared-unit identities shared by decode and loss evidence.

Missing root provenance stays unknown; a record ID is not invented independence.
"""

from __future__ import annotations

import hashlib
import json
import math


def content_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def selected_identity(records) -> dict:
    ids = [record.id for record in records]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("selection must contain nonempty unique case IDs")
    return {
        "selected_record_ids": ids,
        "selected_root_ids": [declared_root(record) for record in records],
        "input_sha256s": [content_digest(record.to_dict()) for record in records],
        "selection_sha256": content_digest([record.to_dict() for record in records]),
    }


def declared_root(record):
    meta = record.meta or {}
    value = meta.get("root_family_id") or meta.get("root_id")
    return str(value) if value is not None else None


def row_identity(record, *, selection_sha256: str, seed: int,
                 estimator_id: str, evaluator_sha256: str) -> dict:
    return {
        "case_id": record.id,
        "root_id": declared_root(record),
        "input_sha256": content_digest(record.to_dict()),
        "seed": seed,
        "estimator_id": estimator_id,
        "evaluator_sha256": evaluator_sha256,
        "selection_sha256": selection_sha256,
    }


def grouped_nll(rows, key: str) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: {
        "n_records": len(group),
        "masked_tokens": sum(row["masked_tokens"] for row in group),
        "mean_nll": sum(row["nll_sum"] for row in group)
        / max(1, sum(row["masked_tokens"] for row in group)),
    } for name, group in sorted(groups.items())}


NLL_IDENTITY_FIELDS = ("case_id", "root_id", "input_sha256", "seed",
    "estimator_id", "evaluator_sha256", "selection_sha256", "units", "nll_sum")


def loss_record_rows(categories: dict) -> list[dict]:
    """Flatten categories without dropping invalid/missing broad observations."""
    keys = {"broad": "nll", "binding": "binding_nll",
            "structural": "structural_nll", "schema_ood": "schema_ood_nll"}
    rows = {}
    for category, key in keys.items():
        report = categories.get(category)
        if report is None:
            continue
        if not isinstance(report, dict):
            raise ValueError("invalid loss category report")
        seen = set()
        for entry in report.get("per_record") or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
                raise ValueError("nonempty string loss record identity required")
            record_id = entry["id"]
            if record_id in seen:
                raise ValueError(f"duplicate loss record identity in {category}: {record_id}")
            seen.add(record_id)
            row = rows.setdefault(record_id, {"id": record_id, **dict.fromkeys(keys.values())})
            _validate_loss_numbers(entry)
            mean = entry.get("mean_nll")
            row[key] = mean
            if category == "broad":
                row.update({field: entry.get(field) for field in NLL_IDENTITY_FIELDS})
                row["masked_tokens"] = entry.get("masked_tokens", 0)
    return [rows[key] for key in sorted(rows)]


def _validate_loss_numbers(entry: dict) -> None:
    for key in ("mean_nll", "nll_sum"):
        value = entry.get(key)
        if value is not None and (type(value) not in (int, float)
                                  or not math.isfinite(value) or value < 0):
            raise ValueError("nonfinite or negative loss evidence")
    if "masked_tokens" in entry:
        count = entry["masked_tokens"]
        if type(count) is not int or count < 0:
            raise ValueError("loss masked_tokens must be a nonnegative integer")
        if (entry.get("mean_nll") is not None) != (count > 0):
            raise ValueError("loss mean must match measured token denominator")
