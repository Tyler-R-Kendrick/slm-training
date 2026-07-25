"""Deterministic full-corpus certification built on the shared G0--G12 stack."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from slm_training.data.leakage import fingerprint_pair
from slm_training.data.quality import assess_record
from slm_training.data.store import write_common_manifest
from slm_training.data.verify.stack import VerificationReport, verify_record
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.versioning import build_version_stamp


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CertifiedRow:
    snapshot_id: str
    record: ExampleRecord
    record_sha256: str
    canonical_snapshot_id: str
    canonical_record_id: str
    report: VerificationReport
    quality_score: float
    quality_reasons: tuple[str, ...]

    @property
    def duplicate(self) -> bool:
        return (self.snapshot_id, self.record.id) != (
            self.canonical_snapshot_id,
            self.canonical_record_id,
        )

    def membership(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "source_record_id": self.record.id,
            "source_record_sha256": self.record_sha256,
            "canonical_snapshot_id": self.canonical_snapshot_id,
            "canonical_record_id": self.canonical_record_id,
            "duplicate": self.duplicate,
            "tier": self.report.tier.value,
            "failing_gate": self.report.failing_gate.value if self.report.failing_gate else None,
            "gates": self.report.to_dict()["gates"],
            "quality_score": self.quality_score,
            "quality_reasons": list(self.quality_reasons),
        }


def certify_snapshots(
    snapshots: Mapping[str, Iterable[ExampleRecord]],
) -> tuple[CertifiedRow, ...]:
    """Certify every input row after deterministic exact deduplication.

    Every source row is retained in the returned membership ledger; duplicate
    rows inherit their canonical row's exact G0--G12 report rather than being
    silently discarded.  Near/semantic dedup remains the separate audited
    policy in ``audit_data_corpora`` because it cannot choose a canonical row
    without a declared similarity engine/version.
    """
    source_rows = [
        (snapshot_id, record)
        for snapshot_id, records in sorted(snapshots.items())
        for record in records
    ]
    canonical: dict[str, tuple[str, ExampleRecord]] = {}
    for snapshot_id, record in source_rows:
        key = fingerprint_pair(record.prompt, record.openui)
        canonical.setdefault(key, (snapshot_id, record))

    reports: dict[str, tuple[VerificationReport, float, tuple[str, ...]]] = {}
    rows: list[CertifiedRow] = []
    for snapshot_id, record in source_rows:
        key = fingerprint_pair(record.prompt, record.openui)
        canonical_snapshot_id, canonical_record = canonical[key]
        if key not in reports:
            quality = assess_record(canonical_record, require_design_md=False)
            reports[key] = (
                verify_record(canonical_record),
                quality.score,
                tuple(quality.reasons),
            )
        report, score, reasons = reports[key]
        rows.append(
            CertifiedRow(
                snapshot_id=snapshot_id,
                record=record,
                record_sha256=_json_sha256(record.to_dict()),
                canonical_snapshot_id=canonical_snapshot_id,
                canonical_record_id=canonical_record.id,
                report=report,
                quality_score=score,
                quality_reasons=reasons,
            )
        )
    return tuple(rows)


def core_records(rows: Iterable[CertifiedRow]) -> list[ExampleRecord]:
    """Return canonical Gold/Silver rows only; Bronze/Quarantine never enter SFT."""
    selected: list[ExampleRecord] = []
    for row in rows:
        if row.duplicate or row.report.tier.value not in {"Gold", "Silver"}:
            continue
        verification = row.report.to_dict()
        selected.append(
            replace(
                row.record,
                meta={
                    **dict(row.record.meta),
                    "verification_tier": row.report.tier.value,
                    "failing_gate": verification["failing_gate"],
                    "verification": verification,
                    "certification": {
                        "snapshot_id": row.snapshot_id,
                        "source_record_sha256": row.record_sha256,
                    },
                },
            )
        )
    return selected


def write_certified_snapshot(
    directory: Path,
    *,
    dataset_id: str,
    rows: Iterable[CertifiedRow],
    source_fingerprints: Mapping[str, str],
) -> dict[str, object]:
    """Write compact immutable certification manifests and the SFT core."""
    rows = tuple(rows)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"certified dataset is immutable: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    by_bucket: dict[str, list[dict[str, object]]] = {
        "all": [], "gold_silver": [], "bronze": [], "quarantine": []
    }
    failures: Counter[tuple[str, str]] = Counter()
    duplicates: list[dict[str, object]] = []
    for row in rows:
        member = row.membership()
        by_bucket["all"].append(member)
        tier = row.report.tier.value
        if tier in {"Gold", "Silver"}:
            by_bucket["gold_silver"].append(member)
        else:
            by_bucket[tier.lower()].append(member)
        if row.duplicate:
            duplicates.append(member)
        if row.report.failing_gate:
            failures[(
                row.report.failing_gate.value,
                "; ".join(row.quality_reasons) or "verification",
            )] += 1

    for name, payload in by_bucket.items():
        _write_jsonl(directory / f"{name}.jsonl", payload)
    _write_jsonl(directory / "duplicates.jsonl", duplicates)
    write_jsonl(directory / "records.jsonl", core_records(rows))
    quarantine = by_bucket["quarantine"]
    _write_jsonl(directory / "rejected.jsonl", quarantine)
    taxonomy = [
        {"failing_gate": gate, "reason": reason, "count": count}
        for (gate, reason), count in sorted(failures.items())
    ]
    (directory / "failure_taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2) + "\n", encoding="utf-8"
    )
    tier_counts = Counter(row.report.tier.value for row in rows)
    quarantine_rate = tier_counts["Quarantine"] / len(rows) if rows else 0.0
    (directory / "quality_report.json").write_text(
        json.dumps(
            {
                "schema": "corpus_certification_quality/v1",
                "source_rows": len(rows),
                "core_records": len(core_records(rows)),
                "tiers": dict(sorted(tier_counts.items())),
                "quarantine_rate": quarantine_rate,
                "warnings": (
                    ["quarantine rate exceeds 10%; run a matched clean-subset control"]
                    if quarantine_rate > 0.10
                    else []
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "synthesis_feedback.json").write_text(
        json.dumps(
            {
                "schema": "synthesis_feedback/v1",
                "recommendations": [],
                "experiment_candidates": (
                    [
                        {
                            "hypothesis": "training only on certified core records changes outcomes when quarantine exceeds 10%",
                            "falsification_criteria": "matched clean-subset control is not required below the preregistered 10% threshold",
                        }
                    ]
                    if quarantine_rate > 0.10
                    else []
                ),
                "note": "certification did not synthesize records; this sidecar records the admission outcome",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "schema": "corpus_certification/v1",
        "dataset_id": dataset_id,
        "source_fingerprints": dict(sorted(source_fingerprints.items())),
        "source_rows": len(rows),
        "core_records": len(core_records(rows)),
        "tiers": dict(sorted(tier_counts.items())),
        "human_audit_gate": "recorded only; never required for admission",
        "version_stamp": build_version_stamp("data.corpus_certification"),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return write_common_manifest(
        directory, kind="train", dataset_id=dataset_id, immutable=True
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = ["CertifiedRow", "certify_snapshots", "core_records", "write_certified_snapshot"]
