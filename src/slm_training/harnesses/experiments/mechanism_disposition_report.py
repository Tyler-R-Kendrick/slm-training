"""SGS-009 (SLM-456): mechanism disposition and stale-evidence supersession
reports.

Generalizes the SLM-160/SPV4-02 causal-disposition precedent
(``slm160_spv_disposition.py``'s ``SPVMechanismDisposition``/``Disposition``,
narrated in ``docs/design/semantic-planning-valid-state-disposition.md``)
into reusable code covering any mechanism in this initiative, not just the
SPV family. Reuses that module's ``Disposition`` enum directly rather than
redefining the same seven values a second time.

This is the canonical ``evidence_class`` owner PCT-007
(``models/decode_stats.py``'s ``MECHANISM_EVIDENCE_CLASSES``) named as
still-Backlog when it stayed a caller-declared advisory field instead of a
closed registry -- ``EVIDENCE_CLASSES`` here is that registry.

Scope note: this module ships the reusable schema, fail-closed validation,
append-only supersession ledger, and report generator, proven against a
fixture matrix (the Validation section's explicit ask). It does not itself
audit and classify every real mechanism across the current SGS/VCE/PCT/SRP/
SIE/RSP backlog -- that data-entry pass is necessarily much larger than one
bounded PR and is follow-on work, exactly as SLM-160's own audit was a
separate pass from building its harness.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from slm_training.harnesses.experiments.slm160_spv_disposition import Disposition

RECORD_SCHEMA = "mechanism_disposition_record/v1"
SUPERSESSION_SCHEMA = "mechanism_supersession_entry/v1"
REPORT_SCHEMA = "mechanism_disposition_report/v1"

# Mirrors the authority-tier vocabulary docs/design/repository-ownership-map.md
# already uses for every subsystem entry -- not a second tier taxonomy.
AUTHORITY_CLASSES = frozenset(
    {"compiler-hard", "verifier-hard", "advisory-learned", "oracle-diagnostic", "evaluation-only"}
)

# The canonical evidence-class registry PCT-007 deferred to this issue.
EVIDENCE_CLASSES = frozenset(
    {"fixture", "scratch", "measured", "promotable", "rejected", "blocked", "ship"}
)
_MEASURED_EVIDENCE_CLASSES = frozenset({"measured", "promotable", "ship"})
_ADOPTED_DISPOSITIONS = frozenset({Disposition.ADOPT_PRIMARY, Disposition.ADOPT_OPTIONAL})

DEFAULT_STATES = frozenset({"on", "off", "n/a", "blocked"})


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class MechanismDispositionRecordV1:
    """One mechanism's current, evidence-backed disposition.

    Built only by :func:`build_mechanism_disposition_record`, which fails
    closed on the two invariants this issue names explicitly: a mechanism
    without measured evidence can never be recorded as adopted, and a
    fixture-only mechanism can never set ``default_state="on"``.
    """

    schema_version: str
    mechanism_id: str
    owner_module: str
    owner_version: str
    default_state: str
    authority_class: str
    evidence_cutoff: str
    evidence_class: str
    activation_evidence: str
    applicable_packs: tuple[str, ...]
    matched_controls: bool
    semantic_effect: str
    cost_cold_effect: str
    safety_invariants: tuple[str, ...]
    external_dependencies: tuple[str, ...]
    disposition: Disposition
    rationale: str
    issues: tuple[str, ...]
    record_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mechanism_id": self.mechanism_id,
            "owner_module": self.owner_module,
            "owner_version": self.owner_version,
            "default_state": self.default_state,
            "authority_class": self.authority_class,
            "evidence_cutoff": self.evidence_cutoff,
            "evidence_class": self.evidence_class,
            "activation_evidence": self.activation_evidence,
            "applicable_packs": list(self.applicable_packs),
            "matched_controls": self.matched_controls,
            "semantic_effect": self.semantic_effect,
            "cost_cold_effect": self.cost_cold_effect,
            "safety_invariants": list(self.safety_invariants),
            "external_dependencies": list(self.external_dependencies),
            "disposition": self.disposition.value,
            "rationale": self.rationale,
            "issues": list(self.issues),
            "record_hash": self.record_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MechanismDispositionRecordV1":
        if data.get("schema_version") != RECORD_SCHEMA:
            raise ValueError(
                f"unsupported mechanism disposition record schema: {data.get('schema_version')!r}"
            )
        record = cls(
            schema_version=data.get("schema_version", RECORD_SCHEMA),
            mechanism_id=str(data["mechanism_id"]),
            owner_module=str(data.get("owner_module", "")),
            owner_version=str(data.get("owner_version", "")),
            default_state=str(data.get("default_state", "off")),
            authority_class=str(data.get("authority_class", "evaluation-only")),
            evidence_cutoff=str(data.get("evidence_cutoff", "")),
            evidence_class=str(data.get("evidence_class", "fixture")),
            activation_evidence=str(data.get("activation_evidence", "")),
            applicable_packs=tuple(data.get("applicable_packs", [])),
            matched_controls=bool(data.get("matched_controls", False)),
            semantic_effect=str(data.get("semantic_effect", "")),
            cost_cold_effect=str(data.get("cost_cold_effect", "")),
            safety_invariants=tuple(data.get("safety_invariants", [])),
            external_dependencies=tuple(data.get("external_dependencies", [])),
            disposition=Disposition(data.get("disposition", "inconclusive")),
            rationale=str(data.get("rationale", "")),
            issues=tuple(data.get("issues", [])),
            record_hash=str(data.get("record_hash", "")),
        )
        payload = record.to_dict()
        payload.pop("record_hash", None)
        if _digest(payload) != record.record_hash:
            raise ValueError(
                "mechanism disposition record hash mismatch (tampered or corrupted payload)"
            )
        return record


def build_mechanism_disposition_record(
    *,
    mechanism_id: str,
    owner_module: str,
    owner_version: str,
    default_state: str,
    authority_class: str,
    evidence_cutoff: str,
    evidence_class: str,
    disposition: Disposition,
    rationale: str,
    activation_evidence: str = "",
    applicable_packs: tuple[str, ...] = (),
    matched_controls: bool = False,
    semantic_effect: str = "",
    cost_cold_effect: str = "",
    safety_invariants: tuple[str, ...] = (),
    external_dependencies: tuple[str, ...] = (),
    issues: tuple[str, ...] = (),
) -> MechanismDispositionRecordV1:
    """Build one record; fails closed rather than recording an honesty violation.

    * a mechanism cannot be ``adopt_primary``/``adopt_optional`` unless its
      ``evidence_class`` is ``measured``/``promotable``/``ship`` *and* it
      carries non-empty ``activation_evidence`` -- a wired-but-unmeasured
      mechanism can never be represented as adopted;
    * ``evidence_class="fixture"`` can never pair with ``default_state="on"``
      -- fixture-only results can never promote a default/checkpoint.
    """
    if not mechanism_id:
        raise ValueError("mechanism_id is required")
    if authority_class not in AUTHORITY_CLASSES:
        raise ValueError(f"unknown authority_class: {authority_class!r}")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"unknown evidence_class: {evidence_class!r}")
    if default_state not in DEFAULT_STATES:
        raise ValueError(f"unknown default_state: {default_state!r}")
    if not isinstance(disposition, Disposition):
        raise ValueError(f"disposition must be a Disposition member, got {disposition!r}")

    if disposition in _ADOPTED_DISPOSITIONS:
        if evidence_class not in _MEASURED_EVIDENCE_CLASSES:
            raise ValueError(
                f"{disposition.value} requires evidence_class in "
                f"{sorted(_MEASURED_EVIDENCE_CLASSES)}, got {evidence_class!r} "
                "-- a wired-but-unmeasured mechanism cannot be adopted"
            )
        if not activation_evidence:
            raise ValueError(
                f"{disposition.value} requires non-empty activation_evidence"
            )
    if evidence_class == "fixture" and default_state == "on":
        raise ValueError(
            "evidence_class='fixture' cannot set default_state='on' -- "
            "fixture-only results cannot promote a default/checkpoint"
        )

    record = MechanismDispositionRecordV1(
        schema_version=RECORD_SCHEMA,
        mechanism_id=mechanism_id,
        owner_module=owner_module,
        owner_version=owner_version,
        default_state=default_state,
        authority_class=authority_class,
        evidence_cutoff=evidence_cutoff,
        evidence_class=evidence_class,
        activation_evidence=activation_evidence,
        applicable_packs=tuple(applicable_packs),
        matched_controls=matched_controls,
        semantic_effect=semantic_effect,
        cost_cold_effect=cost_cold_effect,
        safety_invariants=tuple(safety_invariants),
        external_dependencies=tuple(external_dependencies),
        disposition=disposition,
        rationale=rationale,
        issues=tuple(issues),
        record_hash="",
    )
    payload = record.to_dict()
    payload.pop("record_hash", None)
    return replace(record, record_hash=_digest(payload))


def disposition_record_integrity_ok(record: MechanismDispositionRecordV1) -> bool:
    payload = record.to_dict()
    payload.pop("record_hash", None)
    return _digest(payload) == record.record_hash


@dataclass(frozen=True)
class SupersessionEntryV1:
    """One append-only claim: an earlier record's evidence is now stale."""

    schema_version: str
    superseded_mechanism_id: str
    superseded_evidence_cutoff: str
    superseding_mechanism_id: str
    superseding_evidence_cutoff: str
    reason: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "superseded_mechanism_id": self.superseded_mechanism_id,
            "superseded_evidence_cutoff": self.superseded_evidence_cutoff,
            "superseding_mechanism_id": self.superseding_mechanism_id,
            "superseding_evidence_cutoff": self.superseding_evidence_cutoff,
            "reason": self.reason,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupersessionEntryV1":
        if data.get("schema_version") != SUPERSESSION_SCHEMA:
            raise ValueError(
                f"unsupported supersession entry schema: {data.get('schema_version')!r}"
            )
        entry = cls(
            schema_version=data.get("schema_version", SUPERSESSION_SCHEMA),
            superseded_mechanism_id=str(data["superseded_mechanism_id"]),
            superseded_evidence_cutoff=str(data.get("superseded_evidence_cutoff", "")),
            superseding_mechanism_id=str(data["superseding_mechanism_id"]),
            superseding_evidence_cutoff=str(data.get("superseding_evidence_cutoff", "")),
            reason=str(data.get("reason", "")),
            entry_hash=str(data.get("entry_hash", "")),
        )
        payload = entry.to_dict()
        payload.pop("entry_hash", None)
        if _digest(payload) != entry.entry_hash:
            raise ValueError(
                "supersession entry hash mismatch (tampered or corrupted payload)"
            )
        return entry


def build_supersession_entry(
    *,
    superseded: MechanismDispositionRecordV1,
    superseding: MechanismDispositionRecordV1,
    reason: str,
) -> SupersessionEntryV1:
    """One record's claim is stale; ``superseding`` is the current truth.

    Never rewrites ``superseded`` -- it stays exactly as recorded, visible
    in ``records``, with this entry as the machine-readable pointer saying
    a newer record now supersedes it.
    """
    if not reason:
        raise ValueError("reason is required")
    if superseded.mechanism_id != superseding.mechanism_id:
        raise ValueError(
            "a supersession entry links two records of the same mechanism_id"
        )
    entry = SupersessionEntryV1(
        schema_version=SUPERSESSION_SCHEMA,
        superseded_mechanism_id=superseded.mechanism_id,
        superseded_evidence_cutoff=superseded.evidence_cutoff,
        superseding_mechanism_id=superseding.mechanism_id,
        superseding_evidence_cutoff=superseding.evidence_cutoff,
        reason=reason,
        entry_hash="",
    )
    payload = entry.to_dict()
    payload.pop("entry_hash", None)
    return replace(entry, entry_hash=_digest(payload))


def append_supersession(
    ledger: tuple[SupersessionEntryV1, ...], entry: SupersessionEntryV1
) -> tuple[SupersessionEntryV1, ...]:
    """Pure append -- returns a new tuple, never mutates ``ledger``."""
    return ledger + (entry,)


@dataclass(frozen=True)
class MechanismDispositionReportV1:
    """The full, structured closeout: every record plus the supersession
    ledger, both exhaustive -- rejected/blocked/inconclusive records are
    never filtered out."""

    schema_version: str
    records: tuple[MechanismDispositionRecordV1, ...]
    supersessions: tuple[SupersessionEntryV1, ...]
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "records": [r.to_dict() for r in self.records],
            "supersessions": [s.to_dict() for s in self.supersessions],
            "report_hash": self.report_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MechanismDispositionReportV1":
        if data.get("schema_version") != REPORT_SCHEMA:
            raise ValueError(
                f"unsupported mechanism disposition report schema: {data.get('schema_version')!r}"
            )
        report = cls(
            schema_version=data.get("schema_version", REPORT_SCHEMA),
            records=tuple(
                MechanismDispositionRecordV1.from_dict(r) for r in data.get("records", [])
            ),
            supersessions=tuple(
                SupersessionEntryV1.from_dict(s) for s in data.get("supersessions", [])
            ),
            report_hash=str(data.get("report_hash", "")),
        )
        payload = report.to_dict()
        payload.pop("report_hash", None)
        if _digest(payload) != report.report_hash:
            raise ValueError(
                "mechanism disposition report hash mismatch (tampered or corrupted payload)"
            )
        return report

    def to_markdown(self) -> str:
        lines = [
            "# Mechanism disposition report",
            "",
            "| Mechanism | Disposition | Default state | Evidence class | Rationale |",
            "| --- | --- | --- | --- | --- |",
        ]
        for record in self.records:
            lines.append(
                f"| {record.mechanism_id} | {record.disposition.value} | "
                f"{record.default_state} | {record.evidence_class} | {record.rationale} |"
            )
        if self.supersessions:
            lines += ["", "## Supersessions", ""]
            for entry in self.supersessions:
                lines.append(
                    f"- `{entry.superseded_mechanism_id}`@`{entry.superseded_evidence_cutoff}` "
                    f"superseded by `{entry.superseding_mechanism_id}`@"
                    f"`{entry.superseding_evidence_cutoff}`: {entry.reason}"
                )
        return "\n".join(lines)


def build_disposition_report(
    records: tuple[MechanismDispositionRecordV1, ...],
    supersessions: tuple[SupersessionEntryV1, ...] = (),
) -> MechanismDispositionReportV1:
    """Assemble the closeout report from structured records -- every record
    passed in appears in the output; none are dropped by disposition."""
    report = MechanismDispositionReportV1(
        schema_version=REPORT_SCHEMA,
        records=tuple(records),
        supersessions=tuple(supersessions),
        report_hash="",
    )
    payload = report.to_dict()
    payload.pop("report_hash", None)
    return replace(report, report_hash=_digest(payload))


def disposition_report_integrity_ok(report: MechanismDispositionReportV1) -> bool:
    payload = report.to_dict()
    payload.pop("report_hash", None)
    return _digest(payload) == report.report_hash


__all__ = [
    "AUTHORITY_CLASSES",
    "DEFAULT_STATES",
    "EVIDENCE_CLASSES",
    "RECORD_SCHEMA",
    "REPORT_SCHEMA",
    "SUPERSESSION_SCHEMA",
    "Disposition",
    "MechanismDispositionRecordV1",
    "MechanismDispositionReportV1",
    "SupersessionEntryV1",
    "append_supersession",
    "build_disposition_report",
    "build_mechanism_disposition_record",
    "build_supersession_entry",
    "disposition_record_integrity_ok",
    "disposition_report_integrity_ok",
]
