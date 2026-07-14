"""Fail-closed governance for external and teacher-derived corpus rows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from slm_training.dsl.schema import ExampleRecord

_EXTERNAL_SOURCES = frozenset(
    {"awwwards", "rico", "web_distilled", "external", "external_web"}
)
_TEACHER_SOURCES = frozenset(
    {"frontier_described", "self_distilled_success", "self_distilled_repair", "teacher"}
)
_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}
_SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{8,}"
    ),
}
_ASSET_KINDS = ("images", "fonts", "icons", "embedded_assets")


@dataclass(frozen=True)
class AssetRights:
    """Rights disposition for assets that may accompany a source page."""

    images: str
    fonts: str
    icons: str
    embedded_assets: str

    def missing(self) -> tuple[str, ...]:
        return tuple(
            f"asset_rights.{name}"
            for name in _ASSET_KINDS
            if not str(getattr(self, name)).strip()
        )


@dataclass(frozen=True)
class SourceGovernance:
    """Legal/provenance evidence kept separately from robots guidance."""

    source_url: str | None = None
    domain: str | None = None
    acquisition_date: str | None = None
    terms_snapshot: str | None = None
    policy_id: str | None = None
    rights_basis: str | None = None
    license: str | None = None
    attribution: str | None = None
    asset_rights: AssetRights | None = None
    robots_policy: str | None = None
    robots_checked_at: str | None = None
    content_hash: str | None = None
    withdrawal_procedure: str | None = None
    transformation_history: tuple[str, ...] = ()
    teacher_model: str | None = None
    prompt_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContentScan:
    """Finding categories only; raw PII and secret values are never persisted."""

    pii_types: tuple[str, ...] = ()
    secret_types: tuple[str, ...] = ()
    scanned_fields: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.pii_types and not self.secret_types

    def to_dict(self) -> dict[str, Any]:
        return {
            "pii_types": list(self.pii_types),
            "secret_types": list(self.secret_types),
            "scanned_fields": list(self.scanned_fields),
            "passed": self.passed,
        }


def record_content_hash(record: ExampleRecord) -> str:
    """Bind governance evidence to the exact prompt, target, and design text."""
    payload = json.dumps(
        [record.prompt, record.openui, record.design_md],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def scan_record(record: ExampleRecord) -> ContentScan:
    """Inspect untrusted text as inert strings; never parse it as instructions."""
    meta = {key: value for key, value in record.meta.items() if key != "governance"}
    fields = {
        "prompt": record.prompt,
        "openui": record.openui,
        "design_md": record.design_md or "",
        "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str),
    }
    pii: set[str] = set()
    secrets: set[str] = set()
    for text in fields.values():
        pii.update(
            name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)
        )
        secrets.update(
            name for name, pattern in _SECRET_PATTERNS.items() if pattern.search(text)
        )
    return ContentScan(
        pii_types=tuple(sorted(pii)),
        secret_types=tuple(sorted(secrets)),
        scanned_fields=tuple(fields),
    )


def _is_external(record: ExampleRecord) -> bool:
    return record.source in _EXTERNAL_SOURCES or any(
        record.meta.get(key) for key in ("source_url", "url", "domain")
    )


def _is_teacher(record: ExampleRecord) -> bool:
    return record.source in _TEACHER_SOURCES or any(
        record.meta.get(key) for key in ("teacher_model", "prompt_version")
    )


def _missing_external(governance: SourceGovernance | None) -> list[str]:
    if governance is None:
        return ["governance"]
    missing = [
        name
        for name in (
            "source_url",
            "domain",
            "acquisition_date",
            "rights_basis",
            "license",
            "attribution",
            "robots_policy",
            "robots_checked_at",
            "withdrawal_procedure",
        )
        if not str(getattr(governance, name) or "").strip()
    ]
    if not (governance.terms_snapshot or governance.policy_id):
        missing.append("terms_snapshot_or_policy_id")
    if governance.asset_rights is None:
        missing.append("asset_rights")
    else:
        missing.extend(governance.asset_rights.missing())
    if not governance.transformation_history:
        missing.append("transformation_history")
    return missing


def _evidence_errors(
    record: ExampleRecord,
    governance: SourceGovernance | None,
    *,
    external: bool,
    teacher: bool,
) -> list[str]:
    errors = (
        [f"missing:{name}" for name in _missing_external(governance)]
        if external
        else []
    )
    if governance is None:
        if teacher:
            errors.extend(("missing:teacher_model", "missing:prompt_version"))
        return errors
    if teacher:
        for name in ("teacher_model", "prompt_version"):
            if not str(getattr(governance, name) or "").strip():
                errors.append(f"missing:{name}")
    if governance.source_url:
        parsed = urlparse(governance.source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            errors.append("invalid:source_url")
        elif governance.domain and parsed.hostname.lower() != governance.domain.lower():
            errors.append("mismatch:domain")
    for name in ("acquisition_date", "robots_checked_at"):
        value = getattr(governance, name)
        if value:
            try:
                date.fromisoformat(value)
            except ValueError:
                errors.append(f"invalid:{name}")
    actual_hash = record_content_hash(record)
    if governance.content_hash and governance.content_hash != actual_hash:
        errors.append("mismatch:content_hash")
    return errors


def govern_record(
    record: ExampleRecord,
    governance: SourceGovernance | None = None,
) -> ExampleRecord:
    """Stamp evidence and quarantine unsafe or incomplete external/teacher rows."""
    external = _is_external(record)
    teacher = _is_teacher(record)
    scan = scan_record(record)
    reasons = _evidence_errors(
        record,
        governance,
        external=external,
        teacher=teacher,
    )
    reasons.extend(f"pii:{name}" for name in scan.pii_types)
    reasons.extend(f"secret:{name}" for name in scan.secret_types)
    reasons = sorted(set(reasons))

    evidence = governance.to_dict() if governance is not None else {}
    evidence["content_hash"] = record_content_hash(record)
    evidence["scan"] = scan.to_dict()
    evidence["external"] = external
    evidence["teacher_derived"] = teacher
    evidence["status"] = "quarantined" if reasons else "eligible"
    evidence["reasons"] = reasons
    evidence["robots_is_access_authorization"] = False

    meta = {**record.meta, "governance": evidence}
    if reasons:
        meta["tier"] = "Quarantine"
    elif (external or teacher) and "tier" not in meta:
        meta["tier"] = "Bronze"
    return ExampleRecord(
        id=record.id,
        prompt=record.prompt,
        openui=record.openui,
        placeholders=list(record.placeholders),
        split=record.split,
        source=record.source,
        meta=meta,
        design_md=record.design_md,
    )


def emit_dataset_metadata(
    output_dir: Path | str,
    records: Iterable[ExampleRecord],
    *,
    name: str,
    version: str,
    description: str,
    created_at: str,
) -> dict[str, Path]:
    """Emit deterministic Croissant, Data Card, and SPDX 2.3 JSON artifacts."""
    rows = sorted(records, key=lambda record: record.id)
    if not rows:
        raise ValueError("at least one record is required")
    try:
        datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("created_at must be UTC YYYY-MM-DDTHH:MM:SSZ") from exc

    digest = hashlib.sha256(
        "\n".join(
            json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True)
            for row in rows
        ).encode()
    ).hexdigest()
    governance_rows = [dict(row.meta.get("governance") or {}) for row in rows]
    licenses = sorted(
        {str(item["license"]) for item in governance_rows if item.get("license")}
    )
    domains = sorted(
        {str(item["domain"]) for item in governance_rows if item.get("domain")}
    )
    quarantined = sum(item.get("status") == "quarantined" for item in governance_rows)
    summary = {
        "record_count": len(rows),
        "eligible_count": len(rows) - quarantined,
        "quarantined_count": quarantined,
        "sha256": digest,
        "licenses": licenses,
        "source_domains": domains,
    }

    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
        },
        "@type": "Dataset",
        "name": name,
        "description": description,
        "version": version,
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": licenses,
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "records.jsonl",
                "name": "records.jsonl",
                "encodingFormat": "application/x-ndjson",
                "sha256": digest,
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "records",
                "name": "records",
                "description": f"{len(rows)} governed OpenUI records",
            }
        ],
    }
    data_card = {
        "schema_version": "1.0",
        "name": name,
        "version": version,
        "description": description,
        "created_at": created_at,
        "summary": summary,
        "governance": {
            "rights_required_for_external_records": True,
            "robots_is_access_authorization": False,
            "withdrawal_supported": True,
            "pii_and_secret_scanning": True,
        },
    }
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "dataset"
    license_expression = " AND ".join(licenses) if licenses else "NOASSERTION"
    spdx = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{name}-{version}",
        "documentNamespace": f"https://spdx.org/spdxdocs/{slug}-{version}-{digest[:16]}",
        "creationInfo": {
            "created": created_at,
            "creators": ["Tool: slm-training-governance"],
        },
        "packages": [
            {
                "name": name,
                "SPDXID": "SPDXRef-Dataset",
                "versionInfo": version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": license_expression,
                "copyrightText": "NOASSERTION",
                "checksums": [{"algorithm": "SHA256", "checksumValue": digest}],
                "primaryPackagePurpose": "DATA",
            }
        ],
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Dataset",
            }
        ],
    }

    output = Path(output_dir)
    paths = {
        "croissant": output / "croissant.json",
        "data_card": output / "data-card.json",
        "spdx": output / "dataset.spdx.json",
    }
    for key, payload in (
        ("croissant", croissant),
        ("data_card", data_card),
        ("spdx", spdx),
    ):
        _write_json(paths[key], payload)
    return paths


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
