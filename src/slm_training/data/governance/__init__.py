"""Deterministic rights, provenance, and safety gates for dataset records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from slm_training.dsl.schema import ExampleRecord

ASSET_CLASSES = frozenset({"images", "fonts", "icons", "embedded"})
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
_SECRET_RES = (
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{8,}",
        re.I,
    ),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_-]{20,})\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_INSTRUCTION_RE = re.compile(
    r"ignore (?:all |any )?(?:previous|prior) instructions|"
    r"(?:reveal|print|show) (?:the )?system prompt|"
    r"<\/?(?:tool_call|system|assistant)>|"
    r"(?:call|invoke|run) (?:the )?(?:tool|shell|command)",
    re.I,
)


def content_hash(content: str | bytes) -> str:
    """Return a reproducible SHA-256 over the acquired source bytes."""
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ScanResult:
    """Privacy/security findings without retaining matched sensitive values."""

    pii_kinds: tuple[str, ...] = ()
    secret_kinds: tuple[str, ...] = ()
    instruction_like: bool = False

    @property
    def clean(self) -> bool:
        return not self.pii_kinds and not self.secret_kinds

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "pii_kinds": list(self.pii_kinds),
            "secret_kinds": list(self.secret_kinds),
            "instruction_like": self.instruction_like,
        }


def scan_untrusted_text(text: str) -> ScanResult:
    """Classify inert page/user text; this function never evaluates its content."""
    pii: list[str] = []
    if _EMAIL_RE.search(text):
        pii.append("email")
    if any(
        len(re.sub(r"\D", "", match.group())) >= 10
        for match in _PHONE_RE.finditer(text)
    ):
        pii.append("phone")

    secrets: list[str] = []
    for name, pattern in zip(
        ("credential_assignment", "token", "private_key"),
        _SECRET_RES,
        strict=True,
    ):
        if pattern.search(text):
            secrets.append(name)
    return ScanResult(
        pii_kinds=tuple(pii),
        secret_kinds=tuple(secrets),
        instruction_like=bool(_INSTRUCTION_RE.search(text)),
    )


@dataclass(frozen=True)
class SourceProvenance:
    """Acquisition and rights evidence attached to an external source."""

    source_url: str
    domain: str
    acquisition_date: str
    terms_policy_id: str
    legal_basis: str
    license: str
    attribution: str
    asset_rights: Mapping[str, str]
    robots_policy: str
    deletion_procedure: str
    content_hash: str
    transformation_history: tuple[str, ...] = ()
    teacher_model: str | None = None
    prompt_version: str | None = None

    @classmethod
    def from_content(
        cls,
        *,
        source_url: str,
        acquisition_date: str,
        terms_policy_id: str,
        legal_basis: str,
        license: str,
        attribution: str,
        asset_rights: Mapping[str, str],
        robots_policy: str,
        deletion_procedure: str,
        content: str | bytes,
        transformation_history: Iterable[str] = (),
        teacher_model: str | None = None,
        prompt_version: str | None = None,
    ) -> SourceProvenance:
        return cls(
            source_url=source_url,
            domain=(urlparse(source_url).hostname or "").lower(),
            acquisition_date=acquisition_date,
            terms_policy_id=terms_policy_id,
            legal_basis=legal_basis,
            license=license,
            attribution=attribution,
            asset_rights=dict(asset_rights),
            robots_policy=robots_policy,
            deletion_procedure=deletion_procedure,
            content_hash=content_hash(content),
            transformation_history=tuple(transformation_history),
            teacher_model=teacher_model,
            prompt_version=prompt_version,
        )

    def validation_errors(self, *, require_teacher: bool = False) -> tuple[str, ...]:
        errors: list[str] = []
        parsed = urlparse(self.source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            errors.append("source_url")
        if str(self.domain).lower() != (parsed.hostname or "").lower():
            errors.append("domain")
        try:
            date.fromisoformat(self.acquisition_date)
        except ValueError:
            errors.append("acquisition_date")
        required = {
            "terms_policy_id": self.terms_policy_id,
            "legal_basis": self.legal_basis,
            "license": self.license,
            "attribution": self.attribution,
            "robots_policy": self.robots_policy,
            "deletion_procedure": self.deletion_procedure,
        }
        errors.extend(
            name for name, value in required.items() if not str(value or "").strip()
        )
        if str(self.legal_basis).strip().lower() in {"robots", "robots.txt"}:
            errors.append("legal_basis_is_not_robots_policy")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_hash):
            errors.append("content_hash")
        missing_assets = sorted(ASSET_CLASSES - set(self.asset_rights))
        errors.extend(f"asset_rights.{name}" for name in missing_assets)
        errors.extend(
            f"asset_rights.{name}"
            for name in ASSET_CLASSES
            if not str(self.asset_rights.get(name, "")).strip()
        )
        if bool(self.teacher_model) != bool(self.prompt_version):
            errors.append("teacher_model_prompt_version_pair")
        if require_teacher and not (self.teacher_model and self.prompt_version):
            errors.append("teacher_provenance")
        if any(not str(step).strip() for step in self.transformation_history):
            errors.append("transformation_history")
        return tuple(dict.fromkeys(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "domain": self.domain,
            "acquisition_date": self.acquisition_date,
            "terms_policy_id": self.terms_policy_id,
            "legal_basis": self.legal_basis,
            "license": self.license,
            "attribution": self.attribution,
            "asset_rights": dict(sorted(self.asset_rights.items())),
            "robots_policy": self.robots_policy,
            "deletion_procedure": self.deletion_procedure,
            "content_hash": self.content_hash,
            "transformation_history": list(self.transformation_history),
            "teacher_model": self.teacher_model,
            "prompt_version": self.prompt_version,
        }


def govern_record(
    record: ExampleRecord,
    provenance: SourceProvenance | None,
    *,
    raw_content: str | None = None,
    teacher_generated: bool = False,
) -> ExampleRecord:
    """Attach evidence and quarantine incomplete, PII-bearing, or secret-bearing rows."""
    literal = (
        raw_content
        if raw_content is not None
        else "\n".join(
            part for part in (record.prompt, record.openui, record.design_md) if part
        )
    )
    scan = scan_untrusted_text(literal)
    reasons = list(
        provenance.validation_errors(require_teacher=teacher_generated)
        if provenance
        else ("provenance",)
    )
    if raw_content is not None and provenance:
        if provenance.content_hash != content_hash(raw_content):
            reasons.append("content_hash_mismatch")
    reasons.extend(f"pii.{kind}" for kind in scan.pii_kinds)
    reasons.extend(f"secret.{kind}" for kind in scan.secret_kinds)
    status = "Quarantined" if reasons else "Complete"
    meta = {
        **record.meta,
        "provenance_complete": not reasons,
        "governance": {
            "status": status,
            "reasons": reasons,
            "source": provenance.to_dict() if provenance else None,
            "scan": scan.to_dict(),
        },
    }
    governed = ExampleRecord(
        id=record.id,
        prompt=record.prompt,
        openui=record.openui,
        placeholders=list(record.placeholders),
        split=record.split,
        source=record.source,
        meta=meta,
        design_md=record.design_md,
    )
    from slm_training.data.verify import stamp_record

    return stamp_record(governed)


def emit_dataset_metadata(
    records: Iterable[ExampleRecord],
    output_dir: Path | str,
    *,
    name: str,
    version: str,
) -> dict[str, Path]:
    """Write reproducible Croissant, Data Card, and SPDX JSON documents."""
    rows = sorted(records, key=lambda record: record.id)
    governed = [
        item
        for record in rows
        if isinstance((item := record.meta.get("governance")), dict)
    ]
    sources = sorted(
        {
            source["source_url"]
            for item in governed
            if isinstance((source := item.get("source")), dict)
            and source.get("source_url")
        }
    )
    complete = sum(item.get("status") == "Complete" for item in governed)
    quarantined = len(governed) - complete
    internal = len(rows) - len(governed)
    pii_flagged = sum(
        bool((item.get("scan") or {}).get("pii_kinds")) for item in governed
    )
    secret_flagged = sum(
        bool((item.get("scan") or {}).get("secret_kinds")) for item in governed
    )
    instruction_like = sum(
        bool((item.get("scan") or {}).get("instruction_like")) for item in governed
    )
    records_text = "".join(
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for record in rows
    )
    digest = content_hash(records_text)
    acquired = sorted(
        source["acquisition_date"]
        for item in governed
        if isinstance((source := item.get("source")), dict)
        and source.get("acquisition_date")
    )
    created = f"{acquired[-1] if acquired else '1970-01-01'}T00:00:00Z"

    documents = {
        "croissant.json": {
            "@context": {
                "@vocab": "https://schema.org/",
                "cr": "http://mlcommons.org/croissant/",
                "dct": "http://purl.org/dc/terms/",
                "sc": "https://schema.org/",
                "conformsTo": "dct:conformsTo",
                "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
                "extract": "cr:extract",
                "field": "cr:field",
                "fileObject": "cr:fileObject",
                "jsonPath": "cr:jsonPath",
                "recordSet": "cr:recordSet",
                "source": "cr:source",
            },
            "@type": "sc:Dataset",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "name": name,
            "version": version,
            "description": "Governed OpenUI training records.",
            "url": "https://github.com/Tyler-R-Kendrick/slm-training",
            "distribution": [
                {
                    "@type": "cr:FileObject",
                    "@id": "records.jsonl",
                    "name": "records.jsonl",
                    "contentUrl": "records.jsonl",
                    "encodingFormat": "application/jsonl",
                    "sha256": digest,
                }
            ],
            "recordSet": [
                {
                    "@type": "cr:RecordSet",
                    "@id": "records",
                    "name": "records",
                    "field": [
                        {
                            "@type": "cr:Field",
                            "@id": f"records/{field}",
                            "name": field,
                            "dataType": kind,
                            "source": {
                                "fileObject": {"@id": "records.jsonl"},
                                "extract": {"jsonPath": f"$.{field}"},
                            },
                        }
                        for field, kind in (
                            ("id", "sc:Text"),
                            ("prompt", "sc:Text"),
                            ("openui", "sc:Text"),
                            ("meta", "sc:StructuredValue"),
                        )
                    ],
                }
            ],
        },
        "data_card.json": {
            "schema_version": "1.0",
            "name": name,
            "version": version,
            "record_count": len(rows),
            "governance": {
                "complete": complete,
                "quarantined": quarantined,
                "internal": internal,
                "external_sources": sources,
                "robots_policy_is_not_authorization": True,
                "pii_and_secret_scan_required": True,
                "pii_flagged": pii_flagged,
                "secret_flagged": secret_flagged,
                "instruction_like_content": instruction_like,
            },
        },
        "dataset.spdx.json": {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": name,
            "documentNamespace": f"https://slm-training.invalid/spdx/{digest}",
            "creationInfo": {
                "created": created,
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
                    "licenseDeclared": "NOASSERTION",
                    "copyrightText": "NOASSERTION",
                    "checksums": [{"algorithm": "SHA256", "checksumValue": digest}],
                    "sourceInfo": "External source rights are recorded per row.",
                }
            ],
            "documentDescribes": ["SPDXRef-Dataset"],
        },
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.jsonl").write_text(records_text, encoding="utf-8")
    paths: dict[str, Path] = {}
    for filename, document in documents.items():
        path = output / filename
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths[filename] = path
    return paths


__all__ = [
    "ASSET_CLASSES",
    "ScanResult",
    "SourceProvenance",
    "content_hash",
    "emit_dataset_metadata",
    "govern_record",
    "scan_untrusted_text",
]
