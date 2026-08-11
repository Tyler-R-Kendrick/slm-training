"""in-toto / SLSA-compatible attestation envelopes (unsigned, local-only)."""

from __future__ import annotations

from typing import Any

from slm_training.evidence_interop._envelope import base_envelope
from slm_training.evidence_interop.authority import extract_authority
from slm_training.evidence_interop.profile import INTOTO_PREDICATE_TYPES

FORMAT = "intoto"


def export_intoto(obj: Any, *, predicate: str = "test_result") -> dict[str, Any]:
    """Emit an in-toto Statement-shaped envelope (no signatures, no network).

    Attestation packaging only. Does **not** grant semantic authority, prove
    ship gates, or substitute for FormalObject checker results.
    """
    if predicate not in INTOTO_PREDICATE_TYPES:
        raise ValueError(
            f"unsupported in-toto predicate {predicate!r}; "
            f"known={sorted(INTOTO_PREDICATE_TYPES)}"
        )
    authority = extract_authority(obj)
    primary = _primary_id(authority.authority_ids)
    digest = _best_digest(authority.authority_ids) or "0" * 64

    subject = [
        {
            "name": primary,
            "digest": {"sha256": _strip_algo(digest)},
        }
    ]
    suite = authority.authority_ids.get("suite_hashes") or {}
    if isinstance(suite, dict):
        for name, sha in sorted(suite.items()):
            subject.append(
                {"name": str(name), "digest": {"sha256": _strip_algo(str(sha))}}
            )
    for sha in authority.authority_ids.get("raw_envelope_sha256") or ():
        subject.append(
            {"name": "raw_envelope", "digest": {"sha256": _strip_algo(str(sha))}}
        )

    predicate_body: dict[str, Any] = {
        "result": "PASSED_PROJECTION_ONLY",
        "passed": True,
        "note": (
            "Projection packaging only; does not certify checkers, ship gates, "
            "or campaign locks."
        ),
        "authority_ids": dict(authority.authority_ids),
    }
    if predicate == "slsa_provenance":
        predicate_body = {
            "buildDefinition": {
                "buildType": "https://slm-training.invalid/build/projection/v1",
                "externalParameters": {"kind": authority.kind},
            },
            "runDetails": {
                "builder": {"id": "urn:slm:agent:evidence-interop-exporter"},
            },
            "authority_ids": dict(authority.authority_ids),
        }

    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subject,
        "predicateType": INTOTO_PREDICATE_TYPES[predicate],
        "predicate": predicate_body,
    }

    payload = base_envelope(
        kind=authority.kind, authority=authority, format_name=FORMAT
    )
    payload.update(
        {
            "payloadType": "application/vnd.in-toto+json",
            "payload": statement,
            "signatures": [],
        }
    )
    return payload


def _primary_id(ids: dict[str, Any]) -> str:
    for key in ("object_id", "run_id", "campaign_id", "snapshot_id"):
        value = ids.get(key)
        if value:
            return str(value)
    return "unknown"


def _best_digest(ids: dict[str, Any]) -> str | None:
    for key in (
        "content_digest",
        "config_sha256",
        "records_sha",
        "locked_eval_manifest_sha256",
        "corpus_sha256",
        "source_commit",
    ):
        value = ids.get(key)
        if value:
            return str(value)
    return None


def _strip_algo(digest: str) -> str:
    if digest.startswith("sha256:"):
        return digest[7:]
    return digest
