"""in-toto / SLSA-compatible attestation envelopes (unsigned, local-only)."""

from __future__ import annotations

from typing import Any, Mapping

from slm_training.evidence_interop._envelope import base_envelope
from slm_training.evidence_interop.authority import extract_authority
from slm_training.evidence_interop.profile import INTOTO_PREDICATE_TYPES

FORMAT = "intoto"


def export_intoto(obj: Any, *, predicate: str = "slsa_provenance") -> dict[str, Any]:
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
    digest = _best_digest(authority.authority_ids)
    if digest is None:
        raise ValueError(
            "in-toto export requires a content digest in authority_ids "
            "(no synthetic zero digest)"
        )

    subject = [
        {
            "name": primary,
            "digest": {"sha256": _strip_algo(digest)},
        }
    ]
    suite = authority.authority_ids.get("suite_hashes") or {}
    if isinstance(suite, Mapping):
        for name, sha in sorted(suite.items()):
            subject.append(
                {"name": str(name), "digest": {"sha256": _strip_algo(str(sha))}}
            )
    for sha in authority.authority_ids.get("raw_envelope_sha256") or ():
        subject.append(
            {"name": "raw_envelope", "digest": {"sha256": _strip_algo(str(sha))}}
        )

    # Never emit a PASSED test-result without native test evidence. Default is
    # SLSA provenance packaging; test_result stays FAIL-closed / documentary.
    if predicate == "test_result":
        predicate_body: dict[str, Any] = {
            "result": "FAILED",
            "configuration": [],
            "note": (
                "No verified native test evidence bound to this projection; "
                "unsigned packaging only — not a ship-gate or checker pass."
            ),
            "authority_ids": dict(authority.authority_ids),
        }
    elif predicate == "slsa_provenance":
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
    else:
        predicate_body = {
            "attributes": [
                {
                    "attribute": "slm.projection_only",
                    "value": True,
                }
            ],
            "authority_ids": dict(authority.authority_ids),
            "note": "Unsigned SCAI-shaped projection packaging only.",
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


def _primary_id(ids: Mapping[str, Any]) -> str:
    for key in ("object_id", "run_id", "campaign_id", "snapshot_id"):
        value = ids.get(key)
        if value:
            return str(value)
    return "unknown"


def _best_digest(ids: Mapping[str, Any]) -> str | None:
    for key in (
        "content_digest",
        "config_sha256",
        "records_sha",
        "locked_eval_manifest_sha256",
        "corpus_sha256",
    ):
        value = ids.get(key)
        if not value:
            continue
        text = _strip_algo(str(value)).lower()
        if len(text) == 64 and all(c in "0123456789abcdef" for c in text):
            return text
    return None


def _strip_algo(digest: str) -> str:
    if digest.startswith("sha256:"):
        return digest[7:]
    return digest
