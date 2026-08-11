"""EVID-12 / SLM-543 thin evidence interoperability profile.

Maps repository-native formal/experiment evidence onto established
provenance/attestation/artifact standards. External formats are **projections
only** — repository IDs, digests, and checker capabilities remain authoritative.
"""

from __future__ import annotations

from typing import Final

PROFILE_ID: Final = "slm-evidence-interop/v1"
PROFILE_VERSION: Final = "v1"
ISSUE: Final = "SLM-543"
KEY: Final = "EVID-12"
SEMANTIC_AUTHORITY_DEFAULT: Final = False

# Embedded in every projection so importers can find authority without guessing.
AUTHORITY_BLOCK_KEY: Final = "slm:authority"
PROFILE_BLOCK_KEY: Final = "slm:profile"
SEMANTIC_AUTHORITY_KEY: Final = "slm:semantic_authority"
KIND_KEY: Final = "slm:kind"
UNSUPPORTED_KEY: Final = "slm:unsupported_fields"

# Authority-relevant identifiers that must survive round-trip or fail closed.
AUTHORITY_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "formal_object": (
        "object_id",
        "content_digest",
        "required_checkers",
    ),
    "evidence_bundle": (
        "run_id",
        "config_sha256",
        "corpus_sha256",
        "checkpoint_sha256",
        "suite_hashes",
        "raw_envelope_sha256",
    ),
    "experiment_campaign": (
        "campaign_id",
        "locked_eval_manifest_sha256",
        "source_commit",
    ),
    "data_snapshot": (
        "snapshot_id",
        "records_sha",
        "annotations_sha",
    ),
}

# W3C PROV-O role mapping (entities / activities / agents / derivations).
PROV_O_MAPPING: Final[dict[str, str]] = {
    "entity": "prov:Entity",
    "activity": "prov:Activity",
    "agent": "prov:Agent",
    "derivation": "prov:wasDerivedFrom",
    "generation": "prov:wasGeneratedBy",
    "association": "prov:wasAssociatedWith",
    "attribution": "prov:wasAttributedTo",
}

# Portable research-object metadata (RO-Crate) — thin subset.
RO_CRATE_MAPPING: Final[dict[str, str]] = {
    "root": "RootDataset",
    "file": "File",
    "dataset": "Dataset",
    "person": "Person",
    "organization": "Organization",
    "software": "SoftwareApplication",
}

# SPDX 3 AI / Dataset / Build thin projection kinds.
SPDX_MAPPING: Final[dict[str, str]] = {
    "formal_object": "software_Package",
    "evidence_bundle": "build_Build",
    "experiment_campaign": "ai_AIPackage",
    "data_snapshot": "dataset_Dataset",
}

# in-toto / SLSA-compatible predicate types (URIs are documentary; no network).
INTOTO_PREDICATE_TYPES: Final[dict[str, str]] = {
    "test_result": "https://in-toto.io/attestation/test-result/v0.1",
    "slsa_provenance": "https://slsa.dev/provenance/v1",
    "scai": "https://in-toto.io/attestation/scai/attribute-report/v0.2",
}

# OCI subject / referrer descriptor media types.
OCI_MEDIA_TYPES: Final[dict[str, str]] = {
    "descriptor": "application/vnd.oci.descriptor.v1+json",
    "referrer_index": "application/vnd.oci.image.index.v1+json",
    "slm_authority": "application/vnd.slm-training.authority.v1+json",
}

SUPPORTED_FORMATS: Final[tuple[str, ...]] = (
    "prov-o-jsonld",
    "ro-crate",
    "spdx3",
    "intoto",
    "oci-referrer",
)

# Predicates / keys an importer may see from the wild without treating them
# as authority. Listed explicitly so unknown fields fail closed.
KNOWN_EXTERNAL_PREDICATES: Final[frozenset[str]] = frozenset(
    {
        "@context",
        "@id",
        "@type",
        "@graph",
        "prov:Entity",
        "prov:Activity",
        "prov:Agent",
        "prov:wasDerivedFrom",
        "prov:wasGeneratedBy",
        "prov:wasAssociatedWith",
        "prov:wasAttributedTo",
        "prov:label",
        "prov:value",
        "spdxVersion",
        "SPDXID",
        "name",
        "creationInfo",
        "packages",
        "documentDescribes",
        "dataLicense",
        "documentNamespace",
        "checksums",
        "software_Package",
        "build_Build",
        "ai_AIPackage",
        "dataset_Dataset",
        "_crate",
        "@graph",
        "payloadType",
        "payload",
        "predicateType",
        "predicate",
        "subject",
        "mediaType",
        "digest",
        "size",
        "annotations",
        "artifactType",
        "schema_version",
        "schema",
        "descriptor",
        "referrers",
        "elements",
        "signatures",
        "_type",
        "identifier",
        "encodingFormat",
        "hasPart",
        "description",
        "sha256",
        "version",
    }
) | frozenset(
    {
        PROFILE_BLOCK_KEY,
        AUTHORITY_BLOCK_KEY,
        SEMANTIC_AUTHORITY_KEY,
        KIND_KEY,
        UNSUPPORTED_KEY,
    }
)
