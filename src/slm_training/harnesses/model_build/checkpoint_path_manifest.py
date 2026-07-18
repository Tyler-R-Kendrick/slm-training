"""Checkpoint × decode-path compatibility manifest for EFS0-02.

Given the durable checkpoint references (from the SLM-103 provenance schema) and
the decode-path registry, this builds the versioned manifest the decode-invariance
factorial audit runs over: which checkpoints support which of the three decode
paths, with an explicit incompatible reason where a path would have to coerce the
checkpoint's target representation.

It is **fail-closed and honest**: every audited cell must be eval-only over a
*verified* checkpoint hash, and the manifest states plainly whether enough
complete three-path blocks exist to decide invariance. When no durable
checkpoints are resolvable, the manifest reports the audit as deferred rather
than inventing cells.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from slm_training.harnesses.model_build.checkpoint_reference import (
    DURABLE_CLAIM_CLASSES,
    UNKNOWN,
    CheckpointReferenceV1,
)
from slm_training.harnesses.model_build.decode_path import (
    REQUIRED_DECODE_PATH_IDS,
    compatible_decode_paths,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "MIN_COMPLETE_BLOCKS",
    "CheckpointDecodeIdentity",
    "identity_from_reference_and_meta",
    "build_compatibility_manifest",
    "validate_compatibility_manifest",
]

MANIFEST_SCHEMA = "checkpoint_decode_manifest/v1"

# The audit needs at least this many complete three-path blocks to decide
# invariance (EFS0-02 acceptance criteria).
MIN_COMPLETE_BLOCKS = 6


@dataclass(frozen=True)
class CheckpointDecodeIdentity:
    """The checkpoint identity a decode-path compatibility predicate keys on."""

    run_id: str
    checkpoint_role: str
    sha256: str
    remote_uri: str
    claim_class: str
    model_family: str
    output_codec: str
    output_contract_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def identity_from_reference_and_meta(
    ref: CheckpointReferenceV1, meta: Mapping[str, Any]
) -> CheckpointDecodeIdentity:
    """Derive decode identity from a reference + the checkpoint ``.meta.json``.

    ``model_family``/``output_codec`` come from the checkpoint metadata, never
    guessed. A missing ``output_tokenizer`` defaults to ``compositional`` (the
    ``TwoTowerConfig`` default), matching the loader.
    """
    config = meta.get("config") or {}
    model_family = str(meta.get("kind", UNKNOWN))
    output_codec = str(config.get("output_tokenizer", "compositional"))
    contract = meta.get("output_contract_version")
    return CheckpointDecodeIdentity(
        run_id=ref.run_id,
        checkpoint_role=ref.checkpoint_role,
        sha256=ref.sha256,
        remote_uri=ref.remote_uri,
        claim_class=ref.claim_class,
        model_family=model_family,
        output_codec=output_codec,
        output_contract_version=None if contract is None else int(contract),
    )


def _cell(identity: CheckpointDecodeIdentity) -> dict[str, Any]:
    paths: list[dict[str, Any]] = []
    for spec, ok, reason in compatible_decode_paths(
        model_family=identity.model_family,
        output_codec=identity.output_codec,
        output_contract_version=identity.output_contract_version,
    ):
        paths.append(
            {
                "path_id": spec.path_id,
                "compatible": ok,
                "reason": reason,
                "completion_kind": spec.completion_kind,
                "config_overrides": (
                    spec.resolve_config_overrides(identity.output_codec) if ok else {}
                ),
                "runtime_override_fields": list(spec.runtime_override_fields()),
                "path_fingerprint": spec.fingerprint,
            }
        )
    compatible_required = {
        p["path_id"] for p in paths if p["compatible"] and p["path_id"] in REQUIRED_DECODE_PATH_IDS
    }
    complete_block = compatible_required == set(REQUIRED_DECODE_PATH_IDS)
    return {
        "checkpoint": identity.to_dict(),
        "paths": paths,
        "complete_block": complete_block,
    }


def build_compatibility_manifest(
    identities: Iterable[CheckpointDecodeIdentity],
) -> dict[str, Any]:
    """Build the versioned checkpoint × decode-path compatibility manifest."""
    cells = [_cell(identity) for identity in identities]
    cells.sort(key=lambda c: (c["checkpoint"]["run_id"], c["checkpoint"]["checkpoint_role"]))
    complete_blocks = sum(1 for c in cells if c["complete_block"])
    usable = complete_blocks >= MIN_COMPLETE_BLOCKS
    if not cells:
        note = (
            "No durable checkpoints supplied: the decode-invariance audit is "
            "deferred until frontier/diagnostic checkpoints are synced (see SLM-103)."
        )
    elif not usable:
        note = (
            f"Only {complete_blocks} complete three-path block(s) available; the "
            f"audit needs >= {MIN_COMPLETE_BLOCKS}. Runnable as a partial/diagnostic "
            "audit only until more durable checkpoints are synced."
        )
    else:
        note = f"{complete_blocks} complete three-path blocks available; audit is runnable."
    return {
        "schema_version": MANIFEST_SCHEMA,
        "required_paths": list(REQUIRED_DECODE_PATH_IDS),
        "min_complete_blocks": MIN_COMPLETE_BLOCKS,
        "checkpoint_count": len(cells),
        "complete_blocks": complete_blocks,
        "usable_for_audit": usable,
        "note": note,
        "cells": cells,
    }


def validate_compatibility_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Structural + honesty validation. Returns human-readable error strings.

    Every cell must be eval-only over a **verified** checkpoint hash, and any
    ``frontier``/``ship_candidate`` cell must resolve durably. This never
    weakens a gate — an under-provenanced durable checkpoint is an error, not a
    silently-accepted cell.
    """
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(
            f"unexpected manifest schema_version {manifest.get('schema_version')!r}"
        )
    for cell in manifest.get("cells", []):
        ckpt = cell.get("checkpoint", {})
        label = f"{ckpt.get('run_id')!r}/{ckpt.get('checkpoint_role')!r}"
        sha = ckpt.get("sha256")
        if not sha or sha == UNKNOWN:
            errors.append(f"{label}: checkpoint has no verified SHA-256 (eval cell must be hash-pinned)")
        if ckpt.get("claim_class") in DURABLE_CLAIM_CLASSES:
            remote = ckpt.get("remote_uri")
            if not remote or remote == UNKNOWN:
                errors.append(
                    f"{label}: {ckpt.get('claim_class')} checkpoint has no durable remote_uri"
                )
        if not cell.get("paths"):
            errors.append(f"{label}: no decode paths evaluated for compatibility")
    return errors
