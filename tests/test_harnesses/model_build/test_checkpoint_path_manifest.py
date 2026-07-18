"""Checkpoint x decode-path compatibility manifest."""

from __future__ import annotations

from slm_training.harnesses.model_build.checkpoint_path_manifest import (
    MIN_COMPLETE_BLOCKS,
    CheckpointDecodeIdentity,
    build_compatibility_manifest,
    identity_from_reference_and_meta,
    validate_compatibility_manifest,
)
from slm_training.harnesses.model_build.checkpoint_reference import CheckpointReferenceV1


def _choice_identity(run_id: str) -> CheckpointDecodeIdentity:
    return CheckpointDecodeIdentity(
        run_id=run_id,
        checkpoint_role="last",
        sha256="b" * 64,
        remote_uri=f"hf://buckets/TKendrick/OpenUI/checkpoints/{run_id}/last.pt",
        claim_class="frontier",
        model_family="twotower",
        output_codec="choice",
        output_contract_version=1,
    )


def test_identity_from_meta_reads_family_and_codec() -> None:
    ref = CheckpointReferenceV1(
        run_id="r", claim_class="fixture", checkpoint_role="last", checkpoint_filename="last.pt"
    )
    meta = {"kind": "twotower", "config": {"output_tokenizer": "choice"}, "output_contract_version": 2}
    ident = identity_from_reference_and_meta(ref, meta)
    assert ident.model_family == "twotower"
    assert ident.output_codec == "choice"
    assert ident.output_contract_version == 2
    # Missing output_tokenizer defaults to compositional (loader default).
    ident2 = identity_from_reference_and_meta(ref, {"kind": "twotower", "config": {}})
    assert ident2.output_codec == "compositional"


def test_manifest_counts_complete_blocks_and_usability() -> None:
    idents = [_choice_identity(f"e{i}") for i in range(MIN_COMPLETE_BLOCKS)]
    manifest = build_compatibility_manifest(idents)
    assert manifest["complete_blocks"] == MIN_COMPLETE_BLOCKS
    assert manifest["usable_for_audit"] is True
    # One fewer complete block => not usable, honest note.
    fewer = build_compatibility_manifest(idents[:-1])
    assert fewer["usable_for_audit"] is False
    assert "needs >=" in fewer["note"]


def test_empty_manifest_is_deferred_not_invented() -> None:
    manifest = build_compatibility_manifest([])
    assert manifest["checkpoint_count"] == 0
    assert manifest["usable_for_audit"] is False
    assert "deferred" in manifest["note"]
    assert validate_compatibility_manifest(manifest) == []


def test_validation_flags_unverified_and_undurable_durable_cells() -> None:
    # frontier checkpoint with no durable remote and no hash -> errors.
    bad = CheckpointDecodeIdentity(
        run_id="x",
        checkpoint_role="last",
        sha256="UNKNOWN",
        remote_uri="UNKNOWN",
        claim_class="frontier",
        model_family="twotower",
        output_codec="choice",
    )
    errors = validate_compatibility_manifest(build_compatibility_manifest([bad]))
    assert any("SHA-256" in e for e in errors)
    assert any("durable remote_uri" in e for e in errors)
    # A properly-provenanced frontier checkpoint validates cleanly.
    assert validate_compatibility_manifest(
        build_compatibility_manifest([_choice_identity("ok")])
    ) == []
