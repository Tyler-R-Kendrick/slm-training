"""Shared persisted-record contracts for data admission and trainer preparation."""

from slm_training.dsl.schema import ExampleRecord


def assert_training_record(record: ExampleRecord) -> None:
    """Reject invalid records without rewriting targets or acquiring model weights."""
    from slm_training.data.contract import (
        assert_canonical_template_markers,
        assert_no_template_semantic_labels,
    )
    from slm_training.dsl.analysis.templatize import assert_role_safe_output
    from slm_training.dsl.language_contract import assert_symbol_only_output

    assert_no_template_semantic_labels(record.prompt, record.design_md)
    assert_canonical_template_markers(record)
    pack = _training_pack(record.meta.get("pack_id", "openui"))
    if pack.pack_id != "openui":
        pack.require("training_validator")(record)
        return
    assert_symbol_only_output(record.openui, output_kind=record.target_kind)
    assert_role_safe_output(record.openui, output_kind=record.target_kind)


def assert_training_batch(records, pack_id: str) -> None:
    pack = _training_pack(pack_id)
    for record in records:
        if record.meta.get("pack_id", "openui") != pack.pack_id:
            raise ValueError("training record pack does not match model pack")
        assert_training_record(record)
        if pack.pack_id != "openui":
            canonicalize = pack.require("canonicalize")
            if any(canonicalize(target.text) != target.text for target in record.output_targets):
                raise ValueError("model training requires canonical pack targets")


def require_model_pack(pack_id: str, active_pack_id: str) -> None:
    from slm_training.dsl.pack import get_pack

    if _training_pack(pack_id).pack_id != get_pack(active_pack_id).pack_id:
        raise ValueError("model pack does not match active decode grammar")


def _training_pack(pack_id):
    from slm_training.dsl.pack import get_pack

    if not isinstance(pack_id, str) or not pack_id.strip():
        raise ValueError("training pack authority must be explicit and nonempty")
    pack = get_pack(pack_id)
    if pack.pack_id != pack_id:
        raise ValueError("training pack authority must use its canonical id")
    return pack
