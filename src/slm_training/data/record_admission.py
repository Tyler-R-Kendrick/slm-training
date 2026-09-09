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
    assert_symbol_only_output(record.openui, output_kind=record.target_kind)
    assert_role_safe_output(record.openui, output_kind=record.target_kind)
