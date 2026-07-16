"""Output-kind validation contracts that need no lang-core bridge."""

from __future__ import annotations

import pytest

from slm_training.dsl.parser import ParseError, validate_output
from slm_training.dsl.schema import OUTPUT_KINDS, ExampleRecord


def test_typed_node_kind_is_registered() -> None:
    assert "typed_node" in OUTPUT_KINDS


@pytest.mark.parametrize(
    "text",
    ["Boolean(true)", "Number(42)", 'String("hi")', "Null(null)"],
)
def test_typed_node_accepts_constructor_shapes(text: str) -> None:
    assert validate_output(text, "typed_node") == text


@pytest.mark.parametrize(
    "text",
    ["true", "boolean(true)", "Boolean", "Boolean()", "not a typed node"],
)
def test_typed_node_rejects_non_constructor_shapes(text: str) -> None:
    with pytest.raises(ParseError):
        validate_output(text, "typed_node")


def test_example_record_accepts_typed_node_target_kind() -> None:
    record = ExampleRecord(
        id="typed1",
        prompt="Type this OpenUI token: true",
        openui="Boolean(true)",
        target_kind="typed_node",
    )
    assert record.target_kind == "typed_node"


def test_identity_task_token_is_valid() -> None:
    record = ExampleRecord(
        id="ident1",
        prompt="Emit the OpenUI lexical for this input.\n---INPUT---\ntrue",
        openui="true",
        target_kind="lexical",
        target_category="boolean",
        meta={"task": "identity"},
    )
    assert record.meta["task"] == "identity"
