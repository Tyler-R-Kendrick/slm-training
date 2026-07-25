"""Closed Harness DSL framing, pack validation, and symbolic-surface controls."""

from __future__ import annotations

from dataclasses import replace

import pytest

from slm_training.data.contract import RuntimeSymbol
from slm_training.dsl.harness_dsl import (
    HarnessDslError,
    HarnessOperation,
    HarnessPayloadKind,
    HarnessTaskV1,
    harness_grammar_fingerprint,
    parse_harness_task,
    serialize_harness_task,
)

_PAYLOADS = {
    HarnessPayloadKind.DOCUMENT: ("document", "root = Separator()"),
    HarnessPayloadKind.STATEMENT: ("value_statement", "root = Separator()"),
    HarnessPayloadKind.EXPRESSION: ("call", "Separator()"),
    HarnessPayloadKind.LEXICAL: ("BOOL", "true"),
    HarnessPayloadKind.NODE: ("BOOL", "Stack(true)"),
}


@pytest.mark.parametrize("operation", list(HarnessOperation))
@pytest.mark.parametrize("payload_kind", list(HarnessPayloadKind))
def test_every_reserved_operation_and_payload_kind_round_trips(
    operation: HarnessOperation,
    payload_kind: HarnessPayloadKind,
) -> None:
    category, payload = _PAYLOADS[payload_kind]
    task = HarnessTaskV1(
        operation=operation,
        pack_id="openui",
        payload_kind=payload_kind,
        grammar_category=category,
        payload=payload,
        artifact_refs=("a" * 64,),
    )

    source = serialize_harness_task(task)
    parsed = parse_harness_task(source)

    assert parsed == task
    assert source.count(f"OP {operation.value}") == 1
    assert serialize_harness_task(parsed) == source


def test_pack_category_and_artifact_identity_are_exact() -> None:
    task = HarnessTaskV1(
        operation=HarnessOperation.COMPOSE,
        pack_id="openui",
        payload_kind=HarnessPayloadKind.EXPRESSION,
        grammar_category="call",
        payload="Separator()",
        artifact_refs=("0" * 64, "f" * 64),
    )

    parsed = parse_harness_task(serialize_harness_task(task))

    assert parsed.pack_id == "openui"
    assert parsed.grammar_category == "call"
    assert parsed.artifact_refs == ("0" * 64, "f" * 64)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("OP IDENTITY", "OP INVENT"),
        ("TYPE expression", "TYPE prose"),
        ("CATEGORY call", "CATEGORY user_category"),
        ("PAYLOAD_END", "PAYLOAD_END\nTRAILING"),
    ],
)
def test_unknown_or_trailing_framing_fails_closed(old: str, new: str) -> None:
    source = serialize_harness_task(
        HarnessTaskV1(
            operation=HarnessOperation.IDENTITY,
            pack_id="openui",
            payload_kind=HarnessPayloadKind.EXPRESSION,
            grammar_category="call",
            payload="Separator()",
        )
    )

    with pytest.raises(HarnessDslError):
        parse_harness_task(source.replace(old, new))


@pytest.mark.parametrize(
    ("category", "payload"),
    [
        ("STRING", '"write a dashboard"'),
        ("ref", "userIdentifier"),
        ("call", "Separator(]"),
        ("call", "Separator() // explain"),
    ],
)
def test_open_prose_identifiers_comments_and_invalid_fragments_are_rejected(
    category: str, payload: str
) -> None:
    task = HarnessTaskV1(
        operation=HarnessOperation.IDENTITY,
        pack_id="openui",
        payload_kind=(
            HarnessPayloadKind.LEXICAL
            if category == "STRING"
            else HarnessPayloadKind.EXPRESSION
        ),
        grammar_category=category,
        payload=payload,
    )

    with pytest.raises(HarnessDslError):
        serialize_harness_task(task)


def test_fingerprint_is_stable_and_bound_to_versioned_grammar() -> None:
    fingerprint = harness_grammar_fingerprint()
    assert len(fingerprint) == 64
    task = HarnessTaskV1(
        operation=HarnessOperation.IDENTITY,
        pack_id="openui",
        payload_kind=HarnessPayloadKind.EXPRESSION,
        grammar_category="call",
        payload="Separator()",
    )
    assert task.grammar_fingerprint == fingerprint
    with pytest.raises(HarnessDslError, match="fingerprint mismatch"):
        replace(task, grammar_fingerprint="0" * 64)


def test_declared_user_and_external_markers_are_the_only_open_symbols() -> None:
    task = HarnessTaskV1(
        operation=HarnessOperation.COMPOSE,
        pack_id="openui",
        payload_kind=HarnessPayloadKind.EXPRESSION,
        grammar_category="call",
        payload='Stack([hero, TextContent(":hero.title")], "column")',
        runtime_symbols=(
            RuntimeSymbol(surface="hero", role="alpha_binder"),
            RuntimeSymbol(surface=":hero.title", role="external_entity"),
        ),
    )

    parsed = parse_harness_task(serialize_harness_task(task))

    assert parsed.runtime_symbols == task.runtime_symbols
