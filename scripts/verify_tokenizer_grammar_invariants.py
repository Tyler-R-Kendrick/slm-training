"""Certify tokenizer layouts and grammar round-trips used by CI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from slm_training.dsl.analysis.templatize import templatize
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set
from slm_training.dsl.openui_tokens import (
    TOKEN_ID_NAMESPACE_RANGES,
    TOKEN_ID_NAMESPACE_REGISTRY_VERSION,
    logical_token_id,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.choice_tokenizer import ChoiceDecodeState, ChoiceTokenizer
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, SymbolTable

ROOT = Path(__file__).resolve().parents[1]
CERTIFIED_MANIFEST = ROOT / (
    "src/slm_training/resources/data/eval/e763_symbol_only_eval_r2_20260722/"
    "manifest.json"
)
LAYOUT_REGISTRY = ROOT / "src/slm_training/resources/tokenizer_layout_registry.json"


def _layout_fingerprint(tokenizer: object) -> str:
    id_to_token = getattr(tokenizer, "id_to_token")
    id_to_kind = getattr(tokenizer, "id_to_kind", {})
    rows = [
        [token_id, id_to_token[token_id], id_to_kind.get(token_id, "")]
        for token_id in sorted(id_to_token)
    ]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _records() -> list[ExampleRecord]:
    manifest = json.loads(CERTIFIED_MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("immutable"):
        raise AssertionError("tokenizer certificate requires an immutable eval manifest")
    records: list[ExampleRecord] = []
    for path in manifest["suites"].values():
        records.extend(load_jsonl(ROOT / path))
    if len(records) != len(manifest["ids"]):
        raise AssertionError("certified eval manifest record count drifted")
    return records


def _assert_layout_baseline(tokenizers: Iterable[tuple[str, object]]) -> None:
    registry = json.loads(LAYOUT_REGISTRY.read_text(encoding="utf-8"))
    if registry.get("schema") != "tokenizer-layout-registry/v1":
        raise AssertionError("unknown tokenizer layout registry schema")
    if registry.get("namespace_registry_version") != TOKEN_ID_NAMESPACE_REGISTRY_VERSION:
        raise AssertionError("token namespace registry version drifted")
    for name, tokenizer in tokenizers:
        expected = registry["codecs"].get(name)
        actual = {
            "version": getattr(tokenizer, "version"),
            "vocab_size": getattr(tokenizer, "vocab_size"),
            "layout_sha256": _layout_fingerprint(tokenizer),
            "special_ids": {
                special: getattr(tokenizer, f"{special}_id")
                for special in ("pad", "bos", "eos", "mask", "unk")
            },
        }
        if expected != actual:
            raise AssertionError(
                f"{name} tokenizer layout changed; bump its tokenizer version and "
                "update the checked-in layout registry with a migration note"
            )


def _assert_namespace_registry() -> None:
    ranges = list(TOKEN_ID_NAMESPACE_RANGES.values())
    assert all(token_range.start < token_range.stop for token_range in ranges)
    assert all(
        left.stop <= right.start
        for left, right in zip(ranges, ranges[1:])
    )
    assert logical_token_id("control", 0) == TOKEN_ID_NAMESPACE_RANGES["control"].start
    assert logical_token_id("model_native", 0) == TOKEN_ID_NAMESPACE_RANGES["model_native"].start
    try:
        logical_token_id("unknown", 0)
    except ValueError:
        pass
    else:  # pragma: no cover - guard against accidental fail-open behavior
        raise AssertionError("unknown logical token namespace was accepted")


def _native_legal_sets(
    tokenizer: DSLNativeTokenizer, ids: list[int], table: SymbolTable
) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for index in range(1, len(ids)):
        prefix = tokenizer.decode(
            ids[:index], table=table, preserve_trailing_newline=True
        )
        engine = OpenUIIncrementalEngine()
        if not engine.set_prefix(prefix):
            continue
        allowed = allowed_id_set(tokenizer, engine.next_terminals())
        if allowed is not None:
            out.append(tuple(sorted(allowed)))
    return out


def _choice_legal_sets(
    tokenizer: ChoiceTokenizer, ids: list[int], slot_count: int
) -> list[tuple[int, ...]]:
    state = ChoiceDecodeState(tokenizer, slot_count=slot_count)
    body = ids[1:-1]
    out: list[tuple[int, ...]] = []
    for index, token_id in enumerate(body):
        # The chooser's budget includes the final EOS position.
        allowed = state.allowed_ids(len(body) - index + 1)
        if token_id not in allowed:
            raise AssertionError(f"choice token {token_id} is not grammar-legal")
        out.append(tuple(sorted(allowed)))
        if not state.advance_id(token_id):
            raise AssertionError(f"choice token {token_id} did not advance grammar state")
    return out


def certify() -> dict[str, int]:
    native = DSLNativeTokenizer.build()
    choice = ChoiceTokenizer.build()
    _assert_namespace_registry()
    _assert_layout_baseline((("dsl_native", native), ("choice_codec", choice)))

    native_round_trips = 0
    choice_round_trips = 0
    for record in _records():
        target = templatize(record.openui)
        table = SymbolTable.from_placeholders(target.placeholders, max_slots=native.sym_slots)
        native_ids = native.encode(target.source, table=table, add_special=True)
        if native.unk_id in native_ids:
            raise AssertionError(f"native unknown-token leakage: {record.id}")
        native_decoded = native.decode(native_ids, table=table)
        fresh = SymbolTable.from_placeholders(target.placeholders, max_slots=native.sym_slots)
        native_reencoded = native.encode(native_decoded, table=fresh, add_special=True)
        if native_reencoded != native_ids:
            raise AssertionError(f"native id round-trip drift: {record.id}")
        if _native_legal_sets(native, native_ids, table) != _native_legal_sets(
            native, native_reencoded, fresh
        ):
            raise AssertionError(f"native legal-action drift: {record.id}")
        native_round_trips += 1

        choice_table = SymbolTable.from_placeholders(
            target.placeholders, max_slots=choice.sym_slots
        )
        choice_ids = choice.encode(target.source, table=choice_table, add_special=True)
        if choice.unk_id in choice_ids:
            raise AssertionError(f"choice unknown-token leakage: {record.id}")
        choice_decoded = choice.decode(choice_ids, table=choice_table)
        fresh_choice = SymbolTable.from_placeholders(
            target.placeholders, max_slots=choice.sym_slots
        )
        choice_reencoded = choice.encode(
            choice_decoded, table=fresh_choice, add_special=True
        )
        if choice_reencoded != choice_ids:
            raise AssertionError(f"choice id round-trip drift: {record.id}")
        if _choice_legal_sets(choice, choice_ids, len(target.placeholders)) != _choice_legal_sets(
            choice, choice_reencoded, len(target.placeholders)
        ):
            raise AssertionError(f"choice legal-action drift: {record.id}")
        choice_round_trips += 1
    return {"records": native_round_trips, "choice_records": choice_round_trips}


def main() -> None:
    print(json.dumps(certify(), sort_keys=True))


if __name__ == "__main__":
    main()
