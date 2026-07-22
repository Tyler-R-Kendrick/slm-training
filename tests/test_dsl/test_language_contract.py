"""Tests for the OpenUI language-contract identity (F0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl import lang_core
from slm_training.dsl.language_contract import (
    LANG_SPEC,
    OUTPUT_CONTRACT_VERSION,
    LanguageContract,
    OutputContractError,
    assert_symbol_only_output,
    contract_id,
    current_contract,
)
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import load_jsonl


def test_current_contract_pins_installed_langcore() -> None:
    contract = current_contract()
    versions = dict(contract.openui_versions)
    # The installed language layer is the 0.2.x subset (lang-core caps at 0.2.9).
    assert versions["@openuidev/lang-core"] == "0.2.9"
    assert contract.lang_spec == LANG_SPEC == "openui-lang-0.2.x"


def test_contract_id_is_deterministic_16_hex() -> None:
    first = contract_id()
    second = current_contract().contract_id
    assert first == second
    assert len(first) == 16
    int(first, 16)  # raises if not hex


def test_contract_id_changes_when_surface_changes() -> None:
    base = current_contract()
    bumped = LanguageContract(
        lang_spec=base.lang_spec,
        openui_versions=base.openui_versions,
        grammar_sha256=base.grammar_sha256,
        tokenizer_version=base.tokenizer_version + 1,
        dsl_tokenizer_version=base.dsl_tokenizer_version,
    )
    assert bumped.contract_id != base.contract_id


def test_to_dict_round_trips_fields() -> None:
    contract = current_contract()
    data = contract.to_dict()
    assert data["contract_id"] == contract.contract_id
    assert data["lang_spec"] == contract.lang_spec
    assert set(data["openui_versions"]) == {name for name, _ in contract.openui_versions}
    assert data["output_contract_version"] == OUTPUT_CONTRACT_VERSION == 4


def test_symbol_only_output_contract_rejects_free_form_text() -> None:
    assert_symbol_only_output('root = Stack([TextContent(":hero.title")], "column")')
    with pytest.raises(OutputContractError, match="free-form strings"):
        assert_symbol_only_output('root = TextContent("Welcome back")')
    with pytest.raises(OutputContractError, match="free-form strings"):
        assert_symbol_only_output('root = TagBlock(["New Item Alpha"])')


def test_canonical_eval_seeds_are_symbol_only_and_declared() -> None:
    records = load_jsonl(Path("src/slm_training/resources/test_seeds.jsonl"))
    for record in records:
        assert_symbol_only_output(record.openui, output_kind=record.target_kind)
        assert set(extract_placeholders(record.openui)).issubset(record.placeholders), record.id


def test_library_schema_falls_back_to_committed_snapshot(monkeypatch) -> None:
    lang_core._RESULT_CACHE.pop("schema", None)
    monkeypatch.setattr(
        lang_core,
        "_invoke",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    schema = lang_core.library_schema(refresh=True)
    assert len(schema["properties"]) == 54
    assert schema["$defs"]["Card"]["required"] == ["children"]
    assert list(schema["$defs"]["RadioItem"]["properties"]) == [
        "label",
        "description",
        "value",
    ]


def test_bridge_uses_matching_git_common_checkout_dependencies(
    tmp_path, monkeypatch
) -> None:
    worktree = tmp_path / "worktree"
    common = tmp_path / "common"
    work_bridge = worktree / "src/apps/openui_bridge"
    common_bridge = common / "src/apps/openui_bridge"
    for root in (work_bridge, common_bridge):
        root.mkdir(parents=True)
        for name in lang_core._BRIDGE_SOURCES:
            (root / name).write_text(name)
    (common_bridge / "node_modules/@openuidev/lang-core").mkdir(parents=True)
    monkeypatch.delenv("OPENUI_BRIDGE_CLI", raising=False)
    monkeypatch.setattr(lang_core, "REPO_ROOT", worktree)
    monkeypatch.setattr(lang_core, "DEFAULT_BRIDGE_DIR", work_bridge)
    monkeypatch.setattr(
        lang_core, "checkout_roots", lambda root: (root, common)
    )
    lang_core._bridge_dir.cache_clear()

    assert lang_core._bridge_dir() == common_bridge
    assert lang_core._bridge_cli() == common_bridge / "cli.mjs"

    lang_core._bridge_dir.cache_clear()


def test_committed_schema_matches_pinned_library() -> None:
    if not lang_core.bridge_available():
        pytest.skip("OpenUI bridge dependencies are unavailable")
    live = lang_core.library_schema(refresh=True, allow_snapshot=False)
    assert live == lang_core._schema_snapshot()
