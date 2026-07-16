"""Tests for the OpenUI language-contract identity (F0)."""

from __future__ import annotations

from slm_training.dsl.language_contract import (
    LANG_SPEC,
    LanguageContract,
    contract_id,
    current_contract,
)


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
    assert data["output_contract_version"] == 1
