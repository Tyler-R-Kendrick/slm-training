"""VAR0-01: VariantContractV1 registry, drift gate, and bucket classification.

Mirrors ``tests/test_dsl/test_ops_vocab.py``'s shape: derived-not-authored
registry round-trip, fingerprint stability, and a monkeypatch drift test.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.variants import (
    FACT_CLASSIFICATION,
    FactBucket,
    VARIANT_CONTRACT_SCHEMA,
    VARIANTS,
    VariantContractError,
    VariantContractV1,
    build_variant_contracts,
    fingerprint,
    load_registry,
    manifest,
)


def test_committed_registry_matches_the_derived_contracts() -> None:
    """The gate future regressions trip over."""
    assert load_registry() == manifest()
    assert load_registry()["fingerprint"] == fingerprint()


def test_build_is_deterministic() -> None:
    assert build_variant_contracts() == VARIANTS
    assert fingerprint() == fingerprint()


def test_all_three_variants_are_registered() -> None:
    ids = {entry.variant_id for entry in VARIANTS}
    assert ids == {"repl_operators", "tree_edit_diffusion", "twotower_prompt_ast"}


def test_every_entry_carries_the_schema() -> None:
    for entry in VARIANTS:
        assert entry.schema == VARIANT_CONTRACT_SCHEMA


def test_fact_classification_is_total() -> None:
    """Every fact named in the VAR0-01 implementation contract has exactly one bucket."""
    expected_facts = {
        "component_inventory",
        "component_property_names",
        "component_property_value_domains",
        "component_arity",
        "op_identity",
        "op_family",
        "op_token_ids",
        "action_encoding",
        "edit_budget",
        "decode_loop",
        "search_strategy",
        "seed_selection",
    }
    assert set(FACT_CLASSIFICATION) == expected_facts
    for fact, bucket in FACT_CLASSIFICATION.items():
        assert isinstance(bucket, FactBucket), f"{fact} has no bucket"
    # Total: every bucket is used at least once, none of the facts double up
    # across buckets (a dict key can only map to one value, so this also
    # guards against a future refactor changing FACT_CLASSIFICATION's shape).
    assert {FactBucket.PACK_OWNED, FactBucket.KERNEL_OWNED, FactBucket.VARIANT_OWNED} == set(
        FACT_CLASSIFICATION.values()
    )


def _fixture_kwargs(**overrides: object) -> dict:
    base = dict(
        variant_id="fixture",
        pack_id="openui",
        action_alphabet_id="fixture.actions",
        action_alphabet_fingerprint="deadbeef",
        kernel_ops=(),
        seed_policy_id="fixture.seed",
        inventory_source="n/a",
        source_path="src/slm_training/dsl/variants.py",
    )
    base.update(overrides)
    return base


def test_rejects_a_kernel_op_id_absent_from_ops_vocab() -> None:
    with pytest.raises(VariantContractError, match="OPS_VOCAB"):
        VariantContractV1(**_fixture_kwargs(kernel_ops=("openui.not_a_real_op",)))


def test_rejects_an_unknown_inventory_source() -> None:
    with pytest.raises(VariantContractError, match="inventory_source"):
        VariantContractV1(**_fixture_kwargs(inventory_source="somewhere_else"))


def test_rejects_a_wrong_schema() -> None:
    with pytest.raises(VariantContractError, match="schema"):
        VariantContractV1(**_fixture_kwargs(schema="variant_contract/v2"))


def test_a_false_pack_authority_claim_is_rejected() -> None:
    """Claiming inventory_source='pack' without importing the pack authority is a lie, not a violation."""
    import slm_training.dsl.variants as module

    fixture = VariantContractV1(
        **_fixture_kwargs(
            inventory_source="pack",
            source_path="slm_training/models/blocks.py",
        )
    )
    with pytest.raises(VariantContractError, match="does not import"):
        module._validate_inventory_source(fixture)


def test_module_constant_claim_is_recorded_but_not_certified() -> None:
    """Legacy contracts can record a violation without falsely claiming pack authority."""
    import slm_training.dsl.variants as module

    fixture = VariantContractV1(
        **_fixture_kwargs(
            inventory_source="module_constant",
            source_path=module._TREE_EDIT_DIFFUSION_SOURCE,
        )
    )
    module._validate_inventory_source(fixture)  # must not raise


def test_imports_module_rejects_a_substring_false_positive() -> None:
    """``slm_training.dsl.packaging`` must never satisfy a
    ``slm_training.dsl.pack`` authority check -- only itself or a true
    submodule (dotted prefix) may."""
    import slm_training.dsl.variants as module

    assert module._names_pack_authority(
        "slm_training.dsl.packaging", "slm_training.dsl.pack"
    ) is False
    assert module._names_pack_authority(
        "slm_training.dsl.packs.types", "slm_training.dsl.pack"
    ) is False
    assert module._names_pack_authority(
        "slm_training.dsl.pack", "slm_training.dsl.pack"
    ) is True
    assert module._names_pack_authority(
        "slm_training.dsl.pack.helpers", "slm_training.dsl.pack"
    ) is True


def test_an_unregistered_pack_id_fails_the_build(monkeypatch) -> None:
    """Every registered variant declares ``pack_id="openui"``; if it stops being
    a known pack, the build must reject it rather than silently trust the claim."""
    import slm_training.dsl.pack as pack_module

    monkeypatch.setattr(pack_module, "list_packs", lambda: ["toy-layout", "arith-sketch"])
    with pytest.raises(VariantContractError, match="not a registered pack"):
        build_variant_contracts()
