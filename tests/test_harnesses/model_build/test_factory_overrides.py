from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.factory import (
    _twotower_config_from_build,
    apply_runtime_overrides,
)


def test_none_decode_overrides_preserve_checkpoint_settings() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(grammar_ltr_primary=True, grammar_ltr_repair=True)
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        grammar_ltr_primary=None,
        grammar_ltr_repair=None,
    )

    apply_runtime_overrides(model, config)

    assert model.config.grammar_ltr_primary is True
    assert model.config.grammar_ltr_repair is True


def test_legacy_unsafe_checkpoint_flags_are_normalized_to_mandatory_policy() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            grammar_constrained=False,
            grammar_ltr_primary=False,
            grammar_finalize_validate=False,
            grammar_fastpath=False,
            grammar_sample_decode=True,
            grammar_uniform_at_unforced=True,
            allow_unconstrained_fallback=True,
        )
    )
    config = ModelBuildConfig(train_dir=Path("."))

    apply_runtime_overrides(model, config)

    assert model._declared_generation_policy["grammar_constrained"] is False
    assert model.config.grammar_constrained is True
    assert model.config.grammar_ltr_primary is True
    assert model.config.grammar_finalize_validate is True
    assert model.config.grammar_fastpath is True
    assert model.config.grammar_sample_decode is False
    assert model.config.grammar_uniform_at_unforced is False
    assert model.config.allow_unconstrained_fallback is False


def test_compiler_decode_mode_activates_primary_ltr_path() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            compiler_decode_mode="off",
            grammar_ltr_primary=False,
        )
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        compiler_decode_mode="tree",
        grammar_ltr_primary=None,
    )

    apply_runtime_overrides(model, config)

    assert model.config.compiler_decode_mode == "tree"
    assert model.config.grammar_ltr_primary is True


def test_compiler_search_overrides_are_typed() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(compiler_search_mode="greedy", compiler_search_width=1)
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        compiler_search_mode="gram",
        compiler_search_width=8,
    )
    apply_runtime_overrides(model, config)
    assert model.config.compiler_search_mode == "gram"
    assert model.config.compiler_search_width == 8


def test_semantic_plan_margin_is_an_explicit_runtime_override() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            output_tokenizer="choice",
            semantic_plan_margin_decode_weight=0.0,
        )
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        output_tokenizer="choice",
        semantic_plan_margin_decode_weight=2.0,
    )

    apply_runtime_overrides(model, config)

    assert model.config.semantic_plan_margin_decode_weight == 2.0


def test_design_md_dropout_overrides_resumed_checkpoint() -> None:
    model = SimpleNamespace(config=SimpleNamespace(design_md_dropout=0.0))
    config = ModelBuildConfig(train_dir=Path("."), design_md_dropout=0.5)

    apply_runtime_overrides(model, config)

    assert model.config.design_md_dropout == 0.5


def test_component_token_weight_overrides_resumed_checkpoint() -> None:
    model = SimpleNamespace(config=SimpleNamespace(component_token_loss_weight=0.0))
    config = ModelBuildConfig(train_dir=Path("."), component_token_loss_weight=1.0)

    apply_runtime_overrides(model, config)

    assert model.config.component_token_loss_weight == 1.0


def test_structure_token_weight_overrides_resumed_checkpoint() -> None:
    model = SimpleNamespace(config=SimpleNamespace(structure_token_loss_weight=0.0))
    config = ModelBuildConfig(train_dir=Path("."), structure_token_loss_weight=1.0)

    apply_runtime_overrides(model, config)

    assert model.config.structure_token_loss_weight == 1.0


def test_typed_family_balance_weight_overrides_resumed_checkpoint() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(typed_family_balance_loss_weight=0.0)
    )
    config = ModelBuildConfig(
        train_dir=Path("."), typed_family_balance_loss_weight=0.25
    )

    apply_runtime_overrides(model, config)

    assert model.config.typed_family_balance_loss_weight == 0.25


def test_explicit_runtime_override_fields_preserve_unrelated_checkpoint_config() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            compiler_search_mode="greedy",
            schema_in_context=True,
            output_tokenizer="lexer",
        )
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        runtime_override_fields=frozenset({"compiler_search_mode"}),
        compiler_search_mode="gram",
        schema_in_context=False,
        output_tokenizer="compositional",
    )

    apply_runtime_overrides(model, config)

    assert model.config.compiler_search_mode == "gram"
    assert model.config.schema_in_context is True
    assert model.config.output_tokenizer == "lexer"


def test_runtime_override_cannot_disable_path_used_by_checkpoint_lever() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_name="twotower",
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            binder_arity_decode_weight=1.0,
        )
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        runtime_override_fields=frozenset({"compiler_decode_mode"}),
        compiler_decode_mode="off",
    )

    with pytest.raises(ValueError, match="runtime overrides has invalid enabled levers"):
        apply_runtime_overrides(model, config)


def test_runtime_override_cannot_activate_untrained_decode_head() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_name="twotower",
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            root_reference_arity_loss_weight=0.0,
            root_reference_arity_decode_weight=0.0,
        )
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        runtime_override_fields=frozenset({"root_reference_arity_decode_weight"}),
        output_tokenizer="lexer",
        compiler_decode_mode="tree",
        root_reference_arity_decode_weight=1.0,
    )

    with pytest.raises(ValueError, match="requires a trained checkpoint objective"):
        apply_runtime_overrides(model, config)


def test_action_alias_overrides_round_trip() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            action_alias_mode="canonical",
            action_description_name_mode="schema",
        )
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        action_alias_mode="fixed",
        action_description_name_mode="alias_aware_description",
    )

    apply_runtime_overrides(model, config)

    assert model.config.action_alias_mode == "fixed"
    assert model.config.action_description_name_mode == "alias_aware_description"


def test_compiler_completion_flags_reach_twotower_config() -> None:
    config = ModelBuildConfig(
        train_dir=Path("."),
        output_tokenizer="lexer",
        compiler_decode_mode="tree",
        required_slot_array_completion=True,
        required_slot_root_completion=True,
        slot_alias_unique_decode=True,
        binder_topology_unique_decode=True,
        compiler_schema_component_types=True,
    )

    runtime = _twotower_config_from_build(config)

    assert runtime.required_slot_array_completion is True
    assert runtime.required_slot_root_completion is True
    assert runtime.slot_alias_unique_decode is True
    assert runtime.binder_topology_unique_decode is True
    assert runtime.compiler_schema_component_types is True


def test_compiler_alignment_kind_filter_reaches_twotower_config() -> None:
    config = ModelBuildConfig(
        train_dir=Path("."),
        compiler_alignment_loss_weight=1.0,
        compiler_alignment_kind_filter="literal-close",
    )

    runtime = _twotower_config_from_build(config)

    assert runtime.compiler_alignment_kind_filter == "literal-close"


def test_model_build_rejects_unknown_compiler_alignment_kind_filter() -> None:
    with pytest.raises(ValueError, match="compiler_alignment_kind_filter"):
        ModelBuildConfig(train_dir=Path("."), compiler_alignment_kind_filter="unknown")


def test_unowned_compiler_auxiliary_fails_closed_at_factory_boundary() -> None:
    config = ModelBuildConfig(
        train_dir=Path("."),
        output_tokenizer="lexer",
        compiler_decode_mode="tree",
        binder_slot_ownership_loss_weight=1.0,
        binder_slot_ownership_decode_weight=1.0,
    )

    with pytest.raises(ValueError, match="no runtime owner is implemented"):
        _twotower_config_from_build(config)
