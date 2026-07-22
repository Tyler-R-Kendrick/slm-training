from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.levers import (
    CHANGED_TEST_WORKERS,
    DEFAULT_CONTEXT_BACKEND,
    DEFAULT_DECODE_TIMEOUT_SECONDS,
    DEFAULT_EVAL_DATA_DIR,
    DEFAULT_OUTPUT_TOKENIZER,
    DEFAULT_TRAIN_DATA_DIR,
    HF_JOB_TIMEOUT,
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    LEVER_REQUIREMENTS,
    MAX_RUN_MINUTES,
    MAX_HARNESS_WALL_MINUTES,
    VERCEL_FUNCTION_INCLUDE_FILES,
    MAX_RUN_SECONDS,
    PROHIBITED_TEMPLATE_SEMANTIC_LEVERS,
    TRAINED_DECODE_REQUIREMENTS,
    lever_catalog,
    missing_lever_companions,
    untrained_decode_levers,
)


def test_run_policy_is_derived_from_one_value() -> None:
    assert MAX_RUN_SECONDS == MAX_RUN_MINUTES * 60
    assert INTERRUPT_AFTER_SECONDS + KILL_GRACE_SECONDS == MAX_RUN_SECONDS
    assert HF_JOB_TIMEOUT == f"{MAX_RUN_MINUTES}m"
    assert (
        MAX_HARNESS_WALL_MINUTES * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
        == INTERRUPT_AFTER_SECONDS
    )
    assert "docs/design/**" in VERCEL_FUNCTION_INCLUDE_FILES
    config = ModelBuildConfig(train_dir=Path("outputs/data/train"))
    assert config.output_tokenizer == DEFAULT_OUTPUT_TOKENIZER == "lexer"
    assert config.decode_timeout_seconds == DEFAULT_DECODE_TIMEOUT_SECONDS == 12.0
    assert lever_catalog()["output_tokenizer"]["default"] == "lexer"
    assert lever_catalog()["decode_timeout_seconds"]["default"] == 12.0
    assert "docs/MODEL_CARD.md" in VERCEL_FUNCTION_INCLUDE_FILES
    assert CHANGED_TEST_WORKERS > 0
    assert DEFAULT_TRAIN_DATA_DIR.is_dir()
    assert DEFAULT_EVAL_DATA_DIR.is_dir()
    assert DEFAULT_CONTEXT_BACKEND == "scratch"
    assert lever_catalog()["default_context_backend"]["default"] == "scratch"


def test_catalog_discovers_build_levers_and_context_differences() -> None:
    catalog = lever_catalog()
    assert len(catalog) >= 200
    assert catalog["max_wall_minutes"]["source"] == (
        "slm_training.levers.MAX_HARNESS_WALL_MINUTES"
    )
    assert catalog["vercel_function_include_files"]["default"] == list(
        VERCEL_FUNCTION_INCLUDE_FILES
    )
    assert catalog["changed_test_workers"] == {
        "category": "run",
        "default": CHANGED_TEST_WORKERS,
        "type": "int",
        "source": "slm_training.levers.CHANGED_TEST_WORKERS",
    }
    assert catalog["default_train_data_dir"]["default"] == str(
        DEFAULT_TRAIN_DATA_DIR
    )
    assert catalog["default_eval_data_dir"]["default"] == str(DEFAULT_EVAL_DATA_DIR)
    assert catalog["semantic_plan_decode_weight"]["category"] == "decode"
    assert catalog["context_backend"]["default"] == DEFAULT_CONTEXT_BACKEND
    assert "contexts_diverge" not in catalog["context_backend"]
    assert catalog["evaluation_policy"]["choices"] == [
        "checkpoint_declared",
        "strict_compiler_tree",
    ]
    assert catalog["evaluation_policy"]["default"] == "strict_compiler_tree"
    assert catalog["evaluation_policy"]["config_default"] == "checkpoint_declared"
    assert catalog["root_reference_arity_loss_weight"]["supported_configurations"] == [
        {"model_name": "twotower", "output_tokenizer": "choice"},
        {
            "model_name": "twotower",
            "output_tokenizer": "lexer",
            "compiler_decode_mode": ["restricted", "tree"],
        },
    ]
    assert catalog["root_reference_arity_decode_weight"][
        "supported_configurations"
    ] == [
        {"model_name": "twotower", "output_tokenizer": "choice"},
        {
            "model_name": "twotower",
            "output_tokenizer": "lexer",
            "compiler_decode_mode": ["restricted", "tree"],
        },
    ]
    assert catalog["semantic_plan_decode_weight"]["supported_configurations"] == [
        {"model_name": "twotower", "output_tokenizer": "choice"},
        {
            "model_name": "twotower",
            "output_tokenizer": "lexer",
            "compiler_decode_mode": ["restricted", "tree"],
        },
    ]
    assert (
        catalog["semantic_plan_margin_decode_weight"]["supported_configurations"]
        == catalog["semantic_plan_decode_weight"]["supported_configurations"]
    )
    assert (
        catalog["semantic_plan_typed_array_nonempty_margin_decode_weight"][
            "supported_configurations"
        ]
        == catalog["semantic_plan_decode_weight"]["supported_configurations"]
    )
    assert catalog["schema_role_slot_decode_weight"]["prohibited"] is True
    assert "supported_configurations" not in catalog["schema_role_slot_decode_weight"]
    assert catalog["slot_coverage_close_decode_weight"]["prohibited"] is True
    assert (
        "supported_configurations" not in catalog["slot_coverage_close_decode_weight"]
    )
    assert catalog["binder_arity_decode_weight"]["supported_configurations"] == [
        {
            "model_name": "twotower",
            "output_tokenizer": "lexer",
            "compiler_decode_mode": ["restricted", "tree"],
        }
    ]
    assert catalog["root_reference_identity_decode_weight"][
        "supported_configurations"
    ] == [{"model_name": "twotower", "output_tokenizer": "choice"}]


def test_every_decode_weight_has_a_capability_requirement() -> None:
    decode_weights = {
        item.name
        for item in fields(ModelBuildConfig)
        if item.name.endswith("decode_weight")
    }
    assert decode_weights <= (
        LEVER_REQUIREMENTS.keys() | PROHIBITED_TEMPLATE_SEMANTIC_LEVERS.keys()
    )


def test_learned_decode_dependencies_are_discoverable_and_fail_closed() -> None:
    config = ModelBuildConfig(
        train_dir=Path("outputs/data/train"),
        output_tokenizer="lexer",
        compiler_decode_mode="tree",
        root_reference_arity_decode_weight=1.0,
    )

    assert untrained_decode_levers(config) == {
        "root_reference_arity_decode_weight": ("root_reference_arity_loss_weight",)
    }
    assert lever_catalog()["root_reference_arity_decode_weight"][
        "requires_trained_any"
    ] == ["root_reference_arity_loss_weight"]
    assert set(TRAINED_DECODE_REQUIREMENTS) <= LEVER_REQUIREMENTS.keys()


def test_prohibited_levers_are_not_advertised_as_supported() -> None:
    config = SimpleNamespace(
        output_tokenizer="lexer",
        compiler_decode_mode="tree",
        semantic_role_decode_weight=2.0,
        slot_contract_in_context=False,
        slot_contract_constrained_decode=False,
        template_fill_decode=False,
        honest_slot_contract=False,
        semantic_role_contract_in_context=False,
    )

    assert missing_lever_companions(config) == {}
    entry = lever_catalog()["semantic_role_decode_weight"]
    assert entry["prohibited"] is True
    assert "supported_configurations" not in entry
    assert "requires_companion_configuration" not in entry


def test_template_semantic_role_recipe_is_prohibited() -> None:
    with pytest.raises(ValueError, match="template markers are opaque"):
        ModelBuildConfig(
            train_dir=Path("outputs/data/train"),
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            semantic_role_decode_weight=2.0,
            slot_contract_constrained_decode=True,
            honest_slot_contract=True,
            semantic_role_contract_in_context=True,
        )

    catalog = lever_catalog()
    assert catalog["semantic_role_decode_weight"]["prohibited"] is True
    assert catalog["template_markers_are_opaque"]["default"] is True


def test_symbol_anonymization_cannot_be_disabled() -> None:
    with pytest.raises(ValueError, match="symbol_anonymization=False is prohibited"):
        ModelBuildConfig(
            train_dir=Path("outputs/data/train"),
            symbol_anonymization=False,
        )
    assert lever_catalog()["symbol_anonymization"]["required"] is True
