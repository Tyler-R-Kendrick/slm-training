from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.levers import (
    HF_JOB_TIMEOUT,
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    LEVER_COMPANION_REQUIREMENTS,
    LEVER_REQUIREMENTS,
    MAX_RUN_MINUTES,
    MAX_RUN_SECONDS,
    TRAINED_DECODE_REQUIREMENTS,
    lever_catalog,
    missing_lever_companions,
    untrained_decode_levers,
)


def test_run_policy_is_derived_from_one_value() -> None:
    assert MAX_RUN_SECONDS == MAX_RUN_MINUTES * 60
    assert INTERRUPT_AFTER_SECONDS + KILL_GRACE_SECONDS == MAX_RUN_SECONDS
    assert HF_JOB_TIMEOUT == f"{MAX_RUN_MINUTES}m"


def test_catalog_discovers_build_levers_and_context_differences() -> None:
    catalog = lever_catalog()
    assert len(catalog) >= 200
    assert catalog["max_wall_minutes"]["source"] == (
        "slm_training.levers.MAX_RUN_MINUTES"
    )
    assert catalog["semantic_plan_decode_weight"]["category"] == "decode"
    assert catalog["context_backend"]["contexts_diverge"] is True
    assert catalog["context_backend"]["checkpoint_default"] == "scratch"
    assert catalog["evaluation_policy"]["choices"] == [
        "checkpoint_declared",
        "strict_compiler_tree",
    ]
    assert catalog["root_reference_arity_loss_weight"][
        "supported_configurations"
    ] == [
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
        }
    ]
    assert catalog["semantic_plan_decode_weight"][
        "supported_configurations"
    ] == [
        {"model_name": "twotower", "output_tokenizer": "choice"},
        {
            "model_name": "twotower",
            "output_tokenizer": "lexer",
            "compiler_decode_mode": ["restricted", "tree"],
        },
    ]
    assert catalog["semantic_plan_margin_decode_weight"][
        "supported_configurations"
    ] == catalog["semantic_plan_decode_weight"]["supported_configurations"]
    assert catalog["semantic_plan_typed_array_nonempty_margin_decode_weight"][
        "supported_configurations"
    ] == catalog["semantic_plan_decode_weight"]["supported_configurations"]
    assert catalog["schema_role_slot_decode_weight"][
        "supported_configurations"
    ] == catalog["semantic_plan_decode_weight"]["supported_configurations"]
    assert catalog["slot_coverage_close_decode_weight"][
        "supported_configurations"
    ] == catalog["semantic_plan_decode_weight"]["supported_configurations"]
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
        item.name for item in fields(ModelBuildConfig) if item.name.endswith("decode_weight")
    }
    assert decode_weights <= LEVER_REQUIREMENTS.keys()


def test_learned_decode_dependencies_are_discoverable_and_fail_closed() -> None:
    config = ModelBuildConfig(
        train_dir=Path("outputs/data/train"),
        output_tokenizer="lexer",
        compiler_decode_mode="tree",
        root_reference_arity_decode_weight=1.0,
    )

    assert untrained_decode_levers(config) == {
        "root_reference_arity_decode_weight": (
            "root_reference_arity_loss_weight",
        )
    }
    assert lever_catalog()["root_reference_arity_decode_weight"][
        "requires_trained_any"
    ] == ["root_reference_arity_loss_weight"]
    assert set(TRAINED_DECODE_REQUIREMENTS) <= LEVER_REQUIREMENTS.keys()


def test_runtime_companion_dependencies_are_discoverable_and_fail_closed() -> None:
    config = SimpleNamespace(
        output_tokenizer="lexer",
        compiler_decode_mode="tree",
        semantic_role_decode_weight=2.0,
        slot_contract_constrained_decode=False,
        template_fill_decode=False,
        honest_slot_contract=False,
        semantic_role_contract_in_context=False,
    )

    assert missing_lever_companions(config) == {
        "semantic_role_decode_weight": LEVER_COMPANION_REQUIREMENTS[
            "semantic_role_decode_weight"
        ]
    }
    assert lever_catalog()["semantic_role_decode_weight"][
        "requires_companion_configuration"
    ] == [
        {
            "honest_slot_contract": True,
            "semantic_role_contract_in_context": True,
            "slot_contract_constrained_decode": True,
        },
        {
            "honest_slot_contract": True,
            "semantic_role_contract_in_context": True,
            "template_fill_decode": True,
        },
    ]
