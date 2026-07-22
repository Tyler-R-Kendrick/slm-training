from dataclasses import fields

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.levers import (
    HF_JOB_TIMEOUT,
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    LEVER_REQUIREMENTS,
    MAX_RUN_MINUTES,
    MAX_RUN_SECONDS,
    lever_catalog,
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
    ] == [{"model_name": "twotower", "output_tokenizer": "choice"}]
    assert catalog["binder_arity_decode_weight"]["supported_configurations"] == [
        {
            "model_name": "twotower",
            "output_tokenizer": "lexer",
            "compiler_decode_mode": ["restricted", "tree"],
        }
    ]


def test_every_decode_weight_has_a_capability_requirement() -> None:
    decode_weights = {
        item.name for item in fields(ModelBuildConfig) if item.name.endswith("decode_weight")
    }
    assert decode_weights <= LEVER_REQUIREMENTS.keys()
