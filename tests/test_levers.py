from slm_training.levers import (
    HF_JOB_TIMEOUT,
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
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
    assert catalog["root_reference_arity_loss_weight"][
        "supported_output_tokenizers"
    ] == ["choice"]
    assert catalog["root_reference_arity_loss_weight"]["supported_models"] == [
        "twotower"
    ]
