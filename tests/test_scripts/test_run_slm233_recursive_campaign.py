"""Static and tiny integration contracts for the SLM-233 campaign runner."""

from scripts.run_slm233_recursive_campaign import (
    ARM_SPECS,
    PARAMETER_VIEW_SPECS,
    SEEDS,
    TEST_DEPTHS,
    _accounting,
    _build_model,
    _load_records,
)


def test_primary_matrix_is_five_arms_by_three_paired_seeds() -> None:
    assert [spec.arm for spec in ARM_SPECS] == ["A", "B", "C", "D", "E"]
    assert len(ARM_SPECS) * len(SEEDS) == 15
    assert all(spec.block_evaluations == 4 for spec in ARM_SPECS)
    assert TEST_DEPTHS == (1, 2, 4, 6, 8)


def test_recursive_arms_use_only_authorized_layerscale_configuration() -> None:
    train, _ = _load_records(
        __import__("scripts.run_slm233_recursive_campaign", fromlist=["DEFAULT_DATA"])
        .DEFAULT_DATA
    )
    for spec in ARM_SPECS:
        model = _build_model(spec, SEEDS[0], train)
        if spec.arm == "A":
            assert model.config.recursive_update_mode == "current_v1"
        else:
            assert model.config.recursive_update_mode == "layerscale"
            assert model.config.recursive_empty_f_mode == "zero"
            assert model.config.recursive_norm_mode == "shared"


def test_deep_supervision_weights_are_normalized() -> None:
    for spec in ARM_SPECS:
        assert not spec.depth_weights or sum(spec.depth_weights) == 1.0


def test_objective_only_b_and_c_initialization_is_identical() -> None:
    from scripts.run_slm233_recursive_campaign import DEFAULT_DATA, _model_hash

    train, _ = _load_records(DEFAULT_DATA)
    b = _build_model(ARM_SPECS[1], SEEDS[0], train)
    c = _build_model(ARM_SPECS[2], SEEDS[0], train)
    assert _model_hash(b) == _model_hash(c)


def test_secondary_pair_matches_active_parameters_and_names_byte_residual() -> None:
    from scripts.run_slm233_recursive_campaign import DEFAULT_DATA

    train, _ = _load_records(DEFAULT_DATA)
    left = _build_model(PARAMETER_VIEW_SPECS[0], SEEDS[0], train)
    right = _build_model(PARAMETER_VIEW_SPECS[1], SEEDS[0], train)
    left_accounting = _accounting(left, PARAMETER_VIEW_SPECS[0])
    right_accounting = _accounting(right, PARAMETER_VIEW_SPECS[1])
    assert (
        left_accounting["parameters_trainable"]
        == right_accounting["parameters_trainable"]
    )
    assert (
        left_accounting["parameters_total"]
        != right_accounting["parameters_total"]
    )
    assert (
        left_accounting["checkpoint_state_dict_bytes"]
        != right_accounting["checkpoint_state_dict_bytes"]
    )
