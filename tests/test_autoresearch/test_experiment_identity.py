"""AC-IDENTITY/DATAARM/NOOP: actual domain owners, no copied implementation."""

import copy
import json

import pytest

from tests.casefiles import case_values

from scripts.autotrain_arms import assert_declared_arm_match
from scripts.autotrain_levers import knobs_fingerprint, resolved_treatment_identity
from slm_training.autoresearch.experiment_identity import (
    InterventionContract,
    attempt_identity,
    legacy_identity_reference,
    replicate_identity,
    validate_matrix_role,
)
from slm_training.autoresearch.hillclimb import HillClimbError, assert_warm_start_launch
from slm_training.autoresearch.engine import compile_commands
from slm_training.autoresearch.schemas import ExperimentKnobs
from slm_training.autoresearch.preflight.lever_effects import validate_lever_effects
from slm_training.autoresearch.preflight import run_preflight
from tests.test_autoresearch.test_harness import campaign, experiment


def design():
    control = {
        "steps": 20,
        "batch_size": 2,
        "lr": 0.001,
        "seed": 7,
        "initialize_from": "bundle:a",
        "train_version": "train-a",
    }
    return {
        "control": control,
        "candidate": {**control, "lr": 0.002},
        "intervention": {
            "kind": "mechanism",
            "varied_fields": ["lr"],
            "resource_basis": "updates",
        },
        "resource_totals": [20, 20],
        "trainable_params": [64546, 64546],
        "compiled_commands": compile_commands(
            campaign(), experiment(knobs=ExperimentKnobs(lr=0.002))
        ),
        "endpoint": "denoising_loss",
        "bindings": {
            "architecture": "twotower-small",
            "tokenizer_layout": "layout:a",
            "training_snapshot": "data:a",
            "preprocessing": "prep:a",
            "starting_checkpoint": "bundle:a",
            "starting_checkpoint_role": "warm_start",
            "endpoint": "denoising_loss/v1",
            "resource_contract": {"updates": 20},
        },
    }


def identity(d):
    return resolved_treatment_identity(
        d["candidate"],
        bindings=d["bindings"],
        intervention=InterventionContract.model_validate(d["intervention"]),
    )


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2**31 - 1])
def test_seed_is_replicate_not_treatment(seed):
    d = design()
    original = identity(d)
    d["candidate"].update(seed=seed, attempt_id=f"attempt-{seed}")
    assert identity(d) == original


@pytest.mark.parametrize("key,value", [("steps", 21), ("batch_size", 3), ("lr", 0.003)])
def test_scientific_recipe_change_changes_identity(key, value):
    d = design()
    original = identity(d)
    d["candidate"][key] = value
    assert identity(d) != original


def test_retry_does_not_add_an_independent_unit():
    args = dict(
        hypothesis_id="hyp:a",
        design_digest="plan:a",
        unit_id="root:a",
        seeds={"mask": 1},
        ancestor_digest="bundle:a",
        independence="conditional-on-ancestor",
    )
    unit = replicate_identity(**args)
    treatment = identity(design())
    assert attempt_identity(
        treatment_id=treatment, replicate_id=unit, ordinal=0
    ) != attempt_identity(treatment_id=treatment, replicate_id=unit, ordinal=1)
    assert replicate_identity(**{**args, "seeds": {"mask": 2}}) != unit


def test_legacy_reader_does_not_invent_stronger_semantics():
    assert knobs_fingerprint({"steps": 80, "lr": 0.1}) == knobs_fingerprint(
        {"steps": 81, "lr": 0.1}
    )
    assert (
        legacy_identity_reference("old")["claim"]
        == "unresolved_recipe_not_scientific_identity"
    )
    d = design()
    del d["bindings"]["training_snapshot"]
    with pytest.raises(ValueError, match="incomplete"):
        identity(d)


def test_versioned_identity_golden():
    assert identity(design()) == GOLDEN_IDENTITY


GOLDEN_IDENTITY = "ef2296d0e653659ac06a9248c1f07b4f5f9389f7b7d045df18251658d175fa69"


def test_parameter_consumer_reads_actual_trainer_track(tmp_path):
    from scripts.autotrain_arms import arm_trainable_params

    path = tmp_path / "runs/arm/train_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"track": {"trainable_params": 64546}}))
    assert arm_trainable_params(tmp_path, "arm") == 64546
    path.write_text(json.dumps({"track": {"trainable_params": True}, "n_params": 100}))
    assert arm_trainable_params(tmp_path, "arm") == 0


@pytest.mark.parametrize("bad", [True, -1, 1.5, "6", float("inf"), float("nan")])
def test_completed_count_does_not_coerce_invalid_evidence(bad):
    from scripts.autotrain_arms import arm_completed_n

    assert arm_completed_n({"completed_document_n": bad, "n": 6}) is None


def test_relocated_same_content_checkpoint_is_same_treatment():
    d = design()
    original = identity(d)
    d["candidate"]["initialize_from"] = "another-attempt/same-bundle/last.pt"
    assert identity(d) == original
    d["bindings"]["starting_checkpoint"] = "different-bundle"
    assert identity(d) != original


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_configuration_refused(bad):
    d = design()
    d["candidate"]["lr"] = bad
    with pytest.raises(ValueError):
        identity(d)


@pytest.mark.parametrize(
    "role,count,valid",
    case_values(__file__, "test_matrix_role_needs_no_monitor_padding"),
)
def test_matrix_role_needs_no_monitor_padding(role, count, valid):
    if valid:
        validate_matrix_role(role, count)
    else:
        with pytest.raises(ValueError):
            validate_matrix_role(role, count)


def test_declared_data_change_permitted_without_disabling_ordinary_guard():
    d = design()
    control = d["control"]
    candidate = {**control, "train_version": "train-b"}
    with pytest.raises(HillClimbError, match="unequal_train_data"):
        assert_warm_start_launch(control, candidate)
    contract = InterventionContract(
        kind="data", varied_fields=("train_version",), resource_basis="updates"
    )
    assert_declared_arm_match(
        control,
        candidate,
        contract=contract,
        resource_totals=(20, 20),
        trainable_params=(64546, 64546),
    )
    with pytest.raises(ValueError, match="mismatch"):
        assert_declared_arm_match(
            control,
            {**candidate, "seed": 8},
            contract=contract,
            resource_totals=(20, 20),
            trainable_params=(64546, 64546),
        )


def test_duration_changes_need_explicit_common_resource_basis():
    control = design()["control"]
    candidate = {**control, "steps": 40, "batch_size": 1}
    contract = InterventionContract(
        kind="duration",
        varied_fields=("steps", "batch_size"),
        resource_basis="examples",
    )
    assert_declared_arm_match(
        control,
        candidate,
        contract=contract,
        resource_totals=(40, 40),
        trainable_params=(100, 100),
    )
    with pytest.raises(ValueError, match="unmatched"):
        assert_declared_arm_match(
            control,
            candidate,
            contract=contract,
            resource_totals=(40, 80),
            trainable_params=(100, 100),
        )


@pytest.mark.parametrize(
    "key,value,endpoint,error",
    case_values(__file__, "test_ignored_or_wrong_endpoint_knob_blocks_design"),
)
def test_ignored_or_wrong_endpoint_knob_blocks_design(key, value, endpoint, error):
    control = {**design()["control"], "grammar_draft_window": 1}
    with pytest.raises(ValueError, match=error):
        validate_lever_effects(
            control, {**control, key: value}, varied_fields=(key,), endpoint=endpoint
        )


def test_actual_discovery_calls_design_producer():
    d = design()
    result = {v.check_id: v for v in run_preflight({"treatment_design": d})}[
        "treatment_design"
    ]
    assert result.verdict == "pass", result
    assert result.data["treatment_id"] == identity(d)
    bad = copy.deepcopy(d)
    bad["candidate"]["lr"] = bad["control"]["lr"]
    result = {v.check_id: v for v in run_preflight({"treatment_design": bad})}[
        "treatment_design"
    ]
    assert result.verdict == "block"


def test_preflight_rejects_ignored_compiled_knob():
    d = design()
    d["compiled_commands"] = compile_commands(
        campaign(), experiment(knobs=ExperimentKnobs(lr=0.001))
    )
    result = {v.check_id: v for v in run_preflight({"treatment_design": d})}[
        "treatment_design"
    ]
    assert result.verdict == "block"
    assert "compiled lever" in result.reasons[0]


def test_secondary_objective_knob_with_disabled_owner_is_noop():
    control = {
        **design()["control"],
        "compiler_alignment_margin": 0.0,
        "compiler_alignment_loss_weight": 0.0,
    }
    with pytest.raises(ValueError, match="inactive lever"):
        validate_lever_effects(
            control,
            {**control, "compiler_alignment_margin": 1.0},
            varied_fields=("compiler_alignment_margin",),
            endpoint="denoising_loss",
        )


def test_capacity_is_charged_and_never_automatic_promotion():
    control = design()["control"]
    control["d_model"] = 128
    candidate = {**control, "d_model": 256}
    with pytest.raises(ValueError, match="charging"):
        InterventionContract(
            kind="capacity", varied_fields=("d_model",), resource_basis="updates"
        )
    contract = InterventionContract(
        kind="capacity",
        varied_fields=("d_model",),
        resource_basis="updates",
        capacity_charged=True,
    )
    assert_declared_arm_match(
        control,
        candidate,
        contract=contract,
        resource_totals=(20, 20),
        trainable_params=(100, 200),
    )
    with pytest.raises(ValueError):
        assert_declared_arm_match(
            control,
            candidate,
            contract=contract.model_copy(update={"kind": "mechanism"}),
            resource_totals=(20, 20),
            trainable_params=(100, 200),
        )
