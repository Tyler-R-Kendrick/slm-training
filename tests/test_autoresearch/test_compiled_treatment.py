"""Real compiled configuration, model construction, schema and event integration."""

import pytest

from slm_training.autoresearch import engine
from slm_training.autoresearch.preflight.compiled_treatment import (
    begin_attempt,
    lock_driver_designs,
    persist_pair,
    prepare_pair,
)
from slm_training.autoresearch.schemas import ExperimentKnobs, HypothesisMatrix
from slm_training.autoresearch.storage import CampaignStore
from slm_training.dsl.schema import load_jsonl, write_jsonl
from tests.test_autoresearch.test_harness import campaign, experiment
from tests.test_harnesses.model_build.test_full_state_resume import (
    train_dir as fixture_train_dir,
)

train_dir = fixture_train_dir


def test_compiled_local_arm_preserves_real_trial_resume_state(arms, tmp_path):
    """Actual compiler/CLI child; train-stage evidence only, not an evaluation."""
    from pathlib import Path
    from scripts.train_model import resolve_config
    from scripts.autoresearch_command_cursor import ContinuationGrant
    from scripts.autoresearch_continuation import execute_with_continuation
    from slm_training.harness_core.checkpoint_bundle import validate_bundle
    from slm_training.harness_core.checkpoint_exposure import checkpoint_exposure
    from slm_training.harnesses.model_build.full_state import load_full_state

    control, _ = arms
    control = control.model_copy(update={"knobs": control.knobs.model_copy(update={
        "steps": 6, "max_updates_this_invocation": 3, "d_model": 32,
        "n_heads": 4, "context_layers": 1, "denoiser_layers": 1,
    })})
    commands = engine.compile_commands(
        campaign(), control, output_root=tmp_path / "campaigns"
    )
    command = next(cmd for cmd in commands if "scripts.train_model" in cmd)
    config = resolve_config(command[3:])
    assert config.full_state_checkpoint, (
        "campaign compiler discards optimizer/RNG continuation"
    )
    assert not config.sync_checkpoints and config.device == "cpu"
    outcome = execute_with_continuation(
        control, [command], cwd=Path.cwd(), wall_seconds=60,
        grant=ContinuationGrant("test-current-source", 90, 2),
        campaign_manifest_sha256="a" * 64, execute_commands=engine.execute_commands,
        store=CampaignStore(control.campaign_id, tmp_path / "cursor"),
    )
    assert outcome.status == "completed", outcome.model_dump()
    assert len(outcome.stage_telemetry) == 2
    prefix, final = outcome.stage_telemetry
    assert prefix["parsed_output"]["steps"] == 3
    assert prefix["parsed_output"]["stopped_on"] == "invocation_update_budget"
    summary = final["parsed_output"]
    assert summary["continuation_kind"] == "exact_same_environment"
    assert summary["invocation_start_step"] == 3
    checkpoint = Path(summary["checkpoint"])
    directory, manifest = validate_bundle(
        checkpoint.parent.parent.parent, checkpoint.parent.name
    )
    assert manifest["metadata"]["role"] == "trial_cursor"
    assert manifest["resume_state_present"] is True
    state = load_full_state(directory / "last_full_state.pt")
    assert state["step"] == summary["steps"] == config.steps == state["config"]["steps"] == 6
    assert state["optimizer"] and state["torch_rng"] is not None
    assert checkpoint_exposure(checkpoint) == summary["exposure_by_snapshot"]


@pytest.fixture
def arms(tmp_path, train_dir, monkeypatch):
    monkeypatch.setattr(engine, "DEFAULT_TRAIN_DATA_DIR", train_dir)
    monkeypatch.setenv("SLM_DATA_ROOT", str(tmp_path / "data"))
    eval_dir = tmp_path / "data" / "eval" / "eval-fixture"
    suite = eval_dir / "suites" / "smoke"
    suite.mkdir(parents=True)
    # Public identity-plumbing fixture, intentionally not an independent eval.
    write_jsonl(suite / "records.jsonl", load_jsonl(train_dir / "records.jsonl"))
    knobs = ExperimentKnobs(
        steps=2,
        batch_size=2,
        lr=0.003,
        seed=7,
        context_backend="scratch",
        sync_checkpoints=False,
        output_tokenizer="lexer",
        eval_suites="smoke",
        eval_version="eval-fixture",
    )
    control = experiment(experiment_id="control", knobs=knobs)
    candidate = experiment(
        experiment_id="candidate", knobs=knobs.model_copy(update={"lr": 0.001})
    )
    return control, candidate


def _pair(arms, tmp_path, kind="denoising_loss"):
    return prepare_pair(
        campaign(),
        *arms,
        output_root=tmp_path / "campaigns",
        endpoint={"kind": kind, "version": "public_fixture_v1", "units": "nats/token"},
    )


def test_real_compiler_config_parameter_count_and_retry_events(arms, tmp_path):
    pair = _pair(arms, tmp_path)
    assert pair["preflight"]["verdict"] == "pass"
    assert pair["randomness"]["seed"] == 7
    assert (
        pair["randomness"]["loss_mask_seed"]
        == pair["design"]["candidate"]["loss_mask_seed"]
    )
    assert (
        pair["design"]["trainable_params"][0]
        == pair["design"]["trainable_params"][1]
        > 0
    )
    assert pair["design"]["candidate"]["lr"] == 0.001
    assert pair["treatment_ids"][0] != pair["treatment_ids"][1]
    store = CampaignStore(campaign().campaign_id, tmp_path / "campaigns")
    store.initialize(campaign())
    by_id = {
        arm.experiment_id: store.write_artifact("experiments", arm) for arm in arms
    }
    locked = lock_driver_designs(
        store,
        by_id,
        "control",
        ["candidate"],
        endpoint={
            "kind": "denoising_loss",
            "version": "public_fixture_v1",
            "units": "nats/token",
        },
    )
    assert locked["candidate"]["treatment_ids"] == pair["treatment_ids"]
    first_path = persist_pair(store, pair)
    assert persist_pair(store, pair) == first_path
    first = begin_attempt(store, pair, "candidate")
    retry = begin_attempt(store, pair, "candidate")
    assert first["replicate_id"] == retry["replicate_id"]
    assert first["treatment_id"] == retry["treatment_id"]
    assert first["attempt_id"] != retry["attempt_id"]
    assert retry["independent_sample_increment"] == 0
    assert (
        len(
            [
                e
                for e in store.verify_event_chain()
                if e["event_type"] == "experiment_design_locked"
            ]
        )
        == 1
    )


def test_changed_seed_is_replicate_not_new_treatment(arms, tmp_path):
    first = _pair(arms, tmp_path)
    next_arms = tuple(
        arm.model_copy(update={"knobs": arm.knobs.model_copy(update={"seed": 8})})
        for arm in arms
    )
    second = _pair(next_arms, tmp_path)
    assert first["treatment_ids"] == second["treatment_ids"]
    assert first["replicate_id"] != second["replicate_id"]


def test_compiled_identity_charges_total_grant_without_treating_retries_as_a_treatment(arms, tmp_path):
    from slm_training.autoresearch.schemas import CampaignBudget
    from slm_training.harness_core.activity_contract import ResourceGrant

    legacy = CampaignBudget()
    assert legacy.treatment_resources == {}
    assert "treatment_resources" not in legacy.model_dump()
    assert "continuation_grant" not in legacy.model_dump()
    grant = ResourceGrant()

    def compiled(value):
        spec = campaign().model_copy(update={"budget": CampaignBudget(continuation_grant=value)})
        return prepare_pair(spec, *arms, output_root=tmp_path / "campaigns",
                            endpoint={"kind": "denoising_loss", "version": "public_fixture_v1",
                                      "units": "nats/token"})

    original = compiled(grant)
    retries = compiled(grant.model_copy(update={"max_attempts": 6, "interrupt_seconds": 80}))
    larger = compiled(grant.model_copy(update={"total_seconds": grant.total_seconds + 180}))
    assert original["treatment_ids"] == retries["treatment_ids"]
    assert original["replicate_id"] == retries["replicate_id"]
    assert original["treatment_ids"] != larger["treatment_ids"]
    assert original["design"]["bindings"]["resource_contract"]["continuation"] == {
        "logical_seconds": grant.total_seconds, "cpu_slots": grant.cpu_slots,
        "memory_mb": grant.memory_mb,
    }


def test_legacy_experiment_serialization_does_not_invent_identity_fields():
    from slm_training.autoresearch.schemas import ExperimentSpec

    original = experiment().model_dump(mode="json")
    assert {"intervention", "hypothesis_id", "planned_resource_totals"}.isdisjoint(
        original
    )
    assert ExperimentSpec.model_validate(original).model_dump(mode="json") == original


def test_identity_field_composition_preserves_frozen_and_extra_forbidden():
    from slm_training.autoresearch.schemas import ExperimentSpec

    original = experiment()
    assert original.model_config["frozen"] is True
    assert original.model_config["extra"] == "forbid"
    with pytest.raises(ValueError, match="frozen"):
        original.hypothesis_id = "a" * 64
    with pytest.raises(ValueError, match="Extra inputs"):
        ExperimentSpec.model_validate({**original.model_dump(), "waive_gates": True})


def test_noop_and_wrong_endpoint_block_before_training(arms, tmp_path):
    with pytest.raises(ValueError):
        _pair((arms[0], arms[0].model_copy(update={"experiment_id": "noop"})), tmp_path)
    with pytest.raises(ValueError, match="mandatory_preflight"):
        _pair(arms, tmp_path, "latency")


def test_data_duration_require_explicit_declaration(arms, tmp_path):
    candidate = arms[0].model_copy(
        update={"knobs": arms[0].knobs.model_copy(update={"steps": 3, "batch_size": 1})}
    )
    with pytest.raises(ValueError, match="unequal_extra_steps|undeclared duration"):
        _pair((arms[0], candidate), tmp_path)
    declared = candidate.model_copy(
        update={
            "intervention": {
                "kind": "duration",
                "varied_fields": ["steps", "batch_size"],
                "resource_basis": "examples",
            },
            "planned_resource_totals": (3, 3),
        }
    )
    pair = _pair((arms[0], declared), tmp_path)
    assert pair["preflight"]["verdict"] == "pass"


@pytest.mark.parametrize("role", ["confirm", "promotion"])
def test_actual_matrix_pair_has_no_recipe_padding(role):
    from scripts.run_autotrain_continuous import _matrix

    kwargs = {
        "confirm_levers" if role == "confirm" else "promote_levers": {
            "lr": 0.001,
            "steps": 2,
        },
        "confirm_control_levers" if role == "confirm" else "promote_control_levers": {
            "steps": 2
        },
    }
    matrix = _matrix(
        campaign_id="role-fixture",
        evidence_snapshot_id="snapshot",
        cites=["docs/a.md"],
        role_citations={"research": "docs/a.md"},
        train_version="train-fixture",
        eval_version="eval-fixture",
        steps=2,
        cycle=1,
        role="screening" if role == "confirm" else role,
        **kwargs,
    )
    model = HypothesisMatrix.model_validate(matrix)
    assert model.matrix_role == role
    assert len(model.hypotheses) == 2
    assert [row.experiment.knobs.steps for row in model.hypotheses] == [2, 2]
    matrix.pop("matrix_role")
    with pytest.raises(ValueError, match="at least five"):
        HypothesisMatrix.model_validate(matrix)


def test_compiled_geometry_and_cap_preserve_logical_endpoint(arms, tmp_path):
    from scripts.autotrain_arms import capacity_view, size_match_skip_reason
    from scripts.train_model import resolve_config

    control, _ = arms
    fields = dict(d_model=32, n_heads=4, context_layers=1, denoiser_layers=1,
                  steps=6, max_updates_this_invocation=3)
    pair_arms = tuple(arm.model_copy(update={
        "knobs": arm.knobs.model_copy(update=fields),
    }) for arm in arms)
    for arm in pair_arms:
        assert set(arm.knobs.model_dump(exclude_none=True)) <= campaign().allowed_knobs
        command = engine.compile_commands(campaign(), arm, output_root=tmp_path)[0]
        config = resolve_config(command[3:])
        assert (config.d_model, config.n_heads, config.context_layers, config.denoiser_layers) == (32, 4, 1, 1)
        assert config.steps == 6 and config.max_updates_this_invocation == 3
    pair = _pair(pair_arms, tmp_path)
    assert pair["preflight"]["verdict"] == "pass"
    assert pair["design"]["resource_totals"] == (6, 6)
    assert pair["design"]["trainable_params"][0] == pair["design"]["trainable_params"][1]
    assert "max_updates_this_invocation" not in pair["design"]["candidate"]
    uncapped = tuple(arm.model_copy(update={"knobs": arm.knobs.model_copy(
        update={"max_updates_this_invocation": None},
    )}) for arm in pair_arms)
    other = _pair(uncapped, tmp_path)
    assert other["treatment_ids"] == pair["treatment_ids"]
    assert other["design"]["compiled_commands"] != pair["design"]["compiled_commands"]
    assert engine.normalized_knobs_sha256(uncapped[0]) != engine.normalized_knobs_sha256(pair_arms[0])
    bigger = pair_arms[1].model_copy(update={"knobs": pair_arms[1].knobs.model_copy(update={"d_model":64})})
    serialized = [arm.knobs.model_dump(exclude_none=True) for arm in pair_arms]
    geometry = capacity_view(serialized[0])
    assert (geometry.d_model, geometry.n_heads, geometry.context_layers, geometry.denoiser_layers) == (32, 4, 1, 1)
    assert size_match_skip_reason(*serialized) is None
    assert size_match_skip_reason(serialized[0], bigger.knobs.model_dump(exclude_none=True)).startswith("capacity_unmatched:")
    assert all(getattr(control.knobs, key) is None for key in (
        "d_model", "n_heads", "context_layers", "denoiser_layers",
    ))
    with pytest.raises(ValueError, match="capacity|size|param"):
        _pair((pair_arms[0], bigger), tmp_path)
    legacy = resolve_config(engine.compile_commands(campaign(), control, output_root=tmp_path)[0][3:])
    assert legacy.max_updates_this_invocation is None and legacy.d_model == 128
