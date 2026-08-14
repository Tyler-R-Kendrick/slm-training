import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from slm_training.dsl.language_contract import SymbolicSurfacePolicyV1
from slm_training.harnesses.staged import (
    Capability,
    EvaluationSource,
    SupervisionSource,
)
from slm_training.harnesses.synthesis_plan import (
    CertificateRefV1,
    ComponentRefV1,
    GenerationMode,
    PlanAction,
    PromptSurface,
    SynthesisPlanRegistry,
    SynthesisPlanV1,
    SynthesisPlanV2,
    apply_corpus_generation_overrides,
    load_synthesis_plan,
    migrate_plan_v1_to_v2,
    tiny_corpus_generation_policy,
    validate_capability_prompt_surface,
)
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO_ROOT / "src/slm_training/resources/synthesis_plans/dsh0_cap0_fixture.json"
)
V2_PLAN_PATH = (
    REPO_ROOT / "src/slm_training/resources/synthesis_plans/corpus/cap0_tiny_v2.json"
)
SHA = "a" * 64


def _plan() -> SynthesisPlanV1:
    return SynthesisPlanV1.load(PLAN_PATH)


def _certificate(capability: Capability, *, verified: bool = True) -> CertificateRefV1:
    return CertificateRefV1(
        capability=capability,
        certificate_id=f"{capability.value.lower()}-fixture",
        sha256=SHA,
        verified=verified,
    )


def test_checked_in_plan_is_executable_and_registry_does_not_duplicate_packs() -> None:
    plan = _plan()
    plan.require_executable()
    registry = SynthesisPlanRegistry()
    registry.load_directory(PLAN_PATH.parent)

    assert registry.list() == ("dsh0-cap0-fixture",)
    assert registry.get(plan.plan_id) == plan
    assert not hasattr(registry, "_packs")


def test_json_and_yaml_formatting_have_one_canonical_hash(tmp_path: Path) -> None:
    plan = _plan()
    payload = plan.to_dict()
    yaml_path = tmp_path / "plan.yaml"
    json_path = tmp_path / "plan.json"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    json_path.write_text(
        json.dumps(dict(reversed(list(payload.items()))), indent=7),
        encoding="utf-8",
    )

    loaded_yaml = SynthesisPlanV1.load(yaml_path)
    loaded_json = SynthesisPlanV1.load(json_path)
    assert loaded_yaml == loaded_json == plan
    assert loaded_yaml.sha == loaded_json.sha == plan.sha
    assert json.loads(plan.to_json()) == payload


def test_capability_transitions_and_promotion_eligibility_fail_closed() -> None:
    cap1 = replace(
        _plan(),
        plan_id="cap1",
        capability=Capability.CAP1_SEMANTICS,
        prerequisite=_certificate(Capability.CAP0_GRAMMAR),
    )
    cap1.require_executable()
    cap2 = replace(
        cap1,
        plan_id="cap2",
        capability=Capability.CAP2_TRANSFORM,
        supervision_source=SupervisionSource.SUP_PARAPHRASE,
        prerequisite=_certificate(Capability.CAP1_SEMANTICS),
    )
    cap2.require_executable()
    distill = replace(
        cap2,
        plan_id="distill",
        action=PlanAction.DISTILL,
        supervision_source=SupervisionSource.SUP_DISTILL,
        prerequisite=_certificate(Capability.CAP2_TRANSFORM),
    )
    distill.require_executable()
    trace = replace(
        cap2,
        plan_id="trace",
        action=PlanAction.TRACE_PROMOTE,
        evaluation_source=EvaluationSource.EVAL_TRACE,
        prerequisite=_certificate(Capability.CAP2_TRANSFORM),
    )
    trace.require_executable()

    with pytest.raises(ValueError, match="CAP0_GRAMMAR certificate"):
        replace(cap1, prerequisite=None).require_executable()
    with pytest.raises(ValueError, match="not verified"):
        replace(
            cap1,
            prerequisite=_certificate(Capability.CAP0_GRAMMAR, verified=False),
        ).require_executable()
    with pytest.raises(ValueError, match="CAP2 synthesis requires SUP_PARAPHRASE"):
        replace(
            cap2, supervision_source=SupervisionSource.SUP_COMPILER
        ).require_executable()
    with pytest.raises(ValueError, match="distillation requires SUP_DISTILL"):
        replace(
            distill, supervision_source=SupervisionSource.SUP_PARAPHRASE
        ).require_executable()


def test_unknown_components_versions_and_pack_capabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown generator"):
        replace(
            _plan(),
            generators=(ComponentRefV1("made.up", "v1"),),
        ).require_executable()
    with pytest.raises(ValueError, match="generator version mismatch"):
        replace(
            _plan(),
            generators=(ComponentRefV1("pack.corpus_generator", "v0"),),
        ).require_executable()
    with pytest.raises(ValueError, match="unknown validator"):
        replace(
            _plan(),
            validators=(ComponentRefV1("made.up", "v1"),),
        ).require_executable()
    train_data_version = _plan().generators[0].version
    with pytest.raises(ValueError, match="must include symbolic_surface"):
        replace(
            _plan(),
            validators=(ComponentRefV1("pack.oracle", train_data_version),),
        ).require_executable()
    with pytest.raises(ValueError, match="is not a gate"):
        replace(
            _plan(),
            gate_spec=ComponentRefV1("harness.train_data", train_data_version),
        ).require_executable()

    toy_version = (
        SymbolicSurfacePolicyV1(pack_id="toy-layout").evaluate("").pack_version
    )
    with pytest.raises(ValueError, match="does not provide slot 'corpus_generator'"):
        replace(
            _plan(),
            dsl_pack_id="toy-layout",
            dsl_pack_version=toy_version,
        ).require_executable()


def test_plan_schema_rejects_unknown_keys_and_non_integer_seeds() -> None:
    payload = _plan().to_dict()
    payload["bypass_capability"] = True
    with pytest.raises(ValueError, match="unknown=\\['bypass_capability'\\]"):
        SynthesisPlanV1.from_dict(payload)

    payload = _plan().to_dict()
    payload["seeds"] = [True]
    with pytest.raises(ValueError, match="seed must be an integer"):
        SynthesisPlanV1.from_dict(payload)

    payload = _plan().to_dict()
    payload["destinations"] = [True]
    with pytest.raises(ValueError, match="destination must be a string"):
        SynthesisPlanV1.from_dict(payload)


def test_invalid_plan_fails_before_train_data_producer_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _plan().to_dict()
    payload["generators"] = [{"component_id": "unknown", "version": "v1"}]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    producer_called = False

    def producer(_config):
        nonlocal producer_called
        producer_called = True
        raise AssertionError("producer must not run")

    monkeypatch.setattr(
        "slm_training.harnesses.train_data.pipeline._records_from_fixtures",
        producer,
    )
    with pytest.raises(ValueError, match="unknown generator"):
        build_train_data(
            TrainDataConfig(
                profile="permissive",
                source="fixture",
                synthesis_plan_path=path,
                output_root=tmp_path / "output",
            )
        )
    assert producer_called is False


def test_no_plan_default_preserves_the_existing_config_path() -> None:
    config = TrainDataConfig(profile="permissive", source="fixture")
    assert config.synthesis_plan_path is None
    assert config.curriculum is False


def test_checked_in_v2_tiny_plan_loads_without_touching_v1_registry() -> None:
    plan = load_synthesis_plan(V2_PLAN_PATH)
    assert isinstance(plan, SynthesisPlanV2)
    assert plan.plan_id == "cap0-tiny-corpus-v2"
    assert plan.corpus_generation.requested_unique_roots == 4
    # The v1 registry loader is non-recursive; a v2 file in a subdirectory
    # must not appear as a v1 plan or change the checked-in v1 identity.
    v1 = SynthesisPlanV1.load(PLAN_PATH)
    assert v1.plan_id == "dsh0-cap0-fixture"
    assert "corpus_generation" not in v1.to_dict()


def test_legacy_v1_plan_hash_is_stable_and_shared_loader_preserves_type() -> None:
    plan = _plan()
    loaded = load_synthesis_plan(PLAN_PATH)
    assert isinstance(loaded, SynthesisPlanV1)
    assert loaded == plan
    assert loaded.sha == plan.sha
    assert "corpus_generation" not in plan.to_dict()
    # Adding the optional policy via explicit migration must not rewrite v1 bytes.
    migrated = migrate_plan_v1_to_v2(
        plan, tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR)
    )
    assert isinstance(migrated, SynthesisPlanV2)
    assert migrated.to_v1() == plan
    assert migrated.to_v1().sha == plan.sha
    assert migrated.sha != plan.sha


def test_v2_plan_round_trips_and_rejects_unknown_keys(tmp_path: Path) -> None:
    plan = migrate_plan_v1_to_v2(
        _plan(), tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR)
    )
    payload = plan.to_dict()
    yaml_path = tmp_path / "plan.yaml"
    json_path = tmp_path / "plan.json"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    loaded_yaml = load_synthesis_plan(yaml_path)
    loaded_json = load_synthesis_plan(json_path)
    assert loaded_yaml == loaded_json == plan
    assert loaded_yaml.sha == loaded_json.sha == plan.sha

    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown=\\['unexpected'\\]"):
        SynthesisPlanV2.from_dict(payload)

    policy = plan.corpus_generation.to_dict()
    policy["extra"] = 1
    with pytest.raises(ValueError, match="unknown=\\['extra'\\]"):
        type(plan.corpus_generation).from_dict(policy)


def test_capability_and_prompt_surface_are_not_conflated() -> None:
    validate_capability_prompt_surface(
        Capability.CAP0_GRAMMAR, PromptSurface.GRAMMAR_SCHEMA
    )
    validate_capability_prompt_surface(
        Capability.CAP1_SEMANTICS, PromptSurface.SIMPLIFIED_NL
    )
    validate_capability_prompt_surface(
        Capability.CAP2_TRANSFORM, PromptSurface.TRANSFORMATION_NL
    )
    with pytest.raises(ValueError, match="CAP1_SEMANTICS requires prompt surface"):
        validate_capability_prompt_surface(
            Capability.CAP1_SEMANTICS, PromptSurface.GRAMMAR_SCHEMA
        )
    with pytest.raises(ValueError, match="complex_nl is unavailable"):
        validate_capability_prompt_surface(
            Capability.CAP1_SEMANTICS, PromptSurface.COMPLEX_NL
        )
    with pytest.raises(ValueError, match="requires prompt surface"):
        migrate_plan_v1_to_v2(
            _plan(),
            tiny_corpus_generation_policy(capability=Capability.CAP1_SEMANTICS),
        )


def test_scaling_ladder_can_name_an_8192_rung_without_building_it() -> None:
    policy = tiny_corpus_generation_policy(
        capability=Capability.CAP0_GRAMMAR,
        mode=GenerationMode.SCALING_LADDER,
        unique_root_targets=(128, 512, 2048, 8192),
        seed=17,
    )
    assert policy.requested_unique_roots == 8192
    assert policy.unique_root_targets == (128, 512, 2048, 8192)


def test_invalid_coverage_cue_anchor_and_ladder_configurations_fail() -> None:
    with pytest.raises(ValueError, match="scaling_ladder requires"):
        tiny_corpus_generation_policy(
            capability=Capability.CAP0_GRAMMAR,
            mode=GenerationMode.SCALING_LADDER,
            unique_root_targets=(4,),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        tiny_corpus_generation_policy(
            capability=Capability.CAP0_GRAMMAR,
            mode=GenerationMode.SCALING_LADDER,
            unique_root_targets=(8, 4),
        )
    policy = tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR).to_dict()
    policy["quality"]["max_cue_only_accuracy_lift"] = 1.5
    with pytest.raises(ValueError, match="max_cue_only_accuracy_lift"):
        type(
            tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR)
        ).from_dict(policy)
    policy = tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR).to_dict()
    policy["prompts"]["hard_forbidden_cues"] = ["ui"]
    policy["prompts"]["balanced_generic_cues"] = ["ui"]
    with pytest.raises(ValueError, match="overlap"):
        type(
            tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR).prompts
        ).from_dict(policy["prompts"])
    policy = tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR).to_dict()
    policy["source_anchors"]["promotion_requires_anchors"] = True
    policy["source_anchors"]["trusted_source_families"] = []
    with pytest.raises(ValueError, match="trusted_source_families"):
        type(
            tiny_corpus_generation_policy(
                capability=Capability.CAP0_GRAMMAR
            ).source_anchors
        ).from_dict(policy["source_anchors"])


def test_apply_corpus_generation_overrides_keeps_schema_keys() -> None:
    policy = tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR)
    updated = apply_corpus_generation_overrides(
        policy,
        unique_root_target=8,
        generator_seed=99,
        generator_max_depth=4,
        prompts_per_root=16,
        retain_trusted_anchors=False,
    )
    assert updated.requested_unique_roots == 8
    assert updated.seed == 99
    assert updated.max_attempts >= 8 * 16
    assert updated.generator.max_depth == 4
    assert updated.prompts.prompts_per_root == 16
    assert updated.quality.per_root_exposure_cap == 16
    assert updated.source_anchors.retain_available_trusted_roots is False
    assert set(updated.to_dict()) == set(policy.to_dict())
    assert apply_corpus_generation_overrides(policy) is policy
