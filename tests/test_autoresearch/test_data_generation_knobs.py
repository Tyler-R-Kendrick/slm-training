"""Typed data-generation knobs compile to the canonical builder."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.climb_policy import (
    data_intervention_action,
    data_intervention_indicated,
    load_climb_policy,
)
from slm_training.autoresearch.engine import compile_commands
from slm_training.autoresearch.hillclimb import data_generation_sha256
from slm_training.autoresearch.schemas import (
    CampaignSpec,
    DataGenerationKnobs,
    ExperimentKnobs,
    ExperimentSpec,
)


def _experiment(knobs: ExperimentKnobs) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="data-arm-1",
        campaign_id="camp",
        hypothesis="Data-only unique-root scale changes learnability",
        rationale="synthesis feedback named insufficient_unique_roots",
        expected_effect="unique roots rise",
        falsification_criteria=("unique roots stay flat",),
        stop_conditions=("budget",),
        citations=("docs/design/capability-driven-data-synthesis.md",),
        knobs=knobs,
    )


def test_old_payloads_still_parse_without_data_generation() -> None:
    knobs = ExperimentKnobs(train_version="wf_smoke_v2", steps=2, seed=0)
    assert knobs.data_generation is None


def test_unknown_nested_field_fails() -> None:
    with pytest.raises(ValidationError):
        DataGenerationKnobs(plan_path="plans/a.json", shell="rm -rf /")


def test_absolute_and_parent_paths_rejected() -> None:
    with pytest.raises(ValidationError, match="relative"):
        DataGenerationKnobs(plan_path="/tmp/plan.json")
    with pytest.raises(ValidationError, match=r"\.\."):
        DataGenerationKnobs(plan_path="src/../secret.json")


def test_data_only_rejects_model_knobs() -> None:
    with pytest.raises(ValidationError, match="data_only"):
        ExperimentKnobs(
            data_source="programspec",
            steps=40,
            data_generation=DataGenerationKnobs(data_only=True, unique_root_target=8),
        )


def test_data_only_rejects_seed_and_backend() -> None:
    with pytest.raises(ValidationError, match="data_only"):
        ExperimentKnobs(
            data_source="programspec",
            seed=0,
            data_generation=DataGenerationKnobs(data_only=True, unique_root_target=8),
        )
    with pytest.raises(ValidationError, match="data_only"):
        ExperimentKnobs(
            data_source="programspec",
            context_backend="scratch",
            data_generation=DataGenerationKnobs(data_only=True, unique_root_target=8),
        )


def test_compile_commands_uses_canonical_builder() -> None:
    campaign = CampaignSpec(
        campaign_id="camp",
        objective="test data-only compilation path",
        primary_metric="smoke.parse_rate",
    )
    knobs = ExperimentKnobs(
        data_source="programspec",
        data_generation=DataGenerationKnobs(
            plan_path="src/slm_training/resources/synthesis_plans/corpus/cap0_tiny_v2.json",
            unique_root_target=4,
            generator_seed=17,
            generation_mode="count",
            generator_max_depth=3,
            generator_max_width=4,
            prompt_surface="grammar_schema",
            prompts_per_root=2,
            renderer_families=("openui", "html"),
            counterfactuals_per_root=1,
            retain_trusted_anchors=True,
            data_only=True,
        ),
    )
    commands = compile_commands(campaign, _experiment(knobs))
    assert commands
    build = commands[0]
    assert "scripts.build_train_data" in build
    assert "--synthesis-plan" in build
    assert "--unique-root-target" in build
    assert "--programspec-count" in build
    assert build[build.index("--unique-root-target") + 1] == "4"
    assert build[build.index("--generator-seed") + 1] == "17"
    assert "4" in build
    assert build[build.index("--generation-mode") + 1] == "count"
    assert build[build.index("--generator-max-depth") + 1] == "3"
    assert build[build.index("--generator-max-width") + 1] == "4"
    assert build[build.index("--prompt-surface") + 1] == "grammar_schema"
    assert build[build.index("--prompts-per-root") + 1] == "2"
    assert build[build.index("--renderer-families") + 1] == "openui,html"
    assert build[build.index("--counterfactuals-per-root") + 1] == "1"
    assert "--retain-trusted-anchors" in build

    off = ExperimentKnobs(
        data_source="programspec",
        data_generation=DataGenerationKnobs(
            data_only=True,
            retain_trusted_anchors=False,
        ),
    )
    off_build = compile_commands(campaign, _experiment(off))[0]
    assert "--no-retain-trusted-anchors" in off_build


def test_data_identity_changes_with_root_target() -> None:
    left = data_generation_sha256({"unique_root_target": 4})
    right = data_generation_sha256({"unique_root_target": 8})
    assert left != right


def test_data_intervention_from_blocking_finding() -> None:
    policy = load_climb_policy()
    assert data_intervention_indicated(
        policy,
        feedback=None,
        unique_roots=4,
        blocking_findings=("cue_only_shortcut",),
    )
    assert not data_intervention_indicated(
        policy,
        feedback={"findings": []},
        unique_roots=128,
        blocking_findings=(),
    )
    action = data_intervention_action(policy)
    assert action["kind"] == "rebuild_data"
    assert action["promotion_authorized"] is False
    assert action["train_version"] == "wf_smoke_v2"
    assert action["claim_class"] == "fixture"
    generation = action["data_generation"]
    assert generation["data_only"] is True
    assert generation["unique_root_target"] == action["min_unique_roots"]
    assert generation["plan_path"].endswith("cap0_tiny_v2.json")
