"""Tiny v2 plan build: unique roots ≠ rows, no OpenUI in simplified fallback."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.staged import Capability
from slm_training.harnesses.synthesis_plan import (
    tiny_corpus_generation_policy,
)
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.data.progspec.generate import generate_program_pool
from dataclasses import replace


def test_pool_roots_are_not_inflated_by_prompt_or_repair_counts() -> None:
    policy = tiny_corpus_generation_policy(
        capability=Capability.CAP0_GRAMMAR, unique_root_targets=(3,), seed=17
    )
    policy = replace(
        policy,
        generator=replace(
            policy.generator,
            components=("TextContent", "Button", "Separator"),
            max_depth=2,
            max_width=3,
        ),
        max_attempts=48,
    )
    pool = generate_program_pool(policy)
    assert len(pool.root_ids) == 3
    assert len(set(pool.root_ids)) == 3
    # Prompts-per-root and repairs are not roots.
    assert policy.prompts.prompts_per_root >= 2
    assert policy.derivatives.repairs_per_root >= 1


def test_v2_plan_loader_does_not_require_staged_graph(tmp_path: Path) -> None:
    plan = Path("src/slm_training/resources/synthesis_plans/corpus/cap0_tiny_v2.json")
    result = build_train_data(
        TrainDataConfig(
            profile="permissive",
            source="programspec",
            synthesis_plan_path=plan,
            output_root=tmp_path / "out",
            version="tiny",
            include_language_contract=False,
            include_scope_corpus=False,
            include_frontier_artifacts=False,
            include_edit_derivatives=False,
            repairs_per_program=0,
            require_design_md=False,
            programspec_count=4,
            programspec_seed=17,
        )
    )
    manifest = result["manifest"]
    assert manifest["corpus_generation"]["unique_root_targets"] == [4]
    assert result["quality_report"]["claim_class"] == "fixture_wiring"
    assert result["quality_report"]["capability_certificate"] is False
    assert result["quality_report"]["unique_roots"]["prompt_provider_authoritative"] is False
    assert "pair_quality" in result["quality_report"]
    assert result["synthesis_feedback"]["capability_certificate"] is False
    assert "findings" in result["synthesis_feedback"]
    records = result["stats"]["record_count"]
    assert records >= 1
