"""Unique-root program pools: count, coverage, ladder, shards, no alpha inflation."""

from __future__ import annotations

import pytest

from slm_training.data.progspec.generate import (
    GenerationExhausted,
    GeneratorConfig,
    ProgramGenerator,
    canonical_root_identity,
    generate_program_pool,
    merge_program_pools,
)
from slm_training.dsl.canonicalize import canonicalize
from slm_training.harnesses.staged import Capability
from slm_training.harnesses.synthesis_plan import (
    ExhaustionBehavior,
    GenerationMode,
    tiny_corpus_generation_policy,
)

from dataclasses import replace


SMALL = GeneratorConfig(
    components=("TextContent", "Button", "Separator"),
    max_depth=3,
    max_width=3,
    viewports=("mobile", "desktop"),
    render_states=("empty", "populated"),
)


def _policy(**overrides):
    policy = tiny_corpus_generation_policy(
        capability=Capability.CAP0_GRAMMAR,
        unique_root_targets=(4,),
        seed=17,
    )
    generator = replace(
        policy.generator,
        components=("TextContent", "Button", "Separator"),
        max_depth=3,
        max_width=3,
    )
    return replace(policy, generator=generator, **overrides)


def test_canonical_root_identity_collapses_alpha_renames() -> None:
    left = (
        'root = Stack([title, cta], "column")\n'
        'title = TextContent(":slot_0")\n'
        'cta = Button(":slot_1")'
    )
    right = (
        'root = Stack([heading, action], "column")\n'
        'heading = TextContent(":slot_0")\n'
        'action = Button(":slot_1")'
    )
    assert canonicalize(left) == canonicalize(right)
    assert canonical_root_identity(left) == canonical_root_identity(right)


def test_count_mode_meets_exact_unique_root_target() -> None:
    result = generate_program_pool(_policy(max_attempts=64))
    assert len(result.root_ids) == 4
    assert len(set(result.root_ids)) == 4
    assert len(result.programs) == 4
    assert result.exhausted is False


def test_same_seed_is_byte_identical() -> None:
    first = generate_program_pool(_policy())
    second = generate_program_pool(_policy())
    assert first.root_ids == second.root_ids
    assert first.sha == second.sha
    assert [spec.canonical_openui for spec in first.programs] == [
        spec.canonical_openui for spec in second.programs
    ]


def test_nested_ladder_prefixes_are_exact() -> None:
    policy = _policy(
        mode=GenerationMode.SCALING_LADDER,
        unique_root_targets=(2, 4),
        max_attempts=80,
        ladder=replace(_policy().ladder, require_nested_prefix=True),
    )
    result = generate_program_pool(policy)
    assert result.ladder[2] == result.root_ids[:2]
    assert result.ladder[4] == result.root_ids[:4]
    assert list(result.ladder[2]) == list(result.root_ids[:2])


def test_shard_union_is_merge_order_independent() -> None:
    base = _policy(max_attempts=80, unique_root_targets=(3,))
    shard0 = generate_program_pool(replace(base, shard_count=2, shard_index=0))
    shard1 = generate_program_pool(replace(base, shard_count=2, shard_index=1))
    assert len(shard0.root_ids) <= 2
    assert len(shard1.root_ids) <= 1
    merged_a = merge_program_pools((shard0, shard1))
    merged_b = merge_program_pools((shard1, shard0))
    assert merged_a.root_ids == merged_b.root_ids
    assert merged_a.sha == merged_b.sha
    assert len(merged_a.root_ids) <= 3


def test_exhaustion_fails_closed_with_typed_report() -> None:
    policy = _policy(unique_root_targets=(64,), max_attempts=3)
    with pytest.raises(GenerationExhausted) as exc:
        generate_program_pool(policy)
    assert exc.value.report["requested"] == 64
    assert exc.value.report["local_target"] == 64
    assert exc.value.report["admitted"] < 64


def test_unmet_coverage_minimum_fails_closed() -> None:
    policy = _policy(max_attempts=32)
    policy = replace(
        policy,
        generator=replace(policy.generator, coverage_minimums=(("component", 999),)),
    )
    with pytest.raises(GenerationExhausted) as exc:
        generate_program_pool(policy)
    unmet = exc.value.report["unmet_minima"]
    assert unmet
    assert unmet[0]["axis"] == "component"
    assert unmet[0]["minimum"] == 999
    assert unmet[0]["hits"] < 999


def test_count_and_until_coverage_share_one_ledger() -> None:
    generator = ProgramGenerator(SMALL, seed=11)
    counted = generator.generate(3)
    covered = ProgramGenerator(SMALL, seed=11).generate_until_covered()
    assert set(counted.coverage["axes"]) == set(covered.coverage["axes"])
    pool = generate_program_pool(_policy())
    assert "unique_roots" in pool.coverage
    assert pool.coverage["unique_roots"] == 4


def test_grid_programs_validate() -> None:
    result = generate_program_pool(_policy())
    for spec in result.programs:
        assert spec.canonical_openui
        assert spec.contract_id
        assert spec.id.startswith("program_")


def test_merge_preserves_report_exhaustion_and_extra_axes() -> None:
    base = _policy(max_attempts=32)
    exhausted_pool = generate_program_pool(
        replace(
            base,
            exhaustion=ExhaustionBehavior.REPORT,
            generator=replace(base.generator, coverage_minimums=(("component", 999),)),
        )
    )
    other = generate_program_pool(base)
    assert exhausted_pool.exhausted is True
    assert exhausted_pool.coverage["unmet_minima"]
    assert exhausted_pool.coverage["extra_axes"]
    merged = merge_program_pools((exhausted_pool, other))
    assert merged.exhausted is True
    assert merged.coverage["extra_axes"]
    assert merged.coverage["unmet_minima"]
    assert merged.coverage["merged_shards"] == 2
    assert merged.attempts == exhausted_pool.attempts + other.attempts
