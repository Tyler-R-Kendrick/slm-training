"""G2 (SLM-35): recipe-evolution invariants — no training involved."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.recipe_evolution import (
    CandidateResult,
    EvolutionConfig,
    RecipeGene,
    crossover,
    gate_check,
    mutate,
    rank_candidates,
    run_evolution,
)
from slm_training.harnesses.model_build import ModelBuildConfig


def test_gene_space_is_bounded_and_deterministic() -> None:
    gene = RecipeGene()
    rng = random.Random(3)
    for _ in range(50):
        gene = mutate(gene, rng)  # __post_init__ re-validates every mutation
    # Same seed -> same trajectory.
    a, b = RecipeGene(), RecipeGene()
    ra, rb = random.Random(9), random.Random(9)
    for _ in range(10):
        a, b = mutate(a, ra), mutate(b, rb)
    assert a == b
    child = crossover(RecipeGene(), gene, random.Random(1))
    assert isinstance(child, RecipeGene)
    with pytest.raises(ValueError):
        RecipeGene(mask_pattern="nope")
    with pytest.raises(ValueError):
        RecipeGene(lr=99.0)


def test_gene_maps_into_model_build_config(tmp_path: Path) -> None:
    gene = RecipeGene(mask_pattern="diffusion", gen_steps=4, lr=2e-4)
    cfg = ModelBuildConfig(
        train_dir=tmp_path,
        run_root=tmp_path,
        run_id="g2_test",
        **gene.to_model_build_kwargs(),
    )
    assert cfg.mask_pattern == "diffusion"
    assert cfg.gen_steps == 4
    assert cfg.lr == pytest.approx(2e-4)


def test_selection_never_prefers_gate_failers_and_fails_closed() -> None:
    """A gate-passing candidate outranks any gate-failing one regardless of
    fitness, and a missing suite fails the gate check (closed)."""
    passer = CandidateResult(gene=RecipeGene(), fitness=9.0, gates_pass=True)
    failer = CandidateResult(
        gene=RecipeGene(gen_steps=4), fitness=1.0, gates_pass=False
    )
    broken = CandidateResult(
        gene=RecipeGene(gen_steps=16), fitness=None, gates_pass=False, error="x"
    )
    ranked = rank_candidates([broken, failer, passer])
    assert ranked[0] is passer
    assert ranked[1] is failer
    assert ranked[2] is broken

    ok, failures = gate_check({"smoke": {"n": 3}})  # metrics + suites missing
    assert not ok
    assert failures


def test_run_evolution_dry_loop_writes_campaign_tree(tmp_path: Path) -> None:
    """The evolve->evaluate->select loop is deterministic under a stub
    evaluator and persists campaign.json + generations + population.json;
    with no gate-passer, promotable is False (honest by construction)."""
    calls: list[str] = []

    def evaluator(gene: RecipeGene, generation: int, slot: int) -> CandidateResult:
        calls.append(f"{generation}:{slot}:{gene.gene_id()}")
        return CandidateResult(
            gene=gene,
            fitness=float(len(calls)),
            gates_pass=False,
            gate_failures=("smoke:meaningful_program_rate",),
        )

    config = EvolutionConfig(
        campaign_id="g2_unit",
        population_size=3,
        generations=2,
        elite_k=2,
        seed=5,
        output_root=tmp_path,
    )
    summary = run_evolution(config, evaluator)
    # 3 initial + 1 fresh child in generation 1 (elites are cached, never
    # re-trained — the no-repeated-knob-signature contract).
    assert len(calls) == 4
    assert summary["promotable"] is False
    assert summary["best"] is not None
    root = tmp_path / "g2_unit"
    assert (root / "campaign.json").exists()
    assert (root / "generation_000.json").exists()
    assert (root / "generation_001.json").exists()
    persisted = json.loads((root / "population.json").read_text())
    assert persisted["campaign_id"] == "g2_unit"
    # Unique genes only: the evaluator never sees the same recipe twice
    # within one run (elites are cached, not re-trained).
    first_run_calls = list(calls)
    ids = [c.split(":")[2] for c in first_run_calls]
    assert len(set(ids)) == len(ids)
    # Determinism: same config + stub -> same best gene id.
    rerun = run_evolution(config, evaluator)
    assert rerun["best"]["gene_id"] == summary["best"]["gene_id"]


def test_gate_passer_becomes_promotable(tmp_path: Path) -> None:
    def evaluator(gene: RecipeGene, generation: int, slot: int) -> CandidateResult:
        return CandidateResult(gene=gene, fitness=1.0, gates_pass=True)

    summary = run_evolution(
        EvolutionConfig(
            campaign_id="g2_pass",
            population_size=2,
            generations=1,
            elite_k=1,
            seed=0,
            output_root=tmp_path,
        ),
        evaluator,
    )
    assert summary["promotable"] is True
