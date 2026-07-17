"""G2 (SLM-35): recipe-evolution harness — population search over recipes.

ShinkaEvolve/AlphaEvolve-**Adapted**: only the population/evaluator pattern
transfers. Genes are training-recipe knobs (corruption schedule, decode
policy, loss weights) on a frozen fixture geometry; the evaluator is the
existing `train()` + `evaluate_suites()` stack; selection is gated by the
UNALTERED honest ship gates. Constraints inherited from
`docs/design/autoresearch-autotraining.md`: bounded self-improvement by
accumulated evidence only — this harness never edits implementations, frozen
evaluations, promotion policy, or ship gates, and any future RL leg must sit
behind `autoresearch.rl_gate.assert_rl_ready` (no RL path exists here).

Selection rule (never weakened): candidates that pass
`evaluate_ship_gates(suites)` with the default policy strictly outrank every
gate-failing candidate; within a tier, lower `best_weighted_nll` wins. At
fixture scale nothing passes the gates, so the harness can rank but can
never promote — promotion eligibility is reported, not asserted.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates

# Bounded, typed gene space. Categorical genes enumerate their full domain;
# float genes carry (lo, hi) and mutate by resampling uniformly. Geometry is
# deliberately NOT a gene — recipe search holds the model fixed.
CATEGORICAL_GENES: dict[str, tuple[Any, ...]] = {
    "mask_pattern": ("random", "mixed", "diffusion"),
    "parallel_unmask": ("adaptive", "fixed"),
    "gen_steps": (4, 8, 16),
    "remask_span": ("token", "statement"),
}
FLOAT_GENES: dict[str, tuple[float, float]] = {
    "statement_mask_prob": (0.15, 0.6),
    "ltr_loss_weight": (0.0, 1.0),
    "fidelity_loss_weight": (0.0, 1.0),
    "lr": (1e-4, 1e-3),
}
GENE_NAMES: tuple[str, ...] = tuple(CATEGORICAL_GENES) + tuple(FLOAT_GENES)


@dataclass(frozen=True)
class RecipeGene:
    """One recipe: a full assignment over the bounded gene space."""

    mask_pattern: str = "diffusion"
    parallel_unmask: str = "adaptive"
    gen_steps: int = 8
    remask_span: str = "token"
    statement_mask_prob: float = 0.35
    ltr_loss_weight: float = 0.5
    fidelity_loss_weight: float = 0.5
    lr: float = 3e-4

    def __post_init__(self) -> None:
        for name, domain in CATEGORICAL_GENES.items():
            if getattr(self, name) not in domain:
                raise ValueError(f"gene {name}={getattr(self, name)!r} not in {domain}")
        for name, (lo, hi) in FLOAT_GENES.items():
            value = float(getattr(self, name))
            if not lo <= value <= hi:
                raise ValueError(f"gene {name}={value} outside [{lo}, {hi}]")

    def gene_id(self) -> str:
        import hashlib

        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:10]

    def to_model_build_kwargs(self) -> dict[str, Any]:
        return {
            "mask_pattern": self.mask_pattern,
            "parallel_unmask": self.parallel_unmask,
            "gen_steps": int(self.gen_steps),
            "remask_span": self.remask_span,
            "statement_mask_prob": float(self.statement_mask_prob),
            "ltr_loss_weight": float(self.ltr_loss_weight),
            "fidelity_loss_weight": float(self.fidelity_loss_weight),
            "lr": float(self.lr),
        }


def mutate(gene: RecipeGene, rng: random.Random) -> RecipeGene:
    """One-gene mutation, resampled from that gene's domain (never outside)."""
    name = rng.choice(GENE_NAMES)
    if name in CATEGORICAL_GENES:
        choices = [v for v in CATEGORICAL_GENES[name] if v != getattr(gene, name)]
        return replace(gene, **{name: rng.choice(choices)})
    lo, hi = FLOAT_GENES[name]
    return replace(gene, **{name: round(rng.uniform(lo, hi), 6)})


def crossover(a: RecipeGene, b: RecipeGene, rng: random.Random) -> RecipeGene:
    """Uniform crossover over the gene names."""
    picks = {
        name: getattr(a if rng.random() < 0.5 else b, name) for name in GENE_NAMES
    }
    return RecipeGene(**picks)


@dataclass(frozen=True)
class CandidateResult:
    """One evaluated recipe. `gates_pass` uses the default ship policy only."""

    gene: RecipeGene
    fitness: float | None  # best_weighted_nll; lower is better; None = failed
    gates_pass: bool
    gate_failures: tuple[str, ...] = ()
    summary_path: str | None = None
    error: str | None = None


def rank_candidates(results: list[CandidateResult]) -> list[CandidateResult]:
    """Gate-passers strictly first, then by fitness (None last). The gate
    policy is the frozen default — this function has no threshold knob on
    purpose, so selection can never be weakened."""

    def key(result: CandidateResult) -> tuple[int, int, float]:
        missing = result.fitness is None
        return (
            0 if result.gates_pass else 1,
            1 if missing else 0,
            float("inf") if missing else float(result.fitness),
        )

    return sorted(results, key=key)


def gate_check(suites: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Evaluate the frozen default ship gates; missing suites fail closed."""
    outcome = evaluate_ship_gates(dict(suites))
    return bool(outcome.get("pass")), tuple(outcome.get("failures") or ())


@dataclass
class EvolutionConfig:
    campaign_id: str
    population_size: int = 4
    generations: int = 2
    elite_k: int = 2
    seed: int = 0
    output_root: Path = Path("outputs/experiments")


Evaluator = Callable[[RecipeGene, int, int], CandidateResult]


@dataclass
class GenerationRecord:
    index: int
    results: list[CandidateResult] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "results": [
                {
                    "gene": asdict(r.gene),
                    "gene_id": r.gene.gene_id(),
                    "fitness": r.fitness,
                    "gates_pass": r.gates_pass,
                    "gate_failures": list(r.gate_failures),
                    "summary_path": r.summary_path,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def _campaign_dir(config: EvolutionConfig) -> Path:
    return Path(config.output_root) / config.campaign_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_evolution(
    config: EvolutionConfig,
    evaluator: Evaluator,
    *,
    initial: RecipeGene | None = None,
) -> dict[str, Any]:
    """Evolve -> evaluate -> gate-checked select, one generation at a time.

    The evaluator owns train+eval (injected so tests run without training).
    Writes `campaign.json`, per-generation artifacts, and `population.json`
    under `<output_root>/<campaign_id>/` — the local tree is the source of
    truth; HF mirroring stays external and optional (`sync_campaign`).
    """
    rng = random.Random(config.seed)
    root = _campaign_dir(config)
    _write_json(
        root / "campaign.json",
        {
            "campaign_id": config.campaign_id,
            "kind": "recipe_evolution",
            "seed": config.seed,
            "population_size": config.population_size,
            "generations": config.generations,
            "elite_k": config.elite_k,
            "gene_space": {
                "categorical": {k: list(v) for k, v in CATEGORICAL_GENES.items()},
                "float": {k: list(v) for k, v in FLOAT_GENES.items()},
            },
        },
    )
    base = initial or RecipeGene()
    population: list[RecipeGene] = [base]
    seen: set[str] = {base.gene_id()}
    while len(population) < config.population_size:
        child = mutate(population[-1], rng)
        if child.gene_id() in seen:
            continue
        seen.add(child.gene_id())
        population.append(child)

    history: list[GenerationRecord] = []
    best: CandidateResult | None = None
    # Each unique recipe is evaluated exactly once (the autoresearch contract
    # forbids repeating a finished knob signature; training is expensive).
    cache: dict[str, CandidateResult] = {}
    for generation in range(config.generations):
        record = GenerationRecord(index=generation)
        for slot, gene in enumerate(population):
            key = gene.gene_id()
            if key not in cache:
                cache[key] = evaluator(gene, generation, slot)
            record.results.append(cache[key])
        ranked = rank_candidates(record.results)
        record.results = ranked
        history.append(record)
        _write_json(
            root / f"generation_{generation:03d}.json", record.to_json()
        )
        if best is None or (
            ranked and rank_candidates([ranked[0], best])[0] is ranked[0]
        ):
            best = ranked[0] if ranked else best
        # Next generation: elites survive; children from crossover+mutation.
        elites = [r.gene for r in ranked[: config.elite_k]]
        population = list(elites)
        while len(population) < config.population_size:
            child = mutate(
                crossover(rng.choice(elites), rng.choice(elites), rng), rng
            )
            if child.gene_id() in seen:
                continue
            seen.add(child.gene_id())
            population.append(child)

    promotable = bool(best and best.gates_pass)
    summary = {
        "campaign_id": config.campaign_id,
        "generations": [g.to_json() for g in history],
        "best": (
            None
            if best is None
            else {
                "gene": asdict(best.gene),
                "gene_id": best.gene.gene_id(),
                "fitness": best.fitness,
                "gates_pass": best.gates_pass,
                "gate_failures": list(best.gate_failures),
            }
        ),
        # Honest: promotion requires the frozen gates to actually pass.
        "promotable": promotable,
        "note": (
            "selection ranks gate-passers first under the unaltered default "
            "ship policy; no candidate passed, so nothing is promotable"
            if not promotable
            else "best candidate passed the frozen ship gates"
        ),
    }
    _write_json(root / "population.json", summary)
    return summary


def train_eval_evaluator(
    *,
    train_dir: Path,
    test_dir: Path,
    run_root: Path,
    campaign_id: str,
    steps: int,
    device: str = "cpu",
    suites: tuple[str, ...] = ("smoke", "held_out"),
    d_model: int = 32,
    n_heads: int = 4,
    context_layers: int = 1,
    denoiser_layers: int = 1,
    seed: int = 0,
) -> Evaluator:
    """The real evaluator: scratch-train the recipe, eval suites, gate-check.

    Fixture geometry is fixed; only the gene fields vary. Evaluating a
    partial suite list is legal for fitness but the gate check fails closed
    on the missing suites — a partial-suite candidate can never be
    promotable, by construction.
    """

    def evaluate(gene: RecipeGene, generation: int, slot: int) -> CandidateResult:
        from slm_training.harnesses.model_build import ModelBuildConfig, train
        from slm_training.harnesses.model_build.eval_runner import evaluate_suites

        run_id = f"{campaign_id}_g{generation}_s{slot}_{gene.gene_id()}"
        cfg = ModelBuildConfig(
            train_dir=Path(train_dir),
            test_dir=Path(test_dir),
            suite="smoke",
            run_root=Path(run_root),
            run_id=run_id,
            steps=steps,
            device=device,
            seed=seed,
            context_backend="scratch",
            output_tokenizer="lexer",
            grammar_ltr_primary=False,
            d_model=d_model,
            n_heads=n_heads,
            context_layers=context_layers,
            denoiser_layers=denoiser_layers,
            **gene.to_model_build_kwargs(),
        )
        try:
            summary = train(cfg)
            board = evaluate_suites(
                cfg, list(suites), checkpoint=Path(summary["checkpoint"])
            )
            gates_pass, failures = gate_check(board.get("suites") or {})
            fitness = summary.get("best_weighted_nll")
            return CandidateResult(
                gene=gene,
                fitness=None if fitness is None else float(fitness),
                gates_pass=gates_pass,
                gate_failures=failures,
                summary_path=str(Path(run_root) / run_id / "train_summary.json"),
            )
        except Exception as exc:  # noqa: BLE001 - candidate failure is data
            return CandidateResult(
                gene=gene,
                fitness=None,
                gates_pass=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    return evaluate
