"""SLM-292 (AP-010): fixture-scale wiring smoke for the semantic-contrast objective.

**Claim class: fixture wiring evidence, not a promotion claim.** This trains a
tiny TwoTowerModel (d_model in the tens, a handful of records) for a small
number of steps -- nowhere near the acceptance bar (meaning-v2 +0.05 absolute,
or binder/reference F1 +0.10, with paired CI excluding zero, replicated across
>=3 seeds on full held-out suites). It exists to prove the objective is wired
correctly end-to-end and logs everything the issue requires: loss weight,
margin, mutation sampling, and positive/negative distances -- see
``docs/design/iter-slm292-semantic-contrast-smoke-<date>.md`` for the honest
disposition and the full-scale run this smoke recommends as follow-up.

Matched-token/update control: the ``control`` and ``treatment`` arms share
identical initialization (same seed before construction), identical
architecture, and identical training records/batching/steps; only
``semantic_contrast_loss_weight`` differs (0.0 vs > 0.0).

**Scope note:** this smoke does NOT run constrained decode / generation.
``TwoTowerModel.generate`` with the required ``grammar_constrained=True``
(unconstrained generation is explicitly refused -- see
``slm_training.models.grammar.require_constrained_generation``) takes >60s for
a single call even on this toy architecture, which would blow the repo's
``MAX_RUN_MINUTES`` hard run cap (``src/slm_training/levers.py``) for a
fixture smoke. Raw/constrained/repaired decode-outcome comparisons are
explicit follow-up for the full-scale promotion run (via
``scripts.evaluate_model``), not measured here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "Slm292Arm",
    "Slm292StepRecord",
    "Slm292MutationEffect",
    "Slm292Report",
    "run_smoke",
    "render_markdown",
]

MATRIX_VERSION = "ap-010-v1"
MATRIX_SET = "slm292_semantic_contrast_smoke"

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    'hero = Card([hero_title, hero_body])'
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'

_TRAIN_RECORDS: tuple[ExampleRecord, ...] = (
    ExampleRecord(id="hero", prompt="Hero", openui=HERO, split="train"),
    ExampleRecord(id="cta", prompt="Call to action", openui=CTA, split="train"),
)


@dataclass(frozen=True)
class Slm292Arm:
    """Matched-control training arm (only the objective weight differs)."""

    arm_id: str
    semantic_contrast_loss_weight: float
    seed: int
    steps: int
    d_model: int
    n_heads: int
    context_layers: int
    denoiser_layers: int

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm292StepRecord:
    """Per-step diagnostics -- required log fields per the AP-010 acceptance bar."""

    arm_id: str
    step: int
    total_loss: float
    semantic_contrast_loss: float | None
    semantic_contrast_loss_weight: float | None
    semantic_contrast_objective: str | None
    semantic_contrast_margin: float | None
    semantic_contrast_temperature: float | None
    semantic_contrast_pairs: int | None
    semantic_contrast_sampling_seed: int | None
    semantic_contrast_family_counts: dict[str, int] | None
    semantic_contrast_transform_counts: dict[str, int] | None
    semantic_contrast_positive_distance_mean: float | None
    semantic_contrast_negative_distance_mean: float | None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm292MutationEffect:
    """Per-mutation-family aggregate over every logged treatment-arm step."""

    family: str
    n_samples: int
    mean_positive_distance: float
    mean_negative_distance: float
    mean_margin: float

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass
class Slm292Report:
    schema: str
    claim_class: str
    disclosure: str
    control_arm: Slm292Arm
    treatment_arm: Slm292Arm
    steps: list[Slm292StepRecord] = field(default_factory=list)
    mutation_effects: list[Slm292MutationEffect] = field(default_factory=list)
    control_final_loss: float | None = None
    treatment_final_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_class": self.claim_class,
            "disclosure": self.disclosure,
            "control_arm": self.control_arm.to_dict(),
            "treatment_arm": self.treatment_arm.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "mutation_effects": [m.to_dict() for m in self.mutation_effects],
            "control_final_loss": self.control_final_loss,
            "treatment_final_loss": self.treatment_final_loss,
        }


def _build_model(
    *,
    seed: int,
    semantic_contrast_loss_weight: float,
    d_model: int,
    n_heads: int,
    context_layers: int,
    denoiser_layers: int,
    batch_pairs: int,
    margin: float,
    corpus_path: str | None,
    sampling_seed: int,
) -> TwoTowerModel:
    import torch

    torch.manual_seed(seed)
    return TwoTowerModel.from_records(
        list(_TRAIN_RECORDS),
        config=TwoTowerConfig(
            d_model=d_model,
            n_heads=n_heads,
            context_layers=context_layers,
            denoiser_layers=denoiser_layers,
            output_tokenizer="lexer",
            semantic_contrast_loss_weight=semantic_contrast_loss_weight,
            semantic_contrast_corpus_path=corpus_path,
            semantic_contrast_batch_pairs=batch_pairs,
            semantic_contrast_margin=margin,
            semantic_contrast_sampling_seed=sampling_seed,
        ),
        device="cpu",
    )


def _train_arm(
    model: TwoTowerModel, *, arm_id: str, steps: int, lr: float
) -> tuple[list[Slm292StepRecord], float]:
    import torch

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    records: list[Slm292StepRecord] = []
    last_loss = float("nan")
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = model.training_loss(list(_TRAIN_RECORDS))
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        metrics = model.last_training_metrics
        records.append(
            Slm292StepRecord(
                arm_id=arm_id,
                step=step,
                total_loss=last_loss,
                semantic_contrast_loss=metrics.get("semantic_contrast_loss"),
                semantic_contrast_loss_weight=metrics.get(
                    "semantic_contrast_loss_weight"
                ),
                semantic_contrast_objective=metrics.get(
                    "semantic_contrast_objective"
                ),
                semantic_contrast_margin=metrics.get("semantic_contrast_margin"),
                semantic_contrast_temperature=metrics.get(
                    "semantic_contrast_temperature"
                ),
                semantic_contrast_pairs=metrics.get("semantic_contrast_pairs"),
                semantic_contrast_sampling_seed=metrics.get(
                    "semantic_contrast_sampling_seed"
                ),
                semantic_contrast_family_counts=metrics.get(
                    "semantic_contrast_family_counts"
                ),
                semantic_contrast_transform_counts=metrics.get(
                    "semantic_contrast_transform_counts"
                ),
                semantic_contrast_positive_distance_mean=metrics.get(
                    "semantic_contrast_positive_distance_mean"
                ),
                semantic_contrast_negative_distance_mean=metrics.get(
                    "semantic_contrast_negative_distance_mean"
                ),
            )
        )
    return records, last_loss


def _mutation_effects(
    treatment_steps: list[Slm292StepRecord],
) -> list[Slm292MutationEffect]:
    """Per-family aggregate distances/margin, weighted by each step's sampled family mix."""
    by_family: dict[str, list[tuple[float, float]]] = {}
    for row in treatment_steps:
        counts = row.semantic_contrast_family_counts or {}
        pos = row.semantic_contrast_positive_distance_mean
        neg = row.semantic_contrast_negative_distance_mean
        if pos is None or neg is None:
            continue
        for family, n in counts.items():
            by_family.setdefault(family, []).extend([(pos, neg)] * int(n))
    effects: list[Slm292MutationEffect] = []
    for family in sorted(by_family):
        pairs = by_family[family]
        mean_pos = sum(p for p, _ in pairs) / len(pairs)
        mean_neg = sum(n for _, n in pairs) / len(pairs)
        effects.append(
            Slm292MutationEffect(
                family=family,
                n_samples=len(pairs),
                mean_positive_distance=mean_pos,
                mean_negative_distance=mean_neg,
                mean_margin=mean_neg - mean_pos,
            )
        )
    return effects


def run_smoke(
    *,
    seed: int = 123,
    steps: int = 20,
    lr: float = 3e-3,
    d_model: int = 32,
    n_heads: int = 4,
    context_layers: int = 1,
    denoiser_layers: int = 1,
    batch_pairs: int = 6,
    margin: float = 0.2,
    treatment_weight: float = 1.0,
    corpus_path: str | None = None,
    sampling_seed: int = 0,
) -> Slm292Report:
    """Run the matched control/treatment smoke and collect required evidence."""
    common = dict(
        seed=seed,
        d_model=d_model,
        n_heads=n_heads,
        context_layers=context_layers,
        denoiser_layers=denoiser_layers,
        batch_pairs=batch_pairs,
        margin=margin,
        corpus_path=corpus_path,
        sampling_seed=sampling_seed,
    )
    control_model = _build_model(semantic_contrast_loss_weight=0.0, **common)
    treatment_model = _build_model(
        semantic_contrast_loss_weight=treatment_weight, **common
    )

    control_steps, control_final = _train_arm(
        control_model, arm_id="control", steps=steps, lr=lr
    )
    treatment_steps, treatment_final = _train_arm(
        treatment_model, arm_id="treatment", steps=steps, lr=lr
    )

    mutation_effects = _mutation_effects(treatment_steps)

    control_arm = Slm292Arm(
        arm_id="control",
        semantic_contrast_loss_weight=0.0,
        seed=seed,
        steps=steps,
        d_model=d_model,
        n_heads=n_heads,
        context_layers=context_layers,
        denoiser_layers=denoiser_layers,
    )
    treatment_arm = Slm292Arm(
        arm_id="treatment",
        semantic_contrast_loss_weight=treatment_weight,
        seed=seed,
        steps=steps,
        d_model=d_model,
        n_heads=n_heads,
        context_layers=context_layers,
        denoiser_layers=denoiser_layers,
    )
    return Slm292Report(
        schema="slm292_semantic_contrast_smoke/v1",
        claim_class="fixture_wiring",
        disclosure=(
            "Fixture-scale wiring evidence only (2 training records, "
            f"d_model={d_model}, {steps} steps). This is NOT the AP-010 "
            "promotion claim: meaning-v2 +0.05 absolute (or binder/reference "
            "F1 +0.10) with paired CI excluding zero, replicated across "
            ">=3 seeds on the full held-out suites, is unmeasured here. "
            "Constrained/raw/repaired decode-outcome comparisons are not run "
            "in this smoke at all (a single constrained TwoTowerModel.generate "
            "call takes >60s even on this toy architecture and would blow the "
            "repo MAX_RUN_MINUTES hard run cap) -- explicit follow-up via "
            "scripts.evaluate_model."
        ),
        control_arm=control_arm,
        treatment_arm=treatment_arm,
        steps=control_steps + treatment_steps,
        mutation_effects=mutation_effects,
        control_final_loss=control_final,
        treatment_final_loss=treatment_final,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SLM-292 (AP-010): semantic-contrast objective -- fixture-scale wiring smoke",
        "",
        f"- claim_class: `{payload['claim_class']}` (NOT a promotion claim)",
        f"- {payload['disclosure']}",
        "",
        "## Matched control/treatment arms",
        "",
        "| arm | semantic_contrast_loss_weight | seed | steps | d_model | final total loss |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| control | {payload['control_arm']['semantic_contrast_loss_weight']} | "
            f"{payload['control_arm']['seed']} | {payload['control_arm']['steps']} | "
            f"{payload['control_arm']['d_model']} | {payload['control_final_loss']:.4f} |"
        ),
        (
            f"| treatment | {payload['treatment_arm']['semantic_contrast_loss_weight']} | "
            f"{payload['treatment_arm']['seed']} | {payload['treatment_arm']['steps']} | "
            f"{payload['treatment_arm']['d_model']} | {payload['treatment_final_loss']:.4f} |"
        ),
        "",
        "## Per-mutation-family effect (treatment arm, aggregated over all logged steps)",
        "",
        "| family | n_samples | mean positive distance | mean negative distance | mean margin (neg - pos) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for effect in payload["mutation_effects"]:
        lines.append(
            f"| {effect['family']} | {effect['n_samples']} | "
            f"{effect['mean_positive_distance']:.4f} | "
            f"{effect['mean_negative_distance']:.4f} | {effect['mean_margin']:.4f} |"
        )
    treatment_rows = [s for s in payload["steps"] if s["arm_id"] == "treatment"]
    if treatment_rows:
        first, last = treatment_rows[0], treatment_rows[-1]
        lines += [
            "",
            "## Required logged fields (first vs. last treatment step)",
            "",
            "| field | first step | last step |",
            "| --- | --- | --- |",
            f"| semantic_contrast_loss_weight | {first['semantic_contrast_loss_weight']} | {last['semantic_contrast_loss_weight']} |",
            f"| semantic_contrast_objective | {first['semantic_contrast_objective']} | {last['semantic_contrast_objective']} |",
            f"| semantic_contrast_margin | {first['semantic_contrast_margin']} | {last['semantic_contrast_margin']} |",
            f"| semantic_contrast_pairs | {first['semantic_contrast_pairs']} | {last['semantic_contrast_pairs']} |",
            f"| semantic_contrast_sampling_seed | {first['semantic_contrast_sampling_seed']} | {last['semantic_contrast_sampling_seed']} |",
            f"| semantic_contrast_family_counts | `{first['semantic_contrast_family_counts']}` | `{last['semantic_contrast_family_counts']}` |",
            f"| semantic_contrast_positive_distance_mean | {first['semantic_contrast_positive_distance_mean']:.4f} | {last['semantic_contrast_positive_distance_mean']:.4f} |",
            f"| semantic_contrast_negative_distance_mean | {first['semantic_contrast_negative_distance_mean']:.4f} | {last['semantic_contrast_negative_distance_mean']:.4f} |",
        ]
    lines += [
        "",
        "## Follow-up (not run in this session)",
        "",
        "- Full >=3-seed promotion campaign on the real training corpus + frozen "
        "held-out suites, gated on meaning-v2 +0.05 absolute or binder/reference "
        "F1 +0.10 with paired CI excluding zero, and syntax/contract validity "
        "regression <=0.01.",
        "- Raw/constrained/repaired decode-outcome comparisons (skipped entirely "
        "here -- see disclosure).",
        "",
    ]
    return "\n".join(lines)
