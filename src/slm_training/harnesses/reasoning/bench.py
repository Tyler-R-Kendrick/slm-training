"""G4 (SLM-36): sketch-of-thought reasoning bench — checkable answers.

Two matched arms, same tiny scratch model class, same corpus, same numeric
scorer (the arith-sketch pack's deterministic evaluator):

- **sketch arm** (PAL/PoT-analog, Gao et al. 2211.10435 / Chen et al.
  2211.12588 — *Adapted*: the reason-in-formal-language + deterministic
  execution split, on a trained tiny model instead of a prompted frozen
  LLM): targets are arith-sketch programs; the emitted trace is validated
  and executed by the pack oracle; an invalid trace scores wrong
  (fail-closed — no repair).
- **direct arm** (no-trace control): targets are the bare numeric answer;
  scored by numeric match.

Sketch-of-Thought (Aytes et al., 2503.05179) positioning: that work is
prompt-level NL-symbolic sketching with a frozen large model; this bench is
the unclaimed trained-tiny-model + externalized-grammar + deterministic
bound-span-expansion variant (refs expand deterministically through the
transformer's resolve pass). Stated boundary: decode is unconstrained
parallel MaskGIT — wiring the incremental grammar engine into non-OpenUI
constrained decode is follow-up work, tracked in the design doc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm_training.dsl.packs.arith_sketch import evaluate_answer
from slm_training.dsl.schema import ExampleRecord

ANSWER_TOLERANCE = 1e-6


def score_sketch_output(text: str, gold: float) -> dict[str, Any]:
    """Fail-closed trace scoring: invalid program == wrong answer."""
    try:
        value = evaluate_answer(text)
        valid = True
    except ValueError as exc:
        return {"valid": False, "correct": False, "error": str(exc)[:120]}
    return {
        "valid": valid,
        "correct": abs(value - gold) <= ANSWER_TOLERANCE,
        "value": value,
    }


def score_direct_output(text: str, gold: float) -> dict[str, Any]:
    token = text.strip().split()[0] if text.strip() else ""
    try:
        value = float(token)
    except ValueError:
        return {"valid": False, "correct": False, "error": "non-numeric"}
    return {
        "valid": True,
        "correct": abs(value - gold) <= ANSWER_TOLERANCE,
        "value": value,
    }


@dataclass
class ReasoningBenchConfig:
    n_train: int = 96
    n_test: int = 24
    seed: int = 0
    steps: int = 60
    device: str = "cpu"
    d_model: int = 64
    n_heads: int = 4
    context_layers: int = 1
    denoiser_layers: int = 2
    output_root: Path = Path("outputs/experiments/reasoning_bench")
    campaign_id: str = "g4_bench"
    model_overrides: dict[str, Any] = field(default_factory=dict)


def _generate_split(config: ReasoningBenchConfig) -> tuple[list[ExampleRecord], list[ExampleRecord]]:
    from slm_training.dsl.packs import get_pack

    pack = get_pack("arith-sketch")
    assert pack.corpus_generator is not None
    train = pack.corpus_generator(config.n_train, config.seed)
    # Disjoint seed stream for test problems; drop accidental prompt dupes.
    test = pack.corpus_generator(config.n_test * 2, config.seed + 10_000)
    train_prompts = {r.prompt for r in train}
    test = [r for r in test if r.prompt not in train_prompts][: config.n_test]
    return train, test


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(value)


def _direct_records(records: list[ExampleRecord]) -> list[ExampleRecord]:
    return [
        replace_record(record, _format_number(float(record.meta["gold_answer"])))
        for record in records
    ]


def replace_record(record: ExampleRecord, target: str) -> ExampleRecord:
    return ExampleRecord(
        id=f"{record.id}_direct",
        prompt=record.prompt,
        openui=target,
        placeholders=[],
        split=record.split,
        source=record.source,
        meta=dict(record.meta),
    )


def _train_and_decode(
    train: list[ExampleRecord],
    test: list[ExampleRecord],
    config: ReasoningBenchConfig,
) -> list[str]:
    import torch

    from slm_training.harnesses.model_build.plugin import GenerationRequest
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    torch.manual_seed(config.seed)
    tt_cfg = TwoTowerConfig(
        # Compositional tokenizer arm: corpus-derived, DSL-agnostic. The
        # lexer arm and grammar gate are OpenUI-hard (stated boundary).
        output_tokenizer="compositional",
        context_backend="scratch",
        grammar_constrained=False,
        d_model=config.d_model,
        n_heads=config.n_heads,
        context_layers=config.context_layers,
        denoiser_layers=config.denoiser_layers,
        max_prompt_len=96,
        max_target_len=96,
        seed=config.seed,
        **config.model_overrides,
    )
    model = TwoTowerModel.from_records(train, config=tt_cfg, device=config.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )
    batch = 4
    for step in range(config.steps):
        start = (step * batch) % max(1, len(train) - batch + 1)
        chunk = train[start : start + batch]
        loss = model.training_loss(chunk)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model.generate_batch_requests(
        [GenerationRequest(prompt=record.prompt) for record in test]
    )


def run_reasoning_bench(config: ReasoningBenchConfig) -> dict[str, Any]:
    """Generate -> train both arms -> decode -> score with one oracle."""
    train, test = _generate_split(config)
    golds = [float(record.meta["gold_answer"]) for record in test]

    sketch_outputs = _train_and_decode(train, test, config)
    direct_outputs = _train_and_decode(_direct_records(train), test, config)

    sketch_scores = [
        score_sketch_output(text, gold)
        for text, gold in zip(sketch_outputs, golds)
    ]
    direct_scores = [
        score_direct_output(text, gold)
        for text, gold in zip(direct_outputs, golds)
    ]

    def _rate(scores: list[dict[str, Any]], key: str) -> float:
        return sum(1 for s in scores if s.get(key)) / max(1, len(scores))

    summary = {
        "campaign_id": config.campaign_id,
        "n_train": len(train),
        "n_test": len(test),
        "steps": config.steps,
        "seed": config.seed,
        "sketch": {
            "answer_accuracy": _rate(sketch_scores, "correct"),
            "trace_validity_rate": _rate(sketch_scores, "valid"),
            "scores": sketch_scores,
            "outputs": sketch_outputs,
        },
        "direct": {
            "answer_accuracy": _rate(direct_scores, "correct"),
            "output_validity_rate": _rate(direct_scores, "valid"),
            "scores": direct_scores,
            "outputs": direct_outputs,
        },
        "golds": golds,
        "note": (
            "fixture-scale matched pair; sketch traces score fail-closed "
            "(invalid program == wrong); decode is unconstrained parallel "
            "MaskGIT in both arms (grammar-engine integration is follow-up)"
        ),
    }
    root = Path(config.output_root) / config.campaign_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "bench_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
