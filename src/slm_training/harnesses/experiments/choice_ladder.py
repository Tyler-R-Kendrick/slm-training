"""B3 (SLM-23): capacity ladder — surface-token vs choice-sequence targets.

The direct empirical test of "removing non-lexical symbols lets a smaller
model learn the grammar": matched tiny models per width, one arm trained on
lexer surface targets, one on B1 choice-decision targets, same records /
steps / seed. Produces the E1 bits-per-semantic-decision quantities: the
choice arm's masked NLL is already per-decision; the surface arm's per-token
NLL is renormalized by (target tokens per program / decisions per program)
so both arms are compared in nats per semantic decision.

Fixture instrument: rides the existing TwoTower owner via
``output_tokenizer`` (no parallel trainer); rows decode unconstrained (the
OpenUI DFA speaks surface tokens — constrained choice decode is future B1
work, recorded in docs/design/choice-sequence-codec.md).
"""

from __future__ import annotations

from typing import Any

from slm_training.dsl.production_codec import encode_choices
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.ladder import proportional_depths

TARGETS = ("lexer", "choice")


def _mean_len(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_choice_ladder(
    train_records: list[ExampleRecord],
    heldout_records: list[ExampleRecord],
    *,
    widths: tuple[int, ...] = (16, 32),
    steps: int = 40,
    lr: float = 3e-4,
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    decisions = [
        len(encode_choices(r.openui).tokens) for r in train_records + heldout_records
    ]
    decisions_per_program = _mean_len(decisions)
    rows: list[dict[str, Any]] = []
    for width in widths:
        n_heads, ctx_layers, den_layers = proportional_depths(width)
        for target in TARGETS:
            torch.manual_seed(seed)
            model = TwoTowerModel.from_records(
                train_records,
                config=TwoTowerConfig(
                    d_model=width,
                    n_heads=n_heads,
                    context_layers=ctx_layers,
                    denoiser_layers=den_layers,
                    seed=seed,
                    gen_steps=2,
                    output_tokenizer=target,
                ),
                device="cpu",
            )
            tokens_per_program = _mean_len(
                [len(model.tokenizer.encode(r.openui)) for r in train_records]
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            final_loss = float("nan")
            for _ in range(max(1, steps)):
                optimizer.zero_grad()
                loss = model.training_loss(train_records)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.item())
            model.eval()
            torch.manual_seed(seed)  # same mask draw across arms
            with torch.no_grad():
                heldout = float(model.training_loss(heldout_records).item())
            per_decision = (
                heldout * tokens_per_program / decisions_per_program
                if decisions_per_program
                else float("nan")
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            rows.append(
                {
                    "d_model": width,
                    "target": target,
                    "trainable_params": int(trainable),
                    "vocab_size": int(model.tokenizer.vocab_size),
                    "tokens_per_program": round(tokens_per_program, 2),
                    "decisions_per_program": round(decisions_per_program, 2),
                    "train_loss_final": round(final_loss, 4),
                    "heldout_nll_per_token": round(heldout, 4),
                    "heldout_nll_per_decision": round(per_decision, 4),
                }
            )
    return {
        "study": "b3-choice-ladder",
        "widths": list(widths),
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "train_n": len(train_records),
        "heldout_n": len(heldout_records),
        "rows": rows,
    }


__all__ = ["run_choice_ladder", "TARGETS"]
