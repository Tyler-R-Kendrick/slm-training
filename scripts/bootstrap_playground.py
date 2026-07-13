#!/usr/bin/env python3
"""Train a small demo checkpoint for the web playground (if missing)."""

from __future__ import annotations

import argparse
from pathlib import Path


DEMO_RECORDS = [
    (
        "Hero card with title and body",
        'root = Stack([hero], "vertical")\nhero = Card(":hero.title", ":hero.body")',
    ),
    (
        "Primary call to action button",
        'root = Stack([cta])\ncta = Button(":cta.label")',
    ),
    (
        "Two feature cards stacked vertically",
        'root = Stack([a, b], "vertical", 8)\n'
        'a = Card(":feat.a.title", ":feat.a.body")\n'
        'b = Card(":feat.b.title", ":feat.b.body")',
    ),
    (
        "Text blurb above a button",
        'root = Stack([copy, cta], "vertical")\n'
        'copy = Text(":copy.line")\n'
        'cta = Button(":cta.label")',
    ),
    (
        "Horizontal row of two buttons",
        'root = Stack([primary, secondary], "horizontal", 4)\n'
        'primary = Button(":actions.primary")\n'
        'secondary = Button(":actions.secondary")',
    ),
    (
        "Pricing card with subscribe button",
        'root = Stack([plan, subscribe], "vertical")\n'
        'plan = Card(":pricing.title", ":pricing.body")\n'
        'subscribe = Button(":pricing.cta")',
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/playground_demo/checkpoints/last.pt"),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        print(f"checkpoint already exists: {args.output}")
        return 0

    import torch

    from slm_training.dsl.parser import validate
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    records = [
        ExampleRecord(id=str(i), prompt=p, openui=o, split="train")
        for i, (p, o) in enumerate(DEMO_RECORDS, start=1)
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=96,
            n_heads=4,
            context_layers=2,
            denoiser_layers=3,
            gen_steps=8,
            grammar_constrained=True,
            seed=0,
        ),
        device="cpu",
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = model.training_loss(records)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step} loss={float(loss.detach()):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    ok = 0
    for record in records:
        pred = model.generate(record.prompt)
        try:
            validate(pred)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"WARN {record.prompt}: {exc}")
    print(f"wrote {args.output} ({ok}/{len(records)} demo prompts parse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
