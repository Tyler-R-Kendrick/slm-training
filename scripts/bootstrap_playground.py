#!/usr/bin/env python3
"""Train a small demo checkpoint for the web playground (if missing)."""

from __future__ import annotations

import argparse
from pathlib import Path


DEMO_RECORDS = [
    (
        "Hero card with title and body",
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":hero.title")\n'
        'hero_body = TextContent(":hero.body")\n'
        "hero = Card([hero_title, hero_body])",
    ),
    (
        "Primary call to action button",
        'root = Stack([cta])\ncta = Button(":cta.label")',
    ),
    (
        "Two feature cards stacked vertically",
        'root = Stack([a, b], "column", "m")\n'
        'a_title = TextContent(":feat.a.title")\n'
        'a_body = TextContent(":feat.a.body")\n'
        "a = Card([a_title, a_body])\n"
        'b_title = TextContent(":feat.b.title")\n'
        'b_body = TextContent(":feat.b.body")\n'
        "b = Card([b_title, b_body])",
    ),
    (
        "Text blurb above a button",
        'root = Stack([copy, cta], "column")\n'
        'copy = TextContent(":copy.line")\n'
        'cta = Button(":cta.label")',
    ),
    (
        "Horizontal row of two buttons",
        'root = Stack([primary, secondary], "row", "s")\n'
        'primary = Button(":actions.primary")\n'
        'secondary = Button(":actions.secondary")',
    ),
    (
        "Pricing card with subscribe button",
        'root = Stack([plan, subscribe], "column")\n'
        'plan_title = TextContent(":pricing.title")\n'
        'plan_body = TextContent(":pricing.body")\n'
        "plan = Card([plan_title, plan_body])\n"
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

    from slm_training.design_md import load_default_design_md
    from slm_training.dsl.parser import validate
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    design = load_default_design_md()
    records = [
        ExampleRecord(
            id=str(i),
            prompt=p,
            openui=validate(o).serialized or o,
            split="train",
            design_md=design,
        )
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
            grammar_ltr_primary=True,
            grammar_ltr_repair=True,
            grammar_finalize_validate=True,
            grammar_ltr_max_tokens=192,
            context_backend="scratch",
            design_md_in_context=True,
            seed=0,
        ),
        device="cpu",
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    model.train()
    last = 0.0
    for step in range(args.steps):
        loss = model.training_loss(records)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{args.steps} loss={last:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"wrote {args.output} (last_loss={last:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
