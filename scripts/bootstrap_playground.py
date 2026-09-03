#!/usr/bin/env python3
"""Train the web playground demo checkpoint and persist it under src/slm_training/resources/checkpoints/."""

from __future__ import annotations

import argparse
from pathlib import Path


#: Markers are opaque and contiguous from ``:slot_0`` within each record.
#: They used to spell their own role (``:hero.title``, ``:cta.label``), which
#: the output contract forbids precisely because a marker that names its role
#: hands the model the semantics the contract withholds -- so
#: ``assert_canonical_template_markers`` refused every record here and this
#: script, the documented way to regenerate the committed demo checkpoint,
#: could not run.
DEMO_RECORDS = [
    (
        "Hero card with title and body",
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":slot_0")\n'
        'hero_body = TextContent(":slot_1")\n'
        "hero = Card([hero_title, hero_body])",
    ),
    (
        "Primary call to action button",
        'root = Stack([cta])\ncta = Button(":slot_0")',
    ),
    (
        "Two feature cards stacked vertically",
        'root = Stack([a, b], "column", "m")\n'
        'a_title = TextContent(":slot_0")\n'
        'a_body = TextContent(":slot_1")\n'
        "a = Card([a_title, a_body])\n"
        'b_title = TextContent(":slot_2")\n'
        'b_body = TextContent(":slot_3")\n'
        "b = Card([b_title, b_body])",
    ),
    (
        "Text blurb above a button",
        'root = Stack([copy, cta], "column")\n'
        'copy = TextContent(":slot_0")\n'
        'cta = Button(":slot_1")',
    ),
    (
        "Horizontal row of two buttons",
        'root = Stack([primary, secondary], "row", "s")\n'
        'primary = Button(":slot_0")\n'
        'secondary = Button(":slot_1")',
    ),
    (
        "Pricing card with subscribe button",
        'root = Stack([plan, subscribe], "column")\n'
        'plan_title = TextContent(":slot_0")\n'
        'plan_body = TextContent(":slot_1")\n'
        "plan = Card([plan_title, plan_body])\n"
        'subscribe = Button(":slot_2")',
    ),
]


def _assert_trainable(records: list) -> None:
    """Apply the trainer's record contracts before spending any training time.

    ``TwoTowerModel.from_records`` asserts these and raises on the first
    violation. Checking here fails in a tenth of a second with the offending
    record named, instead of after the model is built.
    """
    from slm_training.data.contract import (
        assert_canonical_template_markers,
        assert_no_template_semantic_labels,
    )
    from slm_training.dsl.analysis.templatize import assert_role_safe_output
    from slm_training.dsl.language_contract import assert_symbol_only_output

    for record in records:
        try:
            assert_no_template_semantic_labels(record.prompt, record.design_md)
            assert_canonical_template_markers(record)
            assert_symbol_only_output(record.openui, output_kind="document")
            assert_role_safe_output(record.openui, output_kind="document")
        except Exception as exc:  # noqa: BLE001 — re-raised with the record id
            raise ValueError(
                f"demo record {record.id} violates a trainer contract: {exc}"
            ) from exc


def main(argv: list[str] | None = None) -> int:
    from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PLAYGROUND_DEMO_CHECKPOINT,
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        print(f"checkpoint already exists: {args.output}")
        return 0

    import torch

    from slm_training.dsl.design_md import load_default_design_md
    from slm_training.dsl.parser import validate
    from slm_training.dsl.placeholders import extract_placeholders
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    design = load_default_design_md()
    records = []
    for index, (prompt, openui) in enumerate(DEMO_RECORDS, start=1):
        serialized = validate(openui).serialized or openui
        records.append(
            ExampleRecord(
                id=str(index),
                prompt=prompt,
                openui=serialized,
                # Declared from the serialized program so the slot contract
                # and the target can never disagree.
                placeholders=list(extract_placeholders(serialized)),
                split="train",
                design_md=design,
            )
        )
    _assert_trainable(records)
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
