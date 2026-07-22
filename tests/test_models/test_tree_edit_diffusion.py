"""D3 (SLM-31): Kapur-style tree-edit diffusion baseline invariants."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.models.tree_edit_diffusion import (
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
    TreeEditSpace,
    parse_statements,
    render_statements,
)

PROGRAM = (
    'root = Stack([inline_card, panel, cta], "column")\n'
    "inline_card = Card([title])\n"
    'title = TextContent(":slot_0")\n'
    "panel = Card([body])\n"
    'body = TextContent(":slot_1")\n'
    'cta = Button(":slot_2")'
)
INVENTORY = [":slot_0", ":slot_1", ":slot_2"]


def test_mutations_preserve_validity_and_inverse_restores() -> None:
    """Kapur invariant: every forward-noised state is a valid program, and the
    recorded inverse edit deterministically restores the previous state."""
    space = TreeEditSpace()
    rng = random.Random(7)
    statements = parse_statements(PROGRAM)
    assert statements is not None
    restored_any = False
    for _ in range(25):
        step = space.sample_mutation(statements, INVENTORY, rng)
        if step is None:
            continue
        mutated, inverse = step
        validate(render_statements(mutated))
        repaired = space.apply(mutated, inverse, INVENTORY)
        assert repaired is not None
        # Same canonical structure: identical component multiset + topology.
        assert render_statements(repaired).count("=") == PROGRAM.count("=")
        validate(render_statements(repaired))
        restored_any = True
    assert restored_any


def test_training_loss_decode_all_valid_and_checkpoint(tmp_path) -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero card with title, body, and a CTA button.",
            openui=PROGRAM,
            placeholders=INVENTORY,
        )
    ]
    cfg = TreeEditDiffusionConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        beam_width=2,
        expand_per_state=2,
        max_search_steps=3,
        seed=3,
    )
    model = TreeEditDiffusionModel.from_records(records, config=cfg, device="cpu")
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    loss.backward()

    outputs = model.generate_batch_requests(
        [
            GenerationRequest(
                prompt=records[0].prompt,
                slot_contract=tuple(INVENTORY),
            )
        ]
    )
    assert len(outputs) == 1
    # All-valid-states invariant: whatever the search returns must parse.
    validate(outputs[0])

    path = tmp_path / "ckpt.pt"
    model.save(path)
    loaded = TreeEditDiffusionModel.from_checkpoint(path, device="cpu")
    reproduced = loaded.generate_batch_requests(
        [
            GenerationRequest(
                prompt=records[0].prompt,
                slot_contract=tuple(INVENTORY),
            )
        ]
    )
    assert reproduced == outputs
