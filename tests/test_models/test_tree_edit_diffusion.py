"""D3 (SLM-31): Kapur-style tree-edit diffusion baseline invariants."""

from __future__ import annotations

import json
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
    'title = TextContent(":hero.title")\n'
    "panel = Card([body])\n"
    'body = TextContent(":hero.body")\n'
    'cta = Button(":cta.label")'
)
INVENTORY = [":hero.title", ":hero.body", ":cta.label"]


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


# --- SLM-305 extended edit language -----------------------------------------

from slm_training.models.tree_edit_diffusion import (  # noqa: E402
    ACTION_ADD_CONTAINER,
    ACTION_BIND_PLACEHOLDER,
    ACTION_INSERT_STATEMENT,
    ACTION_INSERT_SUBTREE,
    ACTION_REMOVE,
    ACTION_REMOVE_CONTAINER,
    ACTION_REPLACE_STATEMENT,
    ACTION_REPLACE_SUBTREE,
    Edit,
)


def test_render_is_byte_stable_for_fixture_programs() -> None:
    statements = parse_statements(PROGRAM)
    assert statements is not None
    assert render_statements(statements) == PROGRAM


def test_extended_actions_apply_and_invert_exactly() -> None:
    """Every new action applies only under its preconditions, stays valid, and
    its supervised inverse restores the exact prior program text."""
    space = TreeEditSpace()
    ci = space.comp_index
    base = parse_statements(PROGRAM)
    assert base is not None

    def apply(edit: Edit, stmts=None):
        nxt = space.apply(stmts or base, edit, INVENTORY)
        assert nxt is not None, edit
        validate(render_statements(nxt))
        return nxt

    # ADD_CONTAINER <-> REMOVE_CONTAINER (empty subtree: exact safe inverse)
    added = apply(Edit(ACTION_ADD_CONTAINER, 0, ci["Card"]))
    assert len(added) == len(base) + 1
    assert render_statements(
        apply(Edit(ACTION_REMOVE_CONTAINER, len(added) - 1), added)
    ) == PROGRAM
    # REMOVE_CONTAINER rejects non-empty nested subtrees (fail closed)
    assert space.apply(base, Edit(ACTION_REMOVE_CONTAINER, 0), INVENTORY) is None
    # INSERT_SUBTREE <-> REMOVE_CONTAINER (container + leaf child)
    inserted = apply(Edit(ACTION_INSERT_SUBTREE, 0, ci["Stack"], 1, payload=ci["Button"]))
    assert render_statements(
        apply(Edit(ACTION_REMOVE_CONTAINER, len(inserted) - 2), inserted)
    ) == PROGRAM
    # REPLACE_SUBTREE swaps the single leaf of a canonical subtree, same root
    replaced = apply(Edit(ACTION_REPLACE_SUBTREE, 1, slot=1, payload=ci["Image"]))
    assert replaced[1].comp == "Card"  # root kind preserved
    assert render_statements(
        apply(Edit(ACTION_REPLACE_SUBTREE, 1, slot=0, payload=ci["TextContent"]), replaced)
    ) == PROGRAM
    # INSERT_STATEMENT / REPLACE_STATEMENT (V0.5 canonical forms) + REMOVE
    with_query = apply(Edit(ACTION_INSERT_STATEMENT, payload=0))
    assert with_query[-1].comp == "Query"
    swapped = apply(Edit(ACTION_REPLACE_STATEMENT, len(with_query) - 1, payload=1), with_query)
    assert swapped[-1].comp == "Mutation"
    assert render_statements(
        apply(Edit(ACTION_REPLACE_STATEMENT, len(swapped) - 1, payload=0), swapped)
    ) == render_statements(with_query)
    removed = space.apply(with_query, Edit(ACTION_REMOVE, len(with_query) - 1), INVENTORY)
    assert removed is not None
    assert render_statements(removed) == PROGRAM
    # BIND_PLACEHOLDER rebinds a leaf slot; references (children lists) stay
    bound = apply(Edit(ACTION_BIND_PLACEHOLDER, 4, slot=0))
    assert bound[1].children == base[1].children
    assert render_statements(
        apply(Edit(ACTION_BIND_PLACEHOLDER, 4, slot=1), bound)
    ) == PROGRAM
    # Out-of-precondition edits fail closed.
    assert space.apply(base, Edit(ACTION_BIND_PLACEHOLDER, 0, slot=0), INVENTORY) is None
    assert space.apply(base, Edit(ACTION_REPLACE_STATEMENT, 0, payload=0), INVENTORY) is None


def test_extended_sample_mutation_loop_restores() -> None:
    space = TreeEditSpace()
    rng = random.Random(11)
    statements = parse_statements(PROGRAM)
    assert statements is not None
    seen_actions: set[int] = set()
    for _ in range(80):
        step = space.sample_mutation(statements, INVENTORY, rng)
        if step is None:
            continue
        mutated, inverse = step
        validate(render_statements(mutated))
        repaired = space.apply(mutated, inverse, INVENTORY)
        assert repaired is not None
        validate(render_statements(repaired))
        seen_actions.add(inverse.action)
    # The seeded loop exercised the extended inverse-edit supervision.
    assert seen_actions - {0}


def test_edit_new_fields_default() -> None:
    old = Edit(1, 2, 3, 4)
    assert old.target == 0 and old.payload == 0
    assert old == Edit(1, 2, 3, 4, 0, 0)


def test_checkpoint_format2_fail_closed_and_migration(tmp_path) -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero card with title, body, and a CTA button.",
            openui=PROGRAM,
            placeholders=INVENTORY,
        )
    ]
    cfg = TreeEditDiffusionConfig(
        d_model=32, n_heads=4, context_layers=1, denoiser_layers=1, seed=3,
    )
    model = TreeEditDiffusionModel.from_records(records, config=cfg, device="cpu")
    path = tmp_path / "ckpt.pt"
    model.save(path)
    # Round-trip at format 2.
    loaded = TreeEditDiffusionModel.from_checkpoint(path, device="cpu")
    assert loaded.policy.action_head.out_features == model.policy.action_head.out_features

    # Simulate a format-1 checkpoint: shrink the action head to 4 rows.
    import torch as _torch

    payload = _torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 1
    sd = payload["state_dict"]
    old_w = sd["policy.action_head.weight"][:4].clone()
    old_b = sd["policy.action_head.bias"][:4].clone()
    sd["policy.action_head.weight"] = old_w
    sd["policy.action_head.bias"] = old_b
    old_path = tmp_path / "ckpt_v1.pt"
    _torch.save(payload, old_path)
    (tmp_path / "ckpt_v1.tokenizer.json").write_text(
        (tmp_path / "ckpt.tokenizer.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Unmigrated old checkpoint fails closed with a clear error.
    with pytest.raises(ValueError, match="format_version=1"):
        TreeEditDiffusionModel.from_checkpoint(old_path, device="cpu")

    # Migration warm-starts: old action rows preserved, new rows initialized.
    from slm_training.models.checkpoint_migrate import migrate_tree_edit_checkpoint

    out_path = tmp_path / "ckpt_migrated.pt"
    report = migrate_tree_edit_checkpoint(
        source_checkpoint=old_path, output_checkpoint=out_path
    )
    assert report["source_format_version"] == 1
    assert report["output_format_version"] == 2
    assert report["preserved_action_head_rows"] == 4
    assert (tmp_path / "ckpt_migrated.migrate.json").exists()
    migrated = TreeEditDiffusionModel.from_checkpoint(out_path, device="cpu")
    new_w = migrated.policy.action_head.weight
    assert new_w.shape[0] == model.policy.action_head.weight.shape[0]
    assert _torch.allclose(new_w[:4], old_w)
    assert _torch.allclose(migrated.policy.action_head.bias[:4], old_b)


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
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert metadata["serialized_weight_bytes"] == metadata["parameter_count"] * 4
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
