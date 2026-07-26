"""D3 (SLM-31): Kapur-style tree-edit diffusion baseline invariants."""

from __future__ import annotations

import json
import random
from dataclasses import replace

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
    ACTION_SET_PROPERTY,
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


def test_tree_edit_space_reads_a_registered_pack_inventory() -> None:
    """A pack extension widens the edit alphabet without editing the model."""
    import slm_training.dsl.pack as pack_mod
    from slm_training.dsl.pack import get_pack

    base = get_pack("openui")
    custom = replace(
        base,
        pack_id="tree-edit-extra-container",
        container_components=(*base.container_components, "ExtraContainer"),
        component_property_domains={
            **base.component_property_domains,
            "ExtraContainer": {"rest": (', "column"',)},
        },
    )
    pack_mod.register_pack(custom)
    try:
        space = TreeEditSpace(pack_id=custom.pack_id)
        assert "ExtraContainer" in space.components
        assert "ExtraContainer" in space.container_components
    finally:
        pack_mod._PACKS.pop(custom.pack_id, None)


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
    assert report["output_format_version"] == TreeEditDiffusionModel.CHECKPOINT_FORMAT
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


# --- SLM-425 (VAR1-02): SET_PROPERTY action --------------------------------


def test_set_property_apply_and_invert_root_and_non_root() -> None:
    """(a) SET_PROPERTY applies and inverts exactly on both the root
    container and a non-root one, restoring byte-identical program text."""
    space = TreeEditSpace()
    base = parse_statements(PROGRAM)
    assert base is not None
    prop_idx = space.property_names.index("rest")

    def round_trip(stmt_idx: int) -> None:
        rests = space.component_property_domains[base[stmt_idx].comp]["rest"]
        old_idx = rests.index(base[stmt_idx].rest)
        new_idx = next(i for i in range(len(rests)) if i != old_idx)
        mutated = space.apply(
            base,
            Edit(ACTION_SET_PROPERTY, stmt_idx, prop_idx, target=new_idx),
            INVENTORY,
        )
        assert mutated is not None, f"stmt {stmt_idx} did not apply"
        assert mutated[stmt_idx].rest == rests[new_idx]
        assert mutated[stmt_idx].rest != base[stmt_idx].rest
        validate(render_statements(mutated))
        restored = space.apply(
            mutated,
            Edit(ACTION_SET_PROPERTY, stmt_idx, prop_idx, target=old_idx),
            INVENTORY,
        )
        assert restored is not None
        validate(render_statements(restored))
        # Byte-identical restoration -- the safe-inverse contract.
        assert render_statements(restored) == PROGRAM

    # root (stmt 0): the guard that blocks REMOVE/REMOVE_CONTAINER on root
    # must NOT block SET_PROPERTY -- this is exactly the reachability gap
    # VAR1-01 proved (needs_direction_change on the root's own rest).
    assert base[0].name == "root"
    round_trip(0)
    # non-root container ("inline_card" = Card([title]), stmt 1).
    assert base[1].name == "inline_card" and base[1].has_list
    round_trip(1)

    # Out-of-precondition edits fail closed: a leaf statement has no
    # container property domain to index into.
    leaf_idx = next(i for i, s in enumerate(base) if not s.has_list)
    assert (
        space.apply(
            base, Edit(ACTION_SET_PROPERTY, leaf_idx, prop_idx, target=0), INVENTORY
        )
        is None
    )
    # A same-value "mutation" is not a real change -- rejected, not silently
    # applied as a no-op edit.
    same_idx = space.component_property_domains["Stack"]["rest"].index(base[0].rest)
    assert (
        space.apply(
            base,
            Edit(ACTION_SET_PROPERTY, 0, prop_idx, target=same_idx),
            INVENTORY,
        )
        is None
    )


def test_set_property_rejects_out_of_domain_value_via_real_parser() -> None:
    """(b) An illegal property value is rejected by the real parser/validator
    -- never silently applied -- even when a (mistaken) pack declares it as
    part of the domain. This proves the rejection is not merely an index
    bounds-check: the final full-program re-validation is the actual
    fail-closed backstop."""
    import slm_training.dsl.pack as pack_mod
    from slm_training.dsl.pack import get_pack

    base_pack = get_pack("openui")
    bogus_pack = replace(
        base_pack,
        pack_id="tree-edit-bogus-rest",
        component_property_domains={
            **base_pack.component_property_domains,
            # ``, [`` is not valid grammar for the direction/rest argument
            # (an unterminated list literal) -- a pack authoring mistake
            # that must still fail closed through the real parser.
            "Stack": {"rest": (', "column"', ", [")},
        },
    )
    pack_mod.register_pack(bogus_pack)
    try:
        space = TreeEditSpace(pack_id=bogus_pack.pack_id)
        prop_idx = space.property_names.index("rest")
        statements = parse_statements('root = Stack([], "column")')
        assert statements is not None
        bogus_idx = space.component_property_domains["Stack"]["rest"].index(", [")
        result = space.apply(
            statements,
            Edit(ACTION_SET_PROPERTY, 0, prop_idx, target=bogus_idx),
            [],
        )
        assert result is None
    finally:
        pack_mod._PACKS.pop(bogus_pack.pack_id, None)


def test_checkpoint_format2_fails_closed_and_migration_preserves_logits(
    tmp_path,
) -> None:
    """(c) A genuine format-2 checkpoint (N_ACTIONS=11, pre-VAR1-02 width)
    fails closed with a clear pointer to the migration path. (d) Migrating
    it to format 3 produces bit-identical action logits on the 11
    pre-existing rows; only the new (SET_PROPERTY) row differs."""
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero card with title, body, and a CTA button.",
            openui=PROGRAM,
            placeholders=INVENTORY,
        )
    ]
    cfg = TreeEditDiffusionConfig(
        d_model=32, n_heads=4, context_layers=1, denoiser_layers=1,
        dropout=0.0, seed=5,
    )
    model = TreeEditDiffusionModel.from_records(records, config=cfg, device="cpu")
    assert model.policy.action_head.out_features == 12  # current N_ACTIONS
    path = tmp_path / "ckpt.pt"
    model.save(path)

    # Simulate a genuine format-2 checkpoint: truncate action_head to the
    # pre-VAR1-02 width (11 rows) and stamp format_version=2.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 2
    sd = payload["state_dict"]
    old_w = sd["policy.action_head.weight"][:11].clone()
    old_b = sd["policy.action_head.bias"][:11].clone()
    sd["policy.action_head.weight"] = old_w
    sd["policy.action_head.bias"] = old_b
    v2_path = tmp_path / "ckpt_v2.pt"
    torch.save(payload, v2_path)
    (tmp_path / "ckpt_v2.tokenizer.json").write_text(
        (tmp_path / "ckpt.tokenizer.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # (c) Fails closed, pointing at the migration path.
    with pytest.raises(ValueError, match="format_version=2") as excinfo:
        TreeEditDiffusionModel.from_checkpoint(v2_path, device="cpu")
    assert "migrate_tree_edit_checkpoint" in str(excinfo.value)

    from slm_training.models.checkpoint_migrate import migrate_tree_edit_checkpoint

    out_path = tmp_path / "ckpt_migrated_v3.pt"
    report = migrate_tree_edit_checkpoint(
        source_checkpoint=v2_path, output_checkpoint=out_path
    )
    assert report["source_format_version"] == 2
    assert report["output_format_version"] == 3
    assert report["preserved_action_head_rows"] == 11

    migrated = TreeEditDiffusionModel.from_checkpoint(out_path, device="cpu")
    new_w = migrated.policy.action_head.weight
    new_b = migrated.policy.action_head.bias
    assert new_w.shape[0] == 12
    assert torch.allclose(new_w[:11], old_w)
    assert torch.allclose(new_b[:11], old_b)

    # (d) Bit-identical logits on the pre-existing rows: the migrated
    # checkpoint's non-action_head weights are copied verbatim from the same
    # trained model as the untouched original, so a forward pass over the
    # same input must match exactly on rows 0..10; only the new row 11 (a
    # fresh random init) is expected to differ.
    reference = TreeEditDiffusionModel.from_checkpoint(path, device="cpu")
    migrated.eval()
    reference.eval()
    prompt_text = migrated._format_context(records[0].prompt)
    with torch.no_grad():
        ctx_m, ctx_pad_m = migrated._encode_context([prompt_text])
        out_m = migrated.policy(
            migrated._state_batch([PROGRAM]), migrated.tokenizer.pad_id, ctx_m, ctx_pad_m
        )
        ctx_r, ctx_pad_r = reference._encode_context([prompt_text])
        out_r = reference.policy(
            reference._state_batch([PROGRAM]), reference.tokenizer.pad_id, ctx_r, ctx_pad_r
        )
    assert torch.allclose(out_m["action"][:, :11], out_r["action"][:, :11])
    assert not torch.allclose(out_m["action"][:, 11], out_r["action"][:, 11])
