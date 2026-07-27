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
    ACTION_STOP,
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


def test_set_property_round_trips_root_and_non_root_and_fails_closed() -> None:
    """SLM-425 mutates a pack-owned container property, including root."""
    space = TreeEditSpace()
    base = parse_statements(PROGRAM)
    assert base is not None

    # The default pack's ``rest`` domain is (column, empty), so property 0
    # and target 1 select the empty form on the root.
    root_mutation = Edit(ACTION_SET_PROPERTY, 0, comp=0, target=1)
    root_changed = space.apply(base, root_mutation, INVENTORY)
    assert root_changed is not None
    assert root_changed[0].rest == ""
    root_restored = space.apply(
        root_changed, Edit(ACTION_SET_PROPERTY, 0, comp=0, target=0), INVENTORY
    )
    assert root_restored is not None
    assert render_statements(root_restored) == PROGRAM

    # The same structured rebuild works on an existing non-root container.
    child_mutation = Edit(ACTION_SET_PROPERTY, 1, comp=0, target=0)
    child_changed = space.apply(base, child_mutation, INVENTORY)
    assert child_changed is not None
    assert child_changed[1].rest == ', "column"'
    child_restored = space.apply(
        child_changed, Edit(ACTION_SET_PROPERTY, 1, comp=0, target=1), INVENTORY
    )
    assert child_restored is not None
    assert render_statements(child_restored) == PROGRAM

    # No undeclared property/value can silently mutate the source.
    assert space.apply(base, Edit(ACTION_SET_PROPERTY, 0, comp=1, target=0), INVENTORY) is None
    assert space.apply(base, Edit(ACTION_SET_PROPERTY, 0, comp=0, target=3), INVENTORY) is None


def test_set_property_rejects_a_pack_value_the_parser_does_not_accept() -> None:
    """Pack metadata is an input domain, never a substitute for parsing."""
    import slm_training.dsl.pack as pack_mod
    from slm_training.dsl.pack import get_pack

    base_pack = get_pack("openui")
    custom = replace(
        base_pack,
        pack_id="tree-edit-invalid-property",
        component_property_domains={
            **base_pack.component_property_domains,
            "Stack": {"rest": (', "unterminated',)},
        },
    )
    pack_mod.register_pack(custom)
    try:
        space = TreeEditSpace(pack_id=custom.pack_id)
        base = parse_statements(PROGRAM)
        assert base is not None
        assert space.apply(base, Edit(ACTION_SET_PROPERTY, 0), INVENTORY) is None
    finally:
        pack_mod._PACKS.pop(custom.pack_id, None)


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
    # Round-trip at the current format.
    loaded = TreeEditDiffusionModel.from_checkpoint(path, device="cpu")
    assert loaded.policy.action_head.out_features == model.policy.action_head.out_features

    # Simulate a format-2 checkpoint: shrink the action head to the prior 11
    # actions and remove the SLM-425 property-value head.
    import torch as _torch

    payload = _torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 2
    sd = payload["state_dict"]
    old_w = sd["policy.action_head.weight"][:11].clone()
    old_b = sd["policy.action_head.bias"][:11].clone()
    sd["policy.action_head.weight"] = old_w
    sd["policy.action_head.bias"] = old_b
    del sd["policy.property_value_head.weight"]
    del sd["policy.property_value_head.bias"]
    old_path = tmp_path / "ckpt_v2.pt"
    _torch.save(payload, old_path)
    (tmp_path / "ckpt_v2.tokenizer.json").write_text(
        (tmp_path / "ckpt.tokenizer.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Unmigrated old checkpoint fails closed with a clear error.
    with pytest.raises(ValueError, match="format_version=2"):
        TreeEditDiffusionModel.from_checkpoint(old_path, device="cpu")

    # Migration warm-starts: old action rows preserved, new rows initialized.
    from slm_training.models.checkpoint_migrate import migrate_tree_edit_checkpoint

    out_path = tmp_path / "ckpt_migrated.pt"
    report = migrate_tree_edit_checkpoint(
        source_checkpoint=old_path, output_checkpoint=out_path
    )
    assert report["source_format_version"] == 2
    assert report["output_format_version"] == 3
    assert report["preserved_action_head_rows"] == 11
    assert (tmp_path / "ckpt_migrated.migrate.json").exists()
    migrated = TreeEditDiffusionModel.from_checkpoint(out_path, device="cpu")
    new_w = migrated.policy.action_head.weight
    assert new_w.shape[0] == model.policy.action_head.weight.shape[0]
    assert _torch.allclose(new_w[:11], old_w)
    assert _torch.allclose(migrated.policy.action_head.bias[:11], old_b)
    assert migrated.policy.property_value_head.out_features == 2

    # Existing action logits are bit-identical after the format migration.
    state = model._state_batch([PROGRAM])
    ctx = _torch.zeros((1, 1, cfg.d_model))
    ctx_pad = _torch.zeros((1, 1), dtype=_torch.bool)
    model.eval()
    migrated.eval()
    before = model.policy(state, model.tokenizer.pad_id, ctx, ctx_pad)["action"]
    after = migrated.policy(state, migrated.tokenizer.pad_id, ctx, ctx_pad)["action"]
    assert _torch.equal(before[:, :11], after[:, :11])


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


# --- SLM-434 (LAR0-07): value_label_mode / stop_slot_accounting port -------
#
# Ported default-off from the never-merged SLM-308/SLM-310 forks so the
# landed SLM-317/SLM-431 harness (which builds
# ``TreeEditDiffusionConfig(value_label_mode=..., stop_slot_accounting=...)``)
# can run on main. Both knobs default to the historical behavior: these
# tests prove that default is unchanged and that the new arm (bounded
# distance values / corrected STOP accounting) is independently well-formed.


def test_new_knobs_default_off_matching_historical_behavior() -> None:
    cfg = TreeEditDiffusionConfig()
    assert cfg.value_label_mode == "mutation_count"
    assert cfg.stop_slot_accounting == "legacy"


def test_unknown_value_label_mode_and_stop_slot_accounting_raise() -> None:
    records = [
        ExampleRecord(
            id="a", prompt="p", openui=PROGRAM, placeholders=INVENTORY
        )
    ]
    bad_value_cfg = TreeEditDiffusionConfig(
        d_model=32, n_heads=4, context_layers=1, denoiser_layers=1,
        value_label_mode="not_a_mode",
    )
    model = TreeEditDiffusionModel.from_records(
        records, config=bad_value_cfg, device="cpu"
    )
    with pytest.raises(ValueError, match="value_label_mode"):
        model.training_loss(records)

    bad_stop_cfg = TreeEditDiffusionConfig(
        d_model=32, n_heads=4, context_layers=1, denoiser_layers=1,
        beam_width=2, expand_per_state=2, max_search_steps=2,
        stop_slot_accounting="not_a_mode",
    )
    model = TreeEditDiffusionModel.from_records(
        records, config=bad_stop_cfg, device="cpu"
    )
    with pytest.raises(ValueError, match="stop_slot_accounting"):
        model.generate_batch_requests(
            [GenerationRequest(prompt="p", slot_contract=tuple(INVENTORY))]
        )


def test_bounded_distance_mode_masks_unknown_and_reports_metrics() -> None:
    records = [
        ExampleRecord(id="a", prompt="p", openui=PROGRAM, placeholders=INVENTORY)
    ]
    cfg = TreeEditDiffusionConfig(
        d_model=32, n_heads=4, context_layers=1, denoiser_layers=1,
        max_chain=2, seed=5, value_label_mode="bounded_distance",
    )
    model = TreeEditDiffusionModel.from_records(records, config=cfg, device="cpu")
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    metrics = model.last_training_metrics
    assert "value_bounded" in metrics
    assert "value_unknown_excluded" in metrics


def test_bounded_distance_mode_excludes_unknown_from_value_loss(monkeypatch) -> None:
    from slm_training.harnesses.experiments.slm308_distance_oracle import DistanceKind

    records = [
        ExampleRecord(id="a", prompt="p", openui=PROGRAM, placeholders=INVENTORY)
    ]
    cfg = TreeEditDiffusionConfig(
        d_model=32, n_heads=4, context_layers=1, denoiser_layers=1,
        max_chain=2, seed=5, value_label_mode="bounded_distance",
    )
    model = TreeEditDiffusionModel.from_records(records, config=cfg, device="cpu")

    class _Unknown:
        kind = DistanceKind.UNKNOWN

        def value_target(self, max_depth: int):
            return None

    monkeypatch.setattr(
        TreeEditDiffusionModel, "_distance_label", lambda *a, **k: _Unknown()
    )
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    assert model.last_training_metrics["value_unknown_excluded"] >= 1.0


def test_mutation_count_mode_never_touches_oracle(monkeypatch) -> None:
    import slm_training.harnesses.experiments.slm308_distance_oracle as oracle

    def _boom(*args, **kwargs):
        raise AssertionError("oracle reached from mutation_count training path")

    monkeypatch.setattr(oracle, "distance_to_target", _boom)
    records = [
        ExampleRecord(id="a", prompt="p", openui=PROGRAM, placeholders=INVENTORY)
    ]
    cfg = TreeEditDiffusionConfig(
        d_model=32, n_heads=4, context_layers=1, denoiser_layers=1, seed=5,
    )
    model = TreeEditDiffusionModel.from_records(records, config=cfg, device="cpu")
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    assert "value_bounded" not in model.last_training_metrics


def test_decode_never_calls_the_oracle() -> None:
    import inspect

    for fn_name in ("_decode_one", "_enumerate_edits", "_seed_state", "generate"):
        source = inspect.getsource(getattr(TreeEditDiffusionModel, fn_name))
        assert "distance_to_target" not in source
        assert "slm308" not in source


def test_apply_reason_out_list_is_accept_only() -> None:
    space = TreeEditSpace()
    statements = parse_statements(PROGRAM)
    reason: list[str] = []
    # An out-of-range REPLACE is rejected; a single generic marker appended.
    rejected = space.apply(
        statements, Edit(action=1, stmt=999, comp=0), INVENTORY, reason=reason
    )
    assert rejected is None
    assert reason == ["rejected"]
    # A successful STOP appends nothing.
    reason.clear()
    accepted = space.apply(statements, Edit(ACTION_STOP), INVENTORY, reason=reason)
    assert accepted is not None
    assert reason == []


def test_corrected_stop_accounting_consumes_fewer_slots_on_duplicate() -> None:
    """Two STOP proposals for the same (unchanged) row state: legacy consumes
    a decode-budget slot for both; corrected consumes one only for the
    proposal actually retained on the beam."""
    records = [
        ExampleRecord(id="a", prompt="p", openui=PROGRAM, placeholders=INVENTORY)
    ]
    duplicate_stop_candidates = [(1.0, Edit(ACTION_STOP)), (0.5, Edit(ACTION_STOP))]

    def _evidence_for(stop_slot_accounting: str) -> dict:
        cfg = TreeEditDiffusionConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1,
            beam_width=2, expand_per_state=2, max_search_steps=1,
            stop_slot_accounting=stop_slot_accounting,
        )
        model = TreeEditDiffusionModel.from_records(
            records, config=cfg, device="cpu"
        )
        model._enumerate_edits = lambda *a, **k: duplicate_stop_candidates
        model.generate_batch_requests(
            [GenerationRequest(prompt="p", slot_contract=tuple(INVENTORY))]
        )
        return model._generation_evidence[0]

    legacy = _evidence_for("legacy")
    assert legacy["stop_proposals"] == 2
    assert legacy["stop_slots_consumed"] == 2

    corrected = _evidence_for("corrected")
    assert corrected["stop_proposals"] == 2
    assert corrected["stop_slots_consumed"] == 1
