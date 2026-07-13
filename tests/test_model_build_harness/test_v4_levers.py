"""Tests for V4 critic / remask / honest-contract levers."""

from __future__ import annotations

import torch

from slm_training.models.parallel_decode import (
    select_remask_indices,
    select_remask_policy_indices,
)
from slm_training.models.template_fill import (
    build_slot_contract_template,
    ensure_prompt_inventory,
    inventory_from_prompt,
    normalize_placeholders,
)
from slm_training.models.tokenizer import OpenUITokenizer
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.dsl.schema import ExampleRecord


def test_inventory_from_prompt_explicit_line() -> None:
    prompt = "Build a hero.\nPlaceholders: :smoke.hero.title, :smoke.hero.body"
    inv = inventory_from_prompt(prompt, heuristic=False)
    assert inv == [":smoke.hero.title", ":smoke.hero.body"]


def test_inventory_from_prompt_heuristic() -> None:
    inv = inventory_from_prompt(
        "Build a hero card with title and body and a CTA button.",
        heuristic=True,
    )
    assert any(s.endswith(".title") for s in inv)
    assert any("cta" in s or s.endswith(".body") for s in inv)


def test_ensure_prompt_inventory_idempotent() -> None:
    slots = [":smoke.cta.label"]
    once = ensure_prompt_inventory("Single button.", slots)
    twice = ensure_prompt_inventory(once, slots)
    assert once == twice
    assert "Placeholders:" in once
    assert inventory_from_prompt(once, heuristic=False) == slots


def test_select_remask_policy_includes_grammar_and_respects_budget() -> None:
    conf = torch.tensor([[0.9, 0.8, 0.1, 0.2, 0.05]])
    known = torch.tensor([[True, True, True, True, True]])
    gate = torch.tensor([[0.9, 0.9, 0.1, 0.9, 0.2]])
    entropy = torch.tensor([[0.1, 0.1, 2.0, 0.1, 1.5]])
    idxs = select_remask_policy_indices(
        conf,
        known,
        remask_ratio=0.4,
        grammar_positions=[2],
        gate_trust=gate,
        entropy=entropy,
        gate_threshold=0.5,
    )
    assert 0 not in idxs  # BOS protected
    assert 2 in idxs  # grammar hard-error
    assert len(idxs) >= 2
    # Classic remask still works.
    assert select_remask_indices(conf, known, remask_ratio=0.2)


def test_visible_corrupt_marks_predict_mask() -> None:
    src = build_slot_contract_template([":a.title", ":a.body"])
    tok = OpenUITokenizer.build([src, "Make a card"])
    cfg = TwoTowerConfig(
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=64,
        max_target_len=128,
        visible_corrupt_rate=1.0,
        mdlm_schedule=False,
        mask_min=0.0,
        mask_max=0.0,
        seed=0,
    )
    model = TwoTowerModel(tokenizer=tok, config=cfg, device="cpu")
    ids = torch.tensor([tok.encode(src)], dtype=torch.long)
    noisy, predict, _ = model._mask_targets(ids)
    # With mask_min=max=0, only visible corruption should create predict positions.
    assert bool(predict.any())
    # Corrupted visibles differ from gold somewhere.
    assert not torch.equal(noisy, ids) or bool(predict.any())
    special = {
        tok.pad_id,
        tok.bos_id,
        tok.eos_id,
        tok.mask_id,
        tok.unk_id,
    }
    flipped = predict & noisy.ne(tok.mask_id)
    for sid in special:
        assert not bool((flipped & noisy.eq(sid)).any())
    assert not bool((flipped & noisy.eq(ids)).any())


def test_legacy_kind_lookup_checkpoint_is_ignored() -> None:
    """Checkpoints that persisted a unused kind_lookup still load."""
    src = 'root = Stack([t], "column")\nt = TextContent(":a.title")'
    tok = OpenUITokenizer.build([src, "prompt"])
    cfg = TwoTowerConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=32,
        max_target_len=64,
        factorized_embeddings=False,
    )
    model = TwoTowerModel(tokenizer=tok, config=cfg, device="cpu")
    sd = model.state_dict()
    # Simulate a pre-compat checkpoint that always wrote kind_lookup.
    assert "denoiser.kind_lookup" not in sd
    sd = dict(sd)
    sd["denoiser.kind_lookup"] = torch.zeros(tok.vocab_size, dtype=torch.long)
    from slm_training.models.twotower import _load_checkpoint_state

    _load_checkpoint_state(model, sd)

def test_honest_slot_contract_ignores_gold_placeholders() -> None:
    src = 'root = Stack([t], "column")\nt = TextContent(":hero.title")'
    tok = OpenUITokenizer.build([src, "prompt"])
    cfg = TwoTowerConfig(
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        honest_slot_contract=True,
        slot_contract_in_context=True,
        template_fill_decode=True,
        slot_contract_constrained_decode=True,
        grammar_ltr_primary=False,
        gen_steps=2,
        max_target_len=96,
    )
    model = TwoTowerModel(tokenizer=tok, config=cfg, device="cpu")
    gold = ExampleRecord(
        id="x",
        prompt="Build a hero card with title and body.",
        openui=src,
        placeholders=[":LEAKED.GOLD.slot"],
        split="train",
        source="fixture",
    )
    contract = model._resolve_slot_contract(gold.prompt, gold, None)
    assert contract is not None
    assert ":LEAKED.GOLD.slot" not in contract


def test_suffix_rollback_config_roundtrip() -> None:
    cfg = TwoTowerConfig(suffix_rollback_window=8, remask_use_entropy=True)
    assert cfg.suffix_rollback_window == 8
    assert cfg.remask_use_entropy is True


def test_normalize_placeholders_stable() -> None:
    assert normalize_placeholders(["a.b", ":a.b", "a.b"]) == [":a.b"]
