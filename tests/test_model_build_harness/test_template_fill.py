"""Tests for slot-contract template fill and remask helpers."""

from __future__ import annotations

from slm_training.models.parallel_decode import select_remask_indices
from slm_training.models.template_fill import (
    build_slot_contract_template,
    normalize_placeholders,
)
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text


def test_build_slot_contract_template_binds_inventory() -> None:
    src = build_slot_contract_template(
        [":smoke.hero.title", ":smoke.hero.body", "cta.label"]
    )
    assert 'root = Stack([' in src
    assert 'TextContent(":smoke.hero.title")' in src
    assert 'TextContent(":smoke.hero.body")' in src
    assert 'TextContent(":cta.label")' in src
    assert normalize_placeholders(["cta.label", ":cta.label"]) == [":cta.label"]


def test_template_token_length_fits_e18_budget() -> None:
    slots = [f":ns.slot{i}" for i in range(6)]
    src = build_slot_contract_template(slots)
    assert len(tokenize_text(src)) <= 192
    # Larger inventories still fit the E29 256-token canvas.
    big = build_slot_contract_template([f":ns.slot{i}" for i in range(12)])
    assert len(tokenize_text(big)) <= 256


def test_select_remask_indices_protects_bos() -> None:
    import torch

    conf = torch.tensor([[0.9, 0.1, 0.2, 0.05]])
    known = torch.tensor([[True, True, True, True]])
    idxs = select_remask_indices(conf, known, remask_ratio=0.5, protect_bos=True)
    assert 0 not in idxs
    assert len(idxs) >= 1


def test_tokenizer_roundtrip_template() -> None:
    src = build_slot_contract_template([":hero.title", ":hero.body"])
    tok = OpenUITokenizer.build([src])
    ids = tok.encode(src, add_special=False)
    assert tok.unk_id not in ids
