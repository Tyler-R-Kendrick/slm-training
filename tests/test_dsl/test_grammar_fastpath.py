"""Unit tests for grammar fast-path (force-emit + MaskGIT admit)."""

from __future__ import annotations

from pathlib import Path

from slm_training.dsl.schema import ExampleRecord
from slm_training.grammar_fastpath import (
    OpenUIIncrementalEngine,
    admit_fill,
    draft_forced_ids,
    engine_for_dsl,
    force_next_token_id,
)
from slm_training.models.grammar import force_emit_token_id, pick_constrained_token
from slm_training.models.tokenizer import OpenUITokenizer
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

SAMPLE = 'root = Card(":t.x")\n'


def _tok() -> OpenUITokenizer:
    return OpenUITokenizer.build([SAMPLE, 'root = Row(":j")\n', "root = Stack([hero])\n"])


def test_engine_force_equal_after_name() -> None:
    eng = OpenUIIncrementalEngine()
    assert eng.set_prefix("root")
    assert eng.is_deterministic_next() == "="


def test_engine_force_lpar_after_component() -> None:
    eng = OpenUIIncrementalEngine()
    assert eng.set_prefix("root = Row")
    assert eng.is_deterministic_next() == "("


def test_force_next_token_id_maps_equal() -> None:
    tok = _tok()
    eng = engine_for_dsl("openui")
    assert eng is not None
    tid = force_next_token_id(eng, tok, "root")
    assert tid is not None
    assert tok.id_to_token[tid] == "="


def test_force_emit_token_id_via_grammar_helper() -> None:
    tok = _tok()
    ids = tok.encode("root", add_special=False)
    forced = force_emit_token_id(tok, ids)
    assert forced is not None
    assert tok.id_to_token[forced] == "="


def test_pick_constrained_honors_forced_id() -> None:
    import torch

    tok = _tok()
    equal_id = tok.token_to_id["="]
    logits = torch.zeros(tok.vocab_size)
    # Make a wrong token look best so force must win.
    wrong = tok.token_to_id.get("Card", equal_id)
    logits[wrong] = 10.0
    prefix = tok.encode("root", add_special=False)
    choice = pick_constrained_token(
        logits, tok, prefix, forced_token_id=equal_id
    )
    assert choice == equal_id


def test_admit_fill_accepts_partial_with_holes() -> None:
    tok = _tok()
    eng = OpenUIIncrementalEngine()
    ids = tok.encode(SAMPLE, add_special=True)
    # Mask the string literal — left span `root = Card(` is a valid incomplete prefix.
    str_id = tok.token_to_id['":t.x"']
    pos = ids.index(str_id)
    ids[pos] = tok.mask_id
    assert admit_fill(eng, tok, ids) is True


def test_admit_fill_rejects_hard_prefix() -> None:
    tok = _tok()
    eng = OpenUIIncrementalEngine()
    # Leading close-paren is never a valid OpenUI prefix.
    bad = [tok.bos_id, tok.token_to_id[")"], tok.mask_id, tok.eos_id]
    assert admit_fill(eng, tok, bad) is False


def test_draft_forced_ids_emits_equal() -> None:
    tok = _tok()
    eng = OpenUIIncrementalEngine()
    prefix = tok.encode("root", add_special=False)
    drafted = draft_forced_ids(eng, tok, list(prefix), max_tokens=2)
    assert drafted
    assert tok.id_to_token[drafted[0]] == "="


def test_train_fuse_and_cache_smoke() -> None:
    records = [
        ExampleRecord(
            id="t1",
            prompt="hero card",
            openui=SAMPLE,
            design_md="# Design\n",
            split="train",
            source="fixture",
        )
    ]
    cfg = TwoTowerConfig(
        context_backend="scratch",
        d_model=64,
        n_heads=2,
        context_layers=1,
        denoiser_layers=2,
        cache_context=True,
        fuse_ltr_loss=True,
        fastpath_aux_weight=0.1,
        grammar_fastpath=True,
        max_target_len=64,
        max_prompt_len=64,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    loss1 = float(model.training_loss(records).detach())
    loss2 = float(model.training_loss(records).detach())
    assert loss1 == loss1 and loss2 == loss2  # finite
    assert "t1" in model._context_text_cache
    assert "t1" in model._target_ids_cache


def test_cactus_kernel_sketch_files_exist() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "slm_training"
        / "cactus"
        / "kernels"
    )
    assert (root / "force_emit_sketch.hpp").is_file()
    assert (root / "maskgit_admit_sketch.hpp").is_file()
    assert (root / "README.md").is_file()
