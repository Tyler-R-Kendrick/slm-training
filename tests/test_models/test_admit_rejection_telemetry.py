"""N12: read-only admit-probe rejection telemetry for the MaskGIT lane.

`DecodeStats.admit_probe_rejections` / `admit_probe_reject_run_max` count
admit-probe proposal rejections and the longest run of consecutive rejections
with no intervening commit (the LAVE stall signal, arXiv:2602.00612). The
counters are observational only: no decode path reads them back, so a decode
run with no collector attached must emit exactly the same bytes as one run
inside `collect_decode_stats()`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":slot_0")\n'
    'hero_body = TextContent(":slot_1")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":slot_0")'
SEED_SOURCE = 'root = Card([title])\ntitle = TextContent(":slot_0")\n'


def _maskgit_model() -> TwoTowerModel:
    cfg = TwoTowerConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        gen_steps=4,
        seed=0,
        output_tokenizer="lexer",
        grammar_ltr_primary=False,
        grammar_fastpath_mode="mask",
        allow_unconstrained_fallback=False,
    )
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA", openui=CTA, split="train"),
    ]
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    model.eval()
    return model


def _seed_ids(model: TwoTowerModel, *, holes: int = 2) -> list[int]:
    tokenizer = model.tokenizer
    seed = [
        tokenizer.bos_id,
        *tokenizer.encode(SEED_SOURCE, add_special=False),
        tokenizer.eos_id,
    ]
    equals = [
        index
        for index, token_id in enumerate(seed)
        if token_id == tokenizer.token_to_id["="]
    ]
    for position in equals[:holes]:
        seed[position] = tokenizer.mask_id
    return seed


def _decode(model: TwoTowerModel, seed: list[int]) -> str:
    ctx, ctx_pad = model._encode_context(["Hero"])
    return model._generate_maskgit_one(
        ctx,
        ctx_pad,
        len(seed),
        use_grammar=True,
        slot_contract=[":slot_0"],
        seed_ids=list(seed),
    )


def test_admit_rejections_count_every_rejection_as_one_consecutive_run(
    monkeypatch,
) -> None:
    """With every admit probe rejecting, no commit ever breaks the run."""
    model = _maskgit_model()
    seed = _seed_ids(model)
    from slm_training.dsl.grammar import fastpath

    monkeypatch.setattr(fastpath, "admit_fill", lambda *_a, **_k: False)
    with collect_decode_stats() as stats:
        _decode(model, seed)
    assert stats.admit_probe_rejections >= 1
    # No positionwise commit happened, so the longest consecutive-rejection
    # run is the whole rejection sequence.
    assert stats.admit_probe_reject_run_max == stats.admit_probe_rejections


def test_admit_rejections_zero_when_every_probe_admits(monkeypatch) -> None:
    model = _maskgit_model()
    seed = _seed_ids(model)
    from slm_training.dsl.grammar import fastpath

    monkeypatch.setattr(fastpath, "admit_fill", lambda *_a, **_k: True)
    with collect_decode_stats() as stats:
        _decode(model, seed)
    assert stats.admit_probe_rejections == 0
    assert stats.admit_probe_reject_run_max == 0


def test_telemetry_is_off_by_default_and_output_is_byte_identical() -> None:
    """The counters are read-only: collecting them cannot move a single byte."""
    model = _maskgit_model()
    seed = _seed_ids(model)
    uncollected = _decode(model, seed)
    with collect_decode_stats() as stats:
        collected = _decode(model, seed)
    assert collected.encode("utf-8") == uncollected.encode("utf-8")
    # And a decode outside any collector leaves no active stats to mutate.
    from slm_training.models.decode_stats import get_active_stats

    assert get_active_stats() is None
    assert stats.admit_probe_rejections >= 0


def test_reject_run_max_merges_as_a_maximum_not_a_sum() -> None:
    from slm_training.models.decode_stats import DecodeStats

    left = DecodeStats(admit_probe_rejections=3, admit_probe_reject_run_max=3)
    right = DecodeStats(admit_probe_rejections=5, admit_probe_reject_run_max=2)
    left.merge(right)
    assert left.admit_probe_rejections == 8
    assert left.admit_probe_reject_run_max == 3
