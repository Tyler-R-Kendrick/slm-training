"""A2 (SLM-38): ASAp ledger is live on the MaskGIT unmask decode path.

Drives the actual constrained MaskGIT loop (``_generate_maskgit_one`` via
``generate``) with the single-step ASAp correction on vs off and asserts:

* default-off decode never touches the ledger (byte-identical mechanism: the
  ASAp path is inert unless ``asap_reweight`` is set);
* asap-on decode records real grammar-removed mass in the ledger (live).

The frontier quality verdict is out of scope here (fixture/CPU scale). See
docs/design/iter-a2-asap-constrained-decode-20260717.md.
"""

from __future__ import annotations

from dataclasses import replace

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

SAMPLE = 'root = Card(":t.x")\n'


def _records() -> list[ExampleRecord]:
    return [
        ExampleRecord(
            id="t1",
            prompt="hero card",
            openui=SAMPLE,
            design_md="# Design\n",
            split="train",
            source="fixture",
        )
    ]


def _maskgit_config(*, asap: bool) -> TwoTowerConfig:
    # Route through the MaskGIT unmask loop (not LTR / compiler-tree).
    return TwoTowerConfig(
        context_backend="scratch",
        d_model=64,
        n_heads=2,
        context_layers=1,
        denoiser_layers=2,
        grammar_constrained=True,
        grammar_ltr_primary=False,
        compiler_decode_mode="off",
        grammar_fastpath=True,
        gen_steps=8,
        grammar_ltr_max_tokens=24,
        max_target_len=32,
        max_prompt_len=32,
        asap_reweight=asap,
        asap_alpha=1.0,
        asap_defer_mass=0.5,
        seed=0,
    )


def _decode_with_stats(config: TwoTowerConfig):
    model = TwoTowerModel.from_records(_records(), config=config, device="cpu")
    model.eval()
    text, stats = model.generate_with_stats("hero card", grammar_constrained=True)
    return text, stats


def test_asap_off_is_dormant() -> None:
    _text, stats = _decode_with_stats(_maskgit_config(asap=False))
    # Flag off → the ledger is never created or merged.
    assert stats.asap_positions == 0
    assert stats.asap_removed_mass_sum == 0.0
    assert stats.asap_nonzero_removed == 0


def test_asap_on_records_removed_mass() -> None:
    _text, stats = _decode_with_stats(_maskgit_config(asap=True))
    # Ledger is live: the constrained MaskGIT loop measured grammar-removed
    # mass at committed positions.
    assert stats.asap_positions > 0
    assert stats.asap_removed_mass_sum > 0.0
    assert 0.0 <= stats.asap_max_removed_mass <= 1.0


def test_asap_off_decode_is_deterministic() -> None:
    cfg = _maskgit_config(asap=False)
    text_a, _ = _decode_with_stats(cfg)
    text_b, _ = _decode_with_stats(replace(cfg))
    assert text_a == text_b
