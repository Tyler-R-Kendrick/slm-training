"""B3 (SLM-23): matched capacity ladder over target representations."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.choice_ladder import run_choice_ladder
from slm_training.models.choice_tokenizer import ChoiceTokenizer


def _records(n: int, *, split: str) -> list[ExampleRecord]:
    programs = [
        'root = Stack([t], "column")\nt = TextContent(":hero.title")',
        'root = Stack([c])\nc = Card([t])\nt = TextContent(":hero.body")',
        'root = Stack([b])\nb = Button(":cta.label")',
    ]
    return [
        ExampleRecord(
            id=f"{split}-{i}",
            prompt=f"Layout {split} {i}",
            openui=programs[i % len(programs)],
            placeholders=[],
            split="train" if split == "train" else "held_out",
        )
        for i in range(n)
    ]


def test_choice_tokenizer_encode_is_canonical_space() -> None:
    # Main's corpus-independent grammar vocabulary (PR #288). The B3 study
    # only relies on encode length + vocab size, both stable here.
    src = 'root = Stack([t], "column")\nt = TextContent(":hero.title")'
    tok = ChoiceTokenizer.build()
    ids = tok.encode(src, placeholders=[":hero.title"])
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    # Choice targets are strictly shorter than the surface program text.
    assert 0 < len(ids) < len(src)
    # Fail-closed decode: a stream carrying <mask>/<unk> yields "".
    assert tok.decode([tok.bos_id, tok.mask_id, tok.eos_id]) == ""


def test_ladder_produces_matched_per_decision_rows() -> None:
    summary = run_choice_ladder(
        _records(6, split="train"),
        _records(3, split="heldout"),
        widths=(16,),
        steps=4,
        seed=0,
    )
    rows = summary["rows"]
    assert [r["target"] for r in rows] == ["lexer", "choice"]
    lexer, choice = rows
    assert lexer["d_model"] == choice["d_model"] == 16
    for row in rows:
        assert math.isfinite(row["train_loss_final"])
        assert math.isfinite(row["heldout_nll_per_token"])
        assert math.isfinite(row["heldout_nll_per_decision"])
    # The representational asymmetry the study measures: choice targets are
    # strictly shorter than surface targets (fewer decisions per program).
    # (Main's choice vocabulary is corpus-independent full-grammar, so it is
    # NOT necessarily smaller than a tiny-fixture lexer vocab — length, not
    # vocab size, is the load-bearing quantity here.)
    assert choice["tokens_per_program"] < lexer["tokens_per_program"]
    # Per-decision NLL is the per-token NLL renormalized by the tokens→
    # decisions ratio (the E1 bits-per-semantic-decision quantity), for both
    # arms and independent of the tokenizer's byte-framing.
    for row in rows:
        # Reconstructed from the rounded reported fields, so allow rounding
        # slack (the stored value is computed from unrounded quantities).
        assert row["heldout_nll_per_decision"] == pytest.approx(
            row["heldout_nll_per_token"]
            * row["tokens_per_program"]
            / row["decisions_per_program"],
            rel=1e-2,
        )
