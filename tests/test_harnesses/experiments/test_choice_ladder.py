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


def test_choice_tokenizer_round_trips_and_persists(tmp_path) -> None:
    src = 'root = Stack([t], "column")\nt = TextContent(":hero.title")'
    tok = ChoiceTokenizer.build([src])
    ids = tok.encode(src)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    decoded = tok.decode_with_contract(ids, [":hero.title"])
    assert ":hero.title" in decoded and "root = Stack(" in decoded
    # Canonical-space by construction: re-encoding the decode is the identity.
    assert tok.encode(decoded) == ids
    # Generic-contract fallback still produces a parseable program.
    assert "root = Stack(" in tok.decode(ids)
    # Illegal decision streams fail closed to the empty string.
    assert tok.decode([tok.bos_id, tok.token_to_id["."], tok.eos_id]) == ""
    # Non-program probe spans encode empty rather than raising.
    assert tok.encode('":hero.title"', add_special=False) == []
    # Save/load round trip.
    path = tmp_path / "choice_tokenizer.json"
    tok.save(path)
    assert ChoiceTokenizer.load(path).token_to_id == tok.token_to_id


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
    # Matched pair: same width, both trained, finite losses.
    assert lexer["d_model"] == choice["d_model"] == 16
    for row in rows:
        assert math.isfinite(row["train_loss_final"])
        assert math.isfinite(row["heldout_nll_per_token"])
        assert math.isfinite(row["heldout_nll_per_decision"])
    # The representational asymmetry the study measures: choice targets are
    # strictly shorter than surface targets and use a smaller vocabulary.
    assert choice["tokens_per_program"] < lexer["tokens_per_program"]
    assert choice["vocab_size"] < lexer["vocab_size"]
    # Choice NLL-per-token IS per-decision (renormalization ~= identity).
    assert choice["heldout_nll_per_decision"] == pytest.approx(
        choice["heldout_nll_per_token"]
        * choice["tokens_per_program"]
        / choice["decisions_per_program"],
        rel=1e-6,
    )
