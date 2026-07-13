"""Decode canvas feasibility and grammar trust-model admission tests."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.model_build.decode_feasibility import (
    evaluate_decode_feasibility,
    gold_token_len,
    max_achievable_parse_rate,
)
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text


def test_gold_token_len_fixture_smoke_hero() -> None:
    import json
    from pathlib import Path

    for line in Path("fixtures/test_seeds.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("id") == "smoke_hero_01":
            length = gold_token_len(rec["openui"])
            assert length > 64, f"smoke_hero token len {length} should exceed legacy cap"
            assert length <= 256
            return
    raise AssertionError("smoke_hero_01 not found in fixtures")


def test_max_achievable_parse_rate_cap_64() -> None:
    lengths = [160, 107, 106, 53, 40]
    assert max_achievable_parse_rate(lengths, 64) == 0.4


def test_decode_feasibility_fails_at_cap_64(tmp_path: Path) -> None:
    from slm_training.dsl.schema import ExampleRecord, write_jsonl

    test_dir = tmp_path / "test"
    suite_dir = test_dir / "suites" / "held_out"
    suite_dir.mkdir(parents=True)
    long_prog = (
        'root = Stack([a, b, c], "column")\n'
        'a = TextContent(":a")\n'
        'b = TextContent(":b")\n'
        'c = Card([x, y])\n'
        'x = TextContent(":x")\n'
        'y = Button(":y")\n'
    )
    write_jsonl(
        suite_dir / "records.jsonl",
        [
            ExampleRecord(
                id="long",
                prompt="long",
                openui=long_prog,
                split="held_out",
                placeholders=[":a", ":b", ":x", ":y"],
            )
        ],
    )
    feas = evaluate_decode_feasibility(test_dir, canvas_cap=8)
    assert feas["pass"] is False
    assert feas["suites"]["held_out"]["feasible"] is False


def test_pick_constrained_token_admits_placeholder_subtokens() -> None:
    import torch

    from slm_training.models.grammar import pick_constrained_token
    from slm_training.models.tokenizer import OpenUITokenizer, quoted_placeholder_tokens

    text = 'hero = TextContent(":smoke.hero.title")\n'
    tokenizer = OpenUITokenizer.build([text])
    prefix = tokenizer.encode("hero = TextContent(", add_special=False)
    ph_ids = [
        tokenizer.token_to_id[t]
        for t in quoted_placeholder_tokens(":smoke.hero.title")
    ]
    for i, target_id in enumerate(ph_ids):
        logits = torch.zeros(tokenizer.vocab_size)
        logits[target_id] = 100.0
        choice = pick_constrained_token(
            logits,
            tokenizer,
            prefix + ph_ids[:i],
            prefer_structural=False,
            top_k=tokenizer.vocab_size,
        )
        assert choice == target_id, (
            f"placeholder step {i}: expected "
            f"{tokenizer.id_to_token[target_id]!r}, got "
            f"{tokenizer.id_to_token.get(choice)!r}"
        )


def test_stream_probe_accepts_closing_bracket() -> None:
    from slm_training.models.grammar import _stream_probe_ok
    from slm_training.models.tokenizer import OpenUITokenizer

    src = 'root = Stack([cta], "column")\ncta = Button(":cta")\n'
    tokenizer = OpenUITokenizer.build([src])
    ids = tokenizer.encode(src, add_special=False)
    # Find a closing paren in the gold sequence.
    close_id = tokenizer.token_to_id[")"]
    idx = ids.index(close_id)
    prefix = ids[:idx]
    assert _stream_probe_ok(tokenizer, prefix, close_id) is True
