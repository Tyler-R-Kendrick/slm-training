#!/usr/bin/env python3
"""Compare compositional vs lexer-native sequence lengths over a corpus."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from slm_training.dsl.schema import load_jsonl
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, SymbolTable
from slm_training.models.tokenizer import OpenUITokenizer


def _pct(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def diagnose(records_path: Path) -> dict:
    records = load_jsonl(records_path)
    openuis = [r.openui for r in records]
    prompts = [r.prompt for r in records]
    compositional = OpenUITokenizer.build(prompts + openuis)
    lexer = DSLNativeTokenizer.build()

    comp_lens: list[int] = []
    lex_lens_sym: list[int] = []
    lex_lens_lit: list[int] = []
    for r in records:
        comp_lens.append(len(compositional.encode(r.openui, add_special=False)))
        table = SymbolTable.from_placeholders(list(r.placeholders or []))
        lex_lens_sym.append(
            len(
                lexer.encode(
                    r.openui,
                    add_special=False,
                    table=table,
                    use_symbol_table=True,
                )
            )
        )
        lex_lens_lit.append(
            len(
                lexer.encode(
                    r.openui,
                    add_special=False,
                    use_symbol_table=False,
                )
            )
        )

    def summary(name: str, lens: list[int], vocab: int) -> dict:
        return {
            "name": name,
            "n": len(lens),
            "vocab_size": vocab,
            "mean": round(statistics.fmean(lens), 2) if lens else 0.0,
            "p50": _pct(lens, 0.50),
            "p95": _pct(lens, 0.95),
            "max": max(lens) if lens else 0,
        }

    report = {
        "records": str(records_path),
        "compositional": summary("compositional_v2", comp_lens, compositional.vocab_size),
        "lexer_symtable": summary("lexer+symtable", lex_lens_sym, lexer.vocab_size),
        "lexer_literal": summary("lexer+literal", lex_lens_lit, lexer.vocab_size),
        "length_ratio_sym_vs_comp": (
            round(statistics.fmean(lex_lens_sym) / statistics.fmean(comp_lens), 3)
            if comp_lens and lex_lens_sym and statistics.fmean(comp_lens) > 0
            else None
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-records",
        type=Path,
        default=Path("outputs/train_data/v1/records.jsonl"),
    )
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Use src/slm_training/resources/train_seeds.jsonl instead of built corpus.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    path = Path("src/slm_training/resources/train_seeds.jsonl") if args.fixtures else args.train_records
    if not path.is_file():
        # Fall back to fixtures when corpus is missing.
        path = Path("src/slm_training/resources/train_seeds.jsonl")
    report = diagnose(path)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
