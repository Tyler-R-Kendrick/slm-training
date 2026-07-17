# B3 capacity ladder — choice vs surface targets (SLM-23)

Depends on **B1** (the choice-sequence codec, SLM-42, owned on `main`:
`dsl/production_codec.py::encode_choices`/`decode_choices` +
`models/choice_tokenizer.py`; see
[iter-b1-choice-sequence-codec-20260717.md](iter-b1-choice-sequence-codec-20260717.md))
and **B2** (canonical-space alignment, SLM-22, merged). B3 is the direct
empirical test of the program's core hypothesis: *removing non-lexical
symbols lets a smaller model learn the grammar.*

Code: [`harnesses/experiments/choice_ladder.py`](../../src/slm_training/harnesses/experiments/choice_ladder.py).
Tests: `tests/test_harnesses/experiments/test_choice_ladder.py`.

## Instrument

Matched tiny models per width — one arm trained on lexer surface targets,
one on B1 choice-decision targets (`output_tokenizer="choice"`), same
records / steps / seed / held-out mask draw. It rides the existing TwoTower
owner (no parallel trainer) and produces the E1 bits-per-semantic-decision
quantity: each arm's held-out masked NLL renormalized by
(target tokens per program ÷ decisions per program), so both arms are
compared in **nats per semantic decision** regardless of tokenizer
byte-framing.

## B3 fixture ladder (measured 2026-07-17)

32 choice-codec-compatible fixture-v1 records, 8 held-out, 60 CPU steps,
lr 3e-4, seed 0, widths 16/32/64. JSON:
[choice-ladder-results-b3-20260717.json](choice-ladder-results-b3-20260717.json).

| d_model | target | tokens/prog | held-out **NLL/decision** |
| ---: | --- | ---: | ---: |
| 16 | lexer | 46.97 | 35.64 |
| 16 | choice | 21.75 | **17.84** |
| 32 | lexer | 46.97 | 43.73 |
| 32 | choice | 21.75 | **18.26** |
| 64 | lexer | 46.97 | 34.65 |
| 64 | choice | 21.75 | **19.14** |

At every width the choice arm spends roughly **half** the nats per semantic
decision on held-out programs, with 2.2× shorter targets — the direction the
externalized-syntax hypothesis predicts, measured on the canonical B1
tokenizer rather than asserted.

## Caveats (binding)

- Masked-NLL proxy — **not meaningful parse**, which stays the primary metric
  for any real B3 claim; this ladder measures learnability of the target
  representation, not generation quality.
- Constrained decode over choice vocabularies is B1 follow-up work (the
  surface DFA speaks surface lexemes); these rows decode unconstrained.
- Main's choice vocabulary is corpus-independent full-grammar, so the choice
  arm's embedding table is not smaller than a tiny-fixture lexer vocab — the
  load-bearing quantity is *target length* (decisions per program), not vocab
  size.
- The full ladder (`--matrix` rows, meaningful parse primary, frontier scale)
  remains future work.

## Honesty

Fixture-grade wiring + measurement only. No checkpoint promoted, no gate
touched, no ship claim.
